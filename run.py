#!/usr/bin/env python3
"""跑 EvalScope benchmark（任意 OpenAI 兼容端点），把分数记进 SQLite。

    python run.py run --suite reasoning --model minimax
    python run.py run --suite all --model minimax --only gpqa_diamond,ifeval --dry-run
    python run.py show                       # 跨模型对比表
    python run.py selftest                   # 收割逻辑的自检
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'results' / 'bench.db'
EVALSCOPE_BIN = Path(sys.executable).parent / 'evalscope'
# 平均输出 token 超过上限的这个比例就认为可能被截断（evalscope 不暴露 finish_reason）
TRUNC_RATIO = 0.9

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, started_at TEXT, model TEXT, base_url TEXT, suite TEXT,
  benchmark TEXT, limit_n INT, max_tokens INT, generation_config TEXT, dataset_args TEXT,
  work_dir TEXT, status TEXT, duration_s REAL, error TEXT);
CREATE TABLE IF NOT EXISTS scores(
  run_id INT, dataset TEXT, subset TEXT, metric TEXT, score REAL, num INT,
  is_primary INT, display TEXT, avg_latency REAL, avg_out_tokens REAL, truncation_risk INT);
CREATE TABLE IF NOT EXISTS samples(
  run_id INT, dataset TEXT, subset TEXT, idx INT, input_hash TEXT);
"""


def load_yaml(rel):
    return yaml.safe_load((ROOT / rel).read_text())


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute('PRAGMA journal_mode=WAL')  # 允许多个 run 进程同时读写
    conn.executescript(SCHEMA)
    for column in ('is_primary', 'display'):  # 早期版本建的库补列
        try:
            conn.execute(f'ALTER TABLE scores ADD COLUMN {column} INT')
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


# ---------------------------------------------------------------- 收割

def parse_report(report, max_tokens):
    """evalscope 的 report json -> (分数行, 性能摘要)。只留质量指标，不收 diagnostic 类。"""
    perf = report.get('perf_metrics', {}).get('summary', {})
    latency = (perf.get('latency') or {}).get('mean')
    out_tokens = ((perf.get('usage') or {}).get('output_tokens') or {}).get('mean')
    trunc = int(bool(out_tokens and max_tokens and out_tokens >= TRUNC_RATIO * max_tokens))
    primary = (report.get('primary_metric_identity') or {}).get('name')
    shared = dict(avg_latency=latency, avg_out_tokens=out_tokens, truncation_risk=trunc)

    rows = []
    for metric in report.get('metrics', []):
        semantics = metric.get('semantics') or {}
        if semantics.get('kind') != 'quality':
            continue
        name = metric['identity']['name']
        head = dict(metric=name, is_primary=int(name == primary), display=semantics.get('display_kind'), **shared)
        rows.append(dict(subset=None, score=metric.get('score'), num=metric.get('num'), **head))
        for category in metric.get('categories') or []:
            for sub in category.get('subsets') or []:
                # 单 subset 的数据集，subset 行和汇总行完全一样，不重复入库
                if (sub.get('num'), sub.get('score')) == (metric.get('num'), metric.get('score')):
                    continue
                rows.append(dict(subset=sub['name'], score=sub.get('score'), num=sub.get('num'), **head))
    return rows


def input_hashes(pred_file):
    """每道题的输入指纹，用来验证不同模型跑的是同一批题。"""
    # 只能按 \n 切：模型输出里可能含 U+2028 等字符，str.splitlines() 会在字符串中间断开
    for line in pred_file.read_text().split('\n'):
        if not line.strip():
            continue
        rec = json.loads(line)
        blob = json.dumps(rec.get('messages'), ensure_ascii=False, sort_keys=True)
        yield rec.get('index'), hashlib.sha1(blob.encode()).hexdigest()[:12]


def harvest(conn, run_id, out_dir, dataset, max_tokens):
    reports = sorted((out_dir / 'reports').rglob('*.json'))
    if not reports:
        return 0
    for report_path in reports:
        report = json.loads(report_path.read_text())
        name = report.get('dataset_name', dataset)
        for row in parse_report(report, max_tokens):
            conn.execute(
                'INSERT INTO scores(run_id,dataset,subset,metric,score,num,is_primary,display,avg_latency,'
                'avg_out_tokens,truncation_risk) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (run_id, name, row['subset'], row['metric'], row['score'], row['num'], row['is_primary'],
                 row['display'], row['avg_latency'], row['avg_out_tokens'], row['truncation_risk']))
    for pred_file in (out_dir / 'predictions').rglob('*.jsonl'):
        for idx, digest in input_hashes(pred_file):
            conn.execute('INSERT INTO samples(run_id,dataset,subset,idx,input_hash) VALUES(?,?,?,?,?)',
                         (run_id, name, pred_file.stem, idx, digest))
    conn.commit()
    return len(reports)


# ---------------------------------------------------------------- 执行

def resolve_dataset_args(args):
    """把配置里的相对路径变成绝对路径，evalscope 的 loader 认 os.path.exists。"""
    out = json.loads(json.dumps(args)) if args else args
    if out and 'local_path' in out:
        out['local_path'] = str((ROOT / out['local_path']).resolve())
    return out


def build_cmd(cfg, bench, default_max_tokens, work_dir, limit=None, use_cache=None):
    gen_cfg = dict(cfg.get('generation_config') or {})
    gen_cfg.setdefault('max_tokens', bench.get('max_tokens') or default_max_tokens)
    cmd = [
        str(EVALSCOPE_BIN), 'eval',
        '--model', cfg['model'],
        '--api-url', cfg['base_url'],
        '--api-key', os.environ[cfg['api_key_env']],
        '--eval-type', cfg.get('eval_type', 'openai_api'),
        '--datasets', bench['name'],
        '--eval-batch-size', str(cfg.get('batch_size', 8)),
        '--timeout', str(cfg.get('timeout', 3600)),
        '--generation-config', json.dumps(gen_cfg),
        '--work-dir', str(work_dir),
    ]
    # 数据集会额外存一份到 evalscope 自己的 cache，位置得跟数据集 cache 放同一块盘
    if os.environ.get('EVALSCOPE_DATASET_DIR'):
        cmd += ['--dataset-dir', os.environ['EVALSCOPE_DATASET_DIR']]
    cmd += ['--limit', str(limit or bench.get('limit'))]
    cmd += list(bench.get('extra_args') or [])  # 逃生口：清单里可以直接写任意 evalscope 参数
    if bench.get('agent_config'):
        cmd += ['--agent-config', json.dumps(bench['agent_config'])]
    if bench.get('dataset_args'):
        cmd += ['--dataset-args', json.dumps({bench['name']: resolve_dataset_args(bench['dataset_args'])})]
    if use_cache:
        cmd += ['--use-cache', str(use_cache)]
    return cmd, gen_cfg


def latest_output(work_dir):
    """evalscope 会在 work-dir 下面再套一层时间戳，找最新的那份产出。"""
    reports = [p.parent for p in work_dir.rglob('reports') if p.is_dir()]
    return max(reports, key=lambda p: p.stat().st_mtime) if reports else None


def redact_str(text):
    # 兼容 shell 拼接和 subprocess 的 list repr 两种形态
    return re.sub(r'(--api-key[^\w]{0,4})[^\s\'",]+', r'\1***', text)


def redact(cmd):
    """命令行里带明文 key，只给人看脱敏版。"""
    out, skip = [], False
    for arg in cmd:
        if skip:
            out.append('***')
            skip = False
        else:
            skip = arg == '--api-key'
            out.append(arg)
    return redact_str(' '.join(out))


def cmd_run(args):
    bench_cfg = load_yaml('benchmarks.yaml')
    models = load_yaml('configs/models.yaml')['models']
    if args.model not in models:
        sys.exit(f'未知模型 {args.model}，可选: {", ".join(models)}')

    wanted = [s.strip() for s in args.only.split(',')] if args.only else None
    jobs = [(suite, b) for suite, names in bench_cfg['suites'].items() if args.suite in (suite, 'all')
            for b in names if not wanted or b['name'] in wanted]
    if not jobs:
        sys.exit('没有匹配的 benchmark，用 --list 看看可选清单')

    cfg = models[args.model]
    work_root = ROOT / 'outputs' / args.model
    work_root.mkdir(parents=True, exist_ok=True)
    conn = connect()

    for suite, bench in jobs:
        work_dir = work_root / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{bench["name"]}'
        cmd, gen_cfg = build_cmd(cfg, bench, bench_cfg['max_tokens'], work_dir, limit=args.limit,
                                  use_cache=args.use_cache)
        if args.dry_run:
            print(redact(cmd))
            continue

        print(f'\n=== {bench["name"]} ({suite}) ===\n{redact(cmd)}', flush=True)
        cur = conn.execute(
            'INSERT INTO runs(started_at,model,base_url,suite,benchmark,limit_n,max_tokens,generation_config,'
            'dataset_args,work_dir,status) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (now(), cfg['model'], cfg['base_url'], suite, bench['name'],
             args.limit or bench.get('limit'), gen_cfg['max_tokens'], json.dumps(gen_cfg),
             json.dumps(bench.get('dataset_args')), str(work_dir), 'running'))
        conn.commit()  # 尽早释放锁，允许另一个 run 进程同时写库
        run_id, started = cur.lastrowid, time.time()
        status, error = 'ok', None
        try:
            subprocess.run(cmd, check=True, env={**os.environ, 'PYTHONPATH': str(ROOT / 'compat')})
            # --use-cache 复用旧产出时不会在新目录写报告，分数要从旧目录收
            out_dir = latest_output(work_dir) or (Path(args.use_cache) if args.use_cache else None)
            if not out_dir or not harvest(conn, run_id, out_dir, bench['name'], gen_cfg['max_tokens']):
                raise RuntimeError(f'没有产出报告，结果在 {out_dir}')
            conn.execute('UPDATE runs SET work_dir=? WHERE id=?', (str(out_dir), run_id))
        except Exception as exc:  # 单个 benchmark 失败不影响后面的
            message = f'{type(exc).__name__}: {exc}'[:2000]
            status, error = 'failed', redact_str(message)
        conn.execute('UPDATE runs SET status=?, duration_s=?, error=? WHERE id=?',
                     (status, round(time.time() - started, 1), error, run_id))
        conn.commit()
        print(f'--- {bench["name"]}: {status} ({round(time.time() - started, 1)}s) {error or ""}')


# ---------------------------------------------------------------- 展示

def render_table(rows):
    """rows: (dataset, metric, subset, num, score, trunc, display, model) -> markdown 表格"""
    models = sorted({r[7] for r in rows})
    keys = sorted({(r[0], r[1], r[2], r[3]) for r in rows}, key=lambda k: tuple(x or '' for x in k))
    lines = [f'| benchmark | metric | n | ' + ' | '.join(models) + ' |',
             '|---|---|---|' + '---|' * len(models)]
    for dataset, metric, subset, num in keys:
        cells = {r[7]: r for r in rows if (r[0], r[1], r[2], r[3]) == (dataset, metric, subset, num)}
        values = []
        for model in models:
            cell = cells.get(model)
            if not cell:
                values.append('-')
                continue
            score = 'n/a' if cell[4] is None else (
                f'{cell[4] * 100:.1f}%' if cell[6] == 'percent' else f'{cell[4]:g}')
            values.append(score + (' ⚠截断' if cell[5] else ''))
        label = f'{dataset} / {subset}' if subset else dataset
        lines.append(f'| {label} | {metric} | {num} | ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def cmd_show(args):
    conn = connect()
    rows = conn.execute(
        'SELECT s.dataset, s.metric, s.subset, s.num, s.score, s.truncation_risk, s.display, r.model '
        'FROM scores s JOIN runs r ON r.id = s.run_id '
        'WHERE r.status = ? ' + ('' if args.subsets else 'AND s.subset IS NULL AND s.is_primary = 1 ') +
        'ORDER BY s.dataset, s.metric, s.subset, r.model', ('ok',)).fetchall()

    table = render_table(rows)
    print(table)
    if args.save:
        Path(args.save).write_text(table + '\n')
        print(f'\n已写入 {args.save}')


def cmd_reharvest(args):
    """从 outputs/ 里已跑完的产出重新入库。改了收割逻辑、或上次收割失败时用，不花钱。"""
    conn = connect()
    for run_id, work_dir, dataset, max_tokens in conn.execute(
            'SELECT id, work_dir, benchmark, max_tokens FROM runs WHERE status = ?', ('ok',)).fetchall():
        out_dir = Path(work_dir or '')
        if not out_dir.is_dir():
            print(f'run {run_id} ({dataset}) 产出目录不存在，跳过')
            continue
        conn.execute('DELETE FROM scores WHERE run_id = ?', (run_id,))
        conn.execute('DELETE FROM samples WHERE run_id = ?', (run_id,))
        harvest(conn, run_id, out_dir, dataset, max_tokens or 0)
        print(f'run {run_id} ({dataset}) 重新入库 {out_dir}')
    conn.commit()


def cmd_list(args):
    cfg = load_yaml('benchmarks.yaml')
    for suite, benches in cfg['suites'].items():
        print(f'{suite}: ' + ', '.join(f'{b["name"]}({b["limit"]})' for b in benches))


# ---------------------------------------------------------------- 自检

def selftest():
    import tempfile
    fake = {
        'dataset_name': 'demo',
        'primary_metric_identity': {'name': 'accuracy'},
        'metrics': [{
            'identity': {'name': 'accuracy'}, 'num': 4, 'score': 0.75,
            'semantics': {'kind': 'quality', 'display_kind': 'percent'},
            'categories': [{'name': ['main'], 'num': 4, 'score': 0.75,
                            'subsets': [{'name': 'sub-a', 'num': 2, 'score': 0.5}]}],
        }, {
            'identity': {'name': 'avg_len'}, 'num': 4, 'score': 1234.0,  # diagnostic，不该入库
            'semantics': {'kind': 'diagnostic', 'display_kind': 'number'},
        }],
        'perf_metrics': {'summary': {'latency': {'mean': 1.5},
                                     'usage': {'output_tokens': {'mean': 32000}}}},
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / 'reports' / 'm').mkdir(parents=True)
        (out / 'reports' / 'm' / 'demo.json').write_text(json.dumps(fake))
        (out / 'predictions' / 'm').mkdir(parents=True)
        (out / 'predictions' / 'm' / 'demo_main.jsonl').write_text(
            json.dumps({'index': 0, 'messages': [{'content': 'q'}]}) + '\n')
        conn = sqlite3.connect(':memory:')
        conn.executescript(SCHEMA)
        assert harvest(conn, 1, out, 'demo', 32768) == 1
        scores = conn.execute('SELECT subset, score, num, is_primary, display, truncation_risk '
                              'FROM scores ORDER BY subset').fetchall()
        assert scores == [(None, 0.75, 4, 1, 'percent', 1), ('sub-a', 0.5, 2, 1, 'percent', 1)], scores
        assert conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0] == 1
    # 汇总行的 subset 是 None，排序和渲染都不能因此炸
    table = render_table([('demo', 'accuracy', None, 4, 0.75, 0, 'percent', 'm'),
                          ('demo', 'accuracy', 'sub-a', 2, 0.5, 0, 'percent', 'm'),
                          ('demo', 'accuracy', None, 4, 0.75, 1, 'percent', 'm2')])
    assert '| 75.0% ⚠截断 |' in table, table
    print('selftest ok')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='跑 benchmark 并入库')
    p_run.add_argument('--suite', default='all', help='reasoning / coding / vision / all')
    p_run.add_argument('--only', help='逗号分隔，只跑清单里的某几个')
    p_run.add_argument('--model', required=True, help='configs/models.yaml 里的 key')
    p_run.add_argument('--limit', type=int, help='覆盖配置里的 limit')
    p_run.add_argument('--use-cache', help='复用之前某次产出目录里的 prediction（如 outputs/minimax/xxx）')
    p_run.add_argument('--dry-run', action='store_true', help='只打印命令，不花钱')
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser('show', help='从库里出跨模型对比表')
    p_show.add_argument('--subsets', action='store_true', help='连各 subset 明细一起显示')
    p_show.add_argument('--save', help='把表格另存为 markdown')
    p_show.set_defaults(func=cmd_show)

    sub.add_parser('list', help='列出可选的 benchmark').set_defaults(func=cmd_list)
    sub.add_parser('reharvest', help='从已有产出重新入库（不调 API）').set_defaults(func=cmd_reharvest)
    sub.add_parser('selftest', help='自检收割逻辑').set_defaults(func=lambda a: selftest())

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
