#!/usr/bin/env python3
"""LLMBenchmark: run EvalScope benchmarks through OpenAI-compatible APIs.

Usage:
    python bench.py run --model minimax --suite knowledge --profile smoke
    python bench.py run --model minimax --suite all --profile lite --include-extensions
    python bench.py prepare            # one-time: build the pinned SWE-bench Django dataset
    python bench.py report             # regenerate results/summary.md from the SQLite DB

Every dataset is one isolated EvalScope run.  Raw outputs stay under
``$LLMBENCH_DATA_ROOT/outputs``; aggregate scores are recorded in
``$LLMBENCH_DATA_ROOT/results.db`` (SQLite) and rendered to results/summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get('LLMBENCH_DATA_ROOT', '/ssd4/LLMBenchmark'))

SUITE_NAMES = ('knowledge', 'swe', 'vision')
PROFILE_NAMES = ('smoke', 'lite', 'full')

DJANGO_DATASET_NAME = 'swe_bench_verified_django1'
DJANGO_INSTANCE_ID = 'django__django-10097'

# Only images owned by SWE-bench harnesses are ever removed by --cleanup-images.
SWE_IMAGE_PREFIXES = ('swebench/', 'sweb.eval', 'sweb.env', 'sweap', 'jefzda/sweap')

EVALSCOPE_VERSION = None  # filled lazily


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def load_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict | None, override: dict | None) -> dict:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def expand_data_root(value, data_root: Path):
    if isinstance(value, str):
        return value.replace('${DATA_ROOT}', str(data_root))
    return value


def evalscope_version() -> str:
    global EVALSCOPE_VERSION
    if EVALSCOPE_VERSION is None:
        try:
            import evalscope
            EVALSCOPE_VERSION = getattr(evalscope, '__version__', 'unknown')
        except Exception:
            EVALSCOPE_VERSION = 'unknown'
    return EVALSCOPE_VERSION


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(REPO_DIR), 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ''


def setup_cache_env(data_root: Path) -> None:
    """Point EvalScope/HF caches at the data root.  Must run before evalscope import."""
    os.environ.setdefault('EVALSCOPE_CACHE', str(data_root / 'evalscope-cache'))
    os.environ.setdefault('HF_HOME', str(data_root / 'hf-cache'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(data_root / 'hf-cache' / 'datasets'))


def apply_compat_patches() -> None:
    """nltk >= 3.9 refuses ``edit_distance`` on inputs longer than 2000 chars and its
    pure-Python implementation cannot score full-page OCR in reasonable time.
    ``Levenshtein.distance`` computes the same plain edit distance in C."""
    try:
        import Levenshtein
        import nltk
        nltk.edit_distance = Levenshtein.distance  # used by OCRBench-v2 page_ocr_metric
        try:
            import importlib
            importlib.import_module('nltk.metrics.distance').edit_distance = Levenshtein.distance
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - only affects OCRBench-v2 scoring
        print(f'compat patch skipped: {exc}', file=sys.stderr)


# ---------------------------------------------------------------------------
# config resolution
# ---------------------------------------------------------------------------

def pick_datasets(suite_cfg: dict, include_extensions: bool, only: list[str] | None = None):
    datasets = suite_cfg.get('datasets') or {}
    picked = []
    for name, raw_spec in datasets.items():
        if only and name not in only:
            continue
        spec = raw_spec or {}
        if not spec.get('core', False) and not include_extensions:
            continue
        picked.append((name, spec))
    return picked


def resolve_limit(spec: dict, profiles: dict, profile: str, cli_limit: int | None):
    if cli_limit is not None:
        return cli_limit
    if spec.get('pinned'):
        return 1
    if profile == 'full':
        return spec.get('full_max_samples')
    profile_cfg = profiles.get(profile) or {}
    return profile_cfg.get('limit')


def resolve_batch_size(spec: dict, suite_cfg: dict, defaults: dict, cli_batch: int | None):
    for candidate in (cli_batch, spec.get('eval_batch_size'), suite_cfg.get('eval_batch_size'),
                      defaults.get('eval_batch_size')):
        if candidate is not None:
            return int(candidate)
    return 4


def build_task_config(*, model_name: str, model_cfg: dict, dataset: str, spec: dict,
                      suite_cfg: dict, defaults: dict, profiles: dict, profile: str,
                      data_root: Path, output_dir: Path, cli_limit=None, cli_batch=None) -> dict:
    api_key_env = model_cfg.get('api_key_env')
    api_key = os.environ.get(api_key_env, '') if api_key_env else model_cfg.get('api_key', '')
    if not api_key:
        raise SystemExit(f'model {model_name!r}: environment variable {api_key_env!r} is not set')

    generation_config = deep_merge(model_cfg.get('generation_config'), spec.get('generation_config'))
    task_cfg = {
        'model': model_cfg['model_id'],
        'model_id': model_cfg.get('report_id') or model_cfg['model_id'],
        'api_url': model_cfg['api_url'],
        'api_key': api_key,
        'eval_type': model_cfg.get('eval_type', 'openai_api'),
        'datasets': [dataset],
        'dataset_dir': str(data_root / 'datasets'),
        'generation_config': generation_config,
        'limit': resolve_limit(spec, profiles, profile, cli_limit),
        'eval_batch_size': resolve_batch_size(spec, suite_cfg, defaults, cli_batch),
        'work_dir': str(output_dir),
        'no_timestamp': True,
        'seed': 42,
        'collect_perf': True,
        # Record per-sample errors as `errored` instead of aborting a long run.
        'ignore_errors': True,
    }

    dataset_args = dict(spec.get('dataset_args') or {})
    if spec.get('local_path'):
        dataset_args['local_path'] = expand_data_root(spec['local_path'], data_root)
    if spec.get('extra_params'):
        dataset_args['extra_params'] = deep_merge(dataset_args.get('extra_params'), spec['extra_params'])
    if dataset_args:
        task_cfg['dataset_args'] = {dataset: dataset_args}

    if spec.get('agent_config'):
        task_cfg['agent_config'] = dict(spec['agent_config'])

    sandbox = spec.get('sandbox') or suite_cfg.get('sandbox')
    if sandbox:
        task_cfg['sandbox'] = dict(sandbox)

    return task_cfg


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    run_group       TEXT,
    model           TEXT,
    suite           TEXT,
    profile         TEXT,
    dataset         TEXT,
    status          TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    num_requested   INTEGER,
    num_succeeded   INTEGER,
    num_errored     INTEGER,
    incomplete      INTEGER,
    evalscope_version TEXT,
    git_commit      TEXT,
    output_dir      TEXT,
    config_json     TEXT,
    perf_json       TEXT,
    error           TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id     TEXT,
    dataset    TEXT,
    metric     TEXT,
    category   TEXT,
    subset     TEXT,
    num        INTEGER,
    score      REAL,
    macro_score REAL,
    PRIMARY KEY (run_id, metric, category, subset)
);
CREATE TABLE IF NOT EXISTS run_samples (
    run_id     TEXT,
    dataset    TEXT,
    sample_ids TEXT,
    PRIMARY KEY (run_id, dataset)
);
"""


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def db_start_run(conn: sqlite3.Connection, record: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_id, run_group, model, suite, profile, dataset, status, started_at,
            evalscope_version, git_commit, output_dir, config_json)
           VALUES (:run_id, :run_group, :model, :suite, :profile, :dataset, 'running',
                   :started_at, :evalscope_version, :git_commit, :output_dir, :config_json)""",
        record,
    )
    conn.commit()


def db_finish_run(conn: sqlite3.Connection, run_id: str, *, status: str, error: str | None,
                  execution_summary: dict | None, perf_metrics: dict | None) -> None:
    summary = execution_summary or {}
    conn.execute(
        """UPDATE runs SET status=?, finished_at=?, num_requested=?, num_succeeded=?,
           num_errored=?, incomplete=?, perf_json=?, error=? WHERE run_id=?""",
        (status, now_iso(), summary.get('requested'), summary.get('succeeded'),
         summary.get('errored'), 1 if summary.get('incomplete') else 0,
         json.dumps(perf_metrics, ensure_ascii=False) if perf_metrics else None,
         error, run_id),
    )
    conn.commit()


def db_record_metrics(conn: sqlite3.Connection, run_id: str, dataset: str, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO metrics
           (run_id, dataset, metric, category, subset, num, score, macro_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(run_id, dataset, r['metric'], r['category'], r['subset'], r['num'], r['score'], r['macro_score'])
         for r in rows],
    )
    conn.commit()


def db_record_samples(conn: sqlite3.Connection, run_id: str, dataset: str, sample_ids) -> None:
    conn.execute(
        'INSERT OR REPLACE INTO run_samples (run_id, dataset, sample_ids) VALUES (?, ?, ?)',
        (run_id, dataset, json.dumps(sample_ids, ensure_ascii=False) if sample_ids else None),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# EvalScope output parsing
# ---------------------------------------------------------------------------

def find_report(output_dir: Path, dataset: str) -> dict | None:
    reports_dir = output_dir / 'reports'
    if not reports_dir.exists():
        return None
    for path in sorted(reports_dir.rglob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(data, dict) and data.get('metrics') is not None and 'dataset_name' in data:
            return data
    return None


def metric_rows(report: dict) -> list[dict]:
    rows = []
    for metric in report.get('metrics') or []:
        name = (metric.get('identity') or {}).get('name') or metric.get('legacy_name') or 'metric'
        rows.append({
            'metric': name, 'category': '', 'subset': '',
            'num': metric.get('num') or 0,
            'score': float(metric.get('score') or 0.0),
            'macro_score': float(metric.get('macro_score') or 0.0),
        })
        for category in metric.get('categories') or []:
            cat_name = '|'.join(category.get('name') or [])
            rows.append({
                'metric': name, 'category': cat_name, 'subset': '',
                'num': category.get('num') or 0,
                'score': float(category.get('score') or 0.0),
                'macro_score': float(category.get('macro_score') or 0.0),
            })
            for subset in category.get('subsets') or []:
                rows.append({
                    'metric': name, 'category': cat_name, 'subset': subset.get('name') or '',
                    'num': subset.get('num') or 0,
                    'score': float(subset.get('score') or 0.0),
                    'macro_score': 0.0,
                })
    return rows


def sample_ids_from_outputs(output_dir: Path, dataset: str) -> dict | None:
    """Return ``{subset: [sample ids]}`` for the samples actually evaluated.

    EvalScope writes one ``<dataset>_<subset>.jsonl`` per subset under
    ``reviews/`` and ``predictions/``; sample ids restart per subset, so the
    subset name is part of the sample identity.
    """
    per_subset: dict[str, set] = {}

    def subset_from_stem(stem: str) -> str:
        prefix = f'{dataset}_'
        return stem[len(prefix):] if stem.startswith(prefix) else stem

    def collect(data, ids):
        if not isinstance(data, dict):
            return
        sample_id = data.get('sample_id', data.get('index'))
        if sample_id is None:
            score = data.get('sample_score')
            if isinstance(score, dict):
                sample_id = score.get('sample_id')
        if sample_id is not None:
            ids.add(str(sample_id))

    for folder in ('reviews', 'predictions'):
        root = output_dir / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.suffix == '.jsonl':
                ids = set()
                with open(path, encoding='utf-8') as handle:
                    for line in handle:
                        try:
                            collect(json.loads(line), ids)
                        except json.JSONDecodeError:
                            continue
                if ids:
                    per_subset.setdefault(subset_from_stem(path.stem), set()).update(ids)
            elif path.suffix == '.json':
                ids = set()
                try:
                    collect(json.loads(path.read_text(encoding='utf-8')), ids)
                except Exception:
                    continue
                if ids:
                    per_subset.setdefault(subset_from_stem(path.stem), set()).update(ids)
    if not per_subset:
        return None
    return {
        subset: sorted(ids, key=lambda value: (len(value), value))
        for subset, ids in sorted(per_subset.items())
    }


# ---------------------------------------------------------------------------
# summary.md from SQLite
# ---------------------------------------------------------------------------

def build_summary(conn: sqlite3.Connection, data_root: Path) -> str:
    latest = conn.execute(
        """
        SELECT model, dataset, metric, category, subset, num, score, macro_score,
               profile, finished_at, run_id, num_requested, num_succeeded, num_errored
        FROM (
            SELECT m.*, r.model, r.dataset, r.profile, r.finished_at, r.run_id AS run_id,
                   r.num_requested, r.num_succeeded, r.num_errored,
                   ROW_NUMBER() OVER (
                       PARTITION BY r.model, m.dataset, m.metric, m.category, m.subset
                       ORDER BY r.finished_at DESC
                   ) AS rn
            FROM metrics m JOIN runs r ON r.run_id = m.run_id
            WHERE r.status = 'success' AND m.category = '' AND m.subset = ''
        )
        WHERE rn = 1
        ORDER BY model, dataset, metric
        """
    ).fetchall()

    lines = [
        '# LLM Benchmark Summary',
        '',
        f'Generated: {now_iso()}  ',
        f'Data root: `{data_root}`  ',
        f'EvalScope: {evalscope_version()}',
        '',
    ]
    if not latest:
        lines += ['_No successful runs recorded yet._', '']
    current_model = None
    for row in latest:
        if row['model'] != current_model:
            current_model = row['model']
            lines += [f'## {current_model}', '',
                      '| Dataset | Metric | Score | Num | Done | Profile | Finished | Run |',
                      '|---|---|---:|---:|---|---|---|---|']
        done = f"{row['num_succeeded']}/{row['num_requested']}"
        if row['num_errored']:
            done += f" ({row['num_errored']} err)"
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['score']:.4f} | {row['num']} | {done} | "
            f"{row['profile']} | {row['finished_at']} | `{row['run_id']}` |"
        )
    lines.append('')

    failed = conn.execute(
        """SELECT run_id, model, dataset, status, error FROM runs
           WHERE status != 'success' ORDER BY started_at DESC"""
    ).fetchall()
    if failed:
        lines += ['## In-progress / failed runs', '', '| Run | Model | Dataset | Status | Error |', '|---|---|---|---|---|']
        for row in failed:
            error = (row['error'] or '').replace('\n', ' ')[:160].replace('|', '\\|')
            lines.append(f"| `{row['run_id']}` | {row['model']} | {row['dataset']} | {row['status']} | {error} |")
        lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    data_root = Path(args.data_root)
    config_dir = Path(args.config_dir)
    models_cfg = load_yaml(config_dir / 'models.yaml').get('models') or {}
    suites_doc = load_yaml(config_dir / 'suites.yaml')
    suites_cfg = suites_doc.get('suites') or {}
    defaults = suites_doc.get('defaults') or {}
    profiles = suites_doc.get('profiles') or {}

    model_names = list(models_cfg) if args.model == ['all'] else args.model
    for name in model_names:
        if name not in models_cfg:
            raise SystemExit(f'unknown model {name!r}; known: {", ".join(models_cfg)}')

    suite_names = list(SUITE_NAMES) if args.suite == 'all' else [args.suite]
    if args.profile not in PROFILE_NAMES:
        raise SystemExit(f'unknown profile {args.profile!r}')
    if args.datasets:
        known = {name for cfg in suites_cfg.values() for name in (cfg.get('datasets') or {})}
        unknown = [name for name in args.datasets if name not in known]
        if unknown:
            raise SystemExit(f'unknown dataset(s): {", ".join(unknown)}')

    setup_cache_env(data_root)
    outputs_root = data_root / 'outputs'
    conn = connect_db(Path(args.db))
    init_db(conn)

    run_group = time.strftime('%Y%m%d_%H%M%S')
    any_success = False
    for model_name in model_names:
        model_cfg = models_cfg[model_name]
        for suite_name in suite_names:
            suite_cfg = suites_cfg[suite_name]
            for dataset, spec in pick_datasets(suite_cfg, args.include_extensions, args.datasets):
                group = f'{run_group}_{model_name}_{suite_name}_{args.profile}'
                run_id = f'{group}__{dataset}'
                output_dir = outputs_root / group / dataset
                task_cfg = build_task_config(
                    model_name=model_name, model_cfg=model_cfg, dataset=dataset, spec=spec,
                    suite_cfg=suite_cfg, defaults=defaults, profiles=profiles, profile=args.profile,
                    data_root=data_root, output_dir=output_dir,
                    cli_limit=args.limit, cli_batch=args.batch_size,
                )
                safe_cfg = dict(task_cfg)
                safe_cfg['api_key'] = '***'
                if args.dry_run:
                    print(f'--- {run_id}')
                    print(json.dumps(safe_cfg, ensure_ascii=False, indent=2))
                    continue
                print(f'=== {run_id} (limit={task_cfg["limit"]}, batch={task_cfg["eval_batch_size"]})')
                ok = run_one(conn, run_id=run_id, run_group=run_group, model=model_name,
                             suite=suite_name, profile=args.profile, dataset=dataset,
                             task_cfg=task_cfg, safe_cfg=safe_cfg, output_dir=output_dir)
                any_success = any_success or ok

    if args.dry_run:
        return 0
    if args.cleanup_images:
        cleanup_swe_images()
    summary_path = REPO_DIR / 'results' / 'summary.md'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(conn, data_root), encoding='utf-8')
    print(f'summary written to {summary_path}')
    return 0 if any_success else 1


def run_one(conn, *, run_id, run_group, model, suite, profile, dataset, task_cfg, safe_cfg, output_dir) -> bool:
    db_start_run(conn, {
        'run_id': run_id, 'run_group': run_group, 'model': model, 'suite': suite,
        'profile': profile, 'dataset': dataset, 'started_at': now_iso(),
        'evalscope_version': evalscope_version(), 'git_commit': git_commit(),
        'output_dir': str(output_dir), 'config_json': json.dumps(safe_cfg, ensure_ascii=False, default=str),
    })
    started = time.time()
    error = None
    try:
        apply_compat_patches()
        from evalscope import run_task
        run_task(task_cfg)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)

    report = find_report(output_dir, dataset)
    if report:
        db_record_metrics(conn, run_id, dataset, metric_rows(report))
    db_record_samples(conn, run_id, dataset, sample_ids_from_outputs(output_dir, dataset))

    if error is not None:
        db_finish_run(conn, run_id, status='failed', error=error, execution_summary=None, perf_metrics=None)
        print(f'!!! {run_id} FAILED after {time.time() - started:.0f}s')
        return False

    execution_summary = report.get('execution_summary') if report else None
    perf_metrics = report.get('perf_metrics') if report else None
    db_finish_run(conn, run_id, status='success', error=None,
                  execution_summary=execution_summary, perf_metrics=perf_metrics)
    if report:
        top = [r for r in metric_rows(report) if not r['category']]
        print(f'+++ {run_id} done in {time.time() - started:.0f}s: '
              + ', '.join(f"{r['metric']}={r['score']:.4f}" for r in top))
    else:
        print(f'+++ {run_id} done in {time.time() - started:.0f}s (no report file found)')
    return True


def cmd_prepare(args) -> int:
    import datasets

    data_root = Path(args.data_root)
    out_dir = data_root / 'pinned' / DJANGO_DATASET_NAME
    if out_dir.exists() and not args.force:
        print(f'{out_dir} already exists (use --force to rebuild)')
        return 0
    ds = datasets.load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    pinned = ds.filter(lambda record: record['instance_id'] == DJANGO_INSTANCE_ID)
    if len(pinned) != 1:
        raise SystemExit(f'expected exactly one {DJANGO_INSTANCE_ID}, found {len(pinned)}')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A load_dataset-compatible layout (save_to_disk is rejected by load_dataset).
    pinned.to_parquet(str(out_dir / 'test-00000-of-00001.parquet'))
    print(f'pinned dataset ({DJANGO_INSTANCE_ID}) written to {out_dir}')
    return 0


def cmd_report(args) -> int:
    conn = connect_db(Path(args.db))
    init_db(conn)
    out_path = Path(args.out) if args.out else REPO_DIR / 'results' / 'summary.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_summary(conn, Path(args.data_root)), encoding='utf-8')
    print(f'wrote {out_path}')
    return 0


def cleanup_swe_images() -> None:
    try:
        out = subprocess.check_output(
            ['docker', 'image', 'ls', '--format', '{{.Repository}}:{{.Tag}}'],
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()
    except Exception as exc:
        print(f'cleanup skipped: {exc}', file=sys.stderr)
        return
    targets = [name for name in out if name.startswith(SWE_IMAGE_PREFIXES)]
    if not targets:
        print('cleanup: no SWE-bench images found')
        return
    print(f'cleanup: removing {len(targets)} SWE-bench image(s)')
    subprocess.run(['docker', 'rmi', *targets], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def add_common(p):
        p.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT),
                       help=f'data/output root (default: {DEFAULT_DATA_ROOT})')
        p.add_argument('--config-dir', default=str(REPO_DIR / 'configs'), help='config directory')
        p.add_argument('--db', default=None, help='SQLite path (default: <data-root>/results.db)')

    run_p = sub.add_parser('run', help='run benchmarks')
    add_common(run_p)
    run_p.add_argument('--model', action='append', default=None,
                       help='model name from models.yaml (repeatable; "all" for every model)')
    run_p.add_argument('--suite', choices=(*SUITE_NAMES, 'all'), default='all')
    run_p.add_argument('--profile', choices=PROFILE_NAMES, default='lite')
    run_p.add_argument('--datasets', nargs='+', default=None, help='restrict to these dataset names')
    run_p.add_argument('--limit', type=int, default=None, help='override the profile limit')
    run_p.add_argument('--batch-size', type=int, default=None, help='override eval_batch_size')
    run_p.add_argument('--include-extensions', action='store_true',
                       help='also run datasets marked core: false')
    run_p.add_argument('--cleanup-images', action='store_true',
                       help='after the run, remove only SWE-bench images')
    run_p.add_argument('--dry-run', action='store_true', help='print configs without calling the model')

    prep_p = sub.add_parser('prepare', help='build the pinned SWE-bench Django dataset')
    add_common(prep_p)
    prep_p.add_argument('--force', action='store_true')

    rep_p = sub.add_parser('report', help='regenerate results/summary.md from the DB')
    add_common(rep_p)
    rep_p.add_argument('--out', default=None, help='output markdown path')

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.db is None:
        args.db = str(Path(args.data_root) / 'results.db')
    if args.command == 'run':
        if args.model is None:
            args.model = ['minimax']
        return cmd_run(args)
    if args.command == 'prepare':
        return cmd_prepare(args)
    if args.command == 'report':
        return cmd_report(args)
    parser.error(f'unknown command {args.command}')


if __name__ == '__main__':
    sys.exit(main())
