"""Small self-checks for bench.py.  Run with: python tests/test_bench.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench  # noqa: E402

PROFILES = {'smoke': {'limit': 5}, 'lite': {'limit': 20}, 'full': {'limit': None}}


def test_resolve_limit():
    assert bench.resolve_limit({}, PROFILES, 'smoke', None) == 5
    assert bench.resolve_limit({}, PROFILES, 'lite', None) == 20
    assert bench.resolve_limit({}, PROFILES, 'full', None) is None
    assert bench.resolve_limit({'full_max_samples': 500}, PROFILES, 'full', None) == 500
    assert bench.resolve_limit({}, PROFILES, 'lite', 7) == 7
    assert bench.resolve_limit({'pinned': True}, PROFILES, 'full', None) == 1


def test_build_task_config():
    os.environ['BENCH_TEST_KEY'] = 'secret'
    data_root = Path(tempfile.mkdtemp())
    model_cfg = {
        'model_id': 'some-model',
        'api_url': 'http://localhost:8000/v1',
        'api_key_env': 'BENCH_TEST_KEY',
        'generation_config': {'max_tokens': 32768, 'temperature': 0.0},
    }
    common = dict(model_name='m', model_cfg=model_cfg, suite_cfg={}, defaults={'eval_batch_size': 4},
                  profiles=PROFILES, data_root=data_root, output_dir=data_root / 'out')

    cfg = bench.build_task_config(dataset='gpqa_diamond', spec={}, profile='smoke', **common)
    assert cfg['limit'] == 5
    assert cfg['api_key'] == 'secret'
    assert cfg['eval_batch_size'] == 4
    assert cfg['work_dir'] == str(data_root / 'out')

    spec = {
        'pinned': True,
        'local_path': '${DATA_ROOT}/pinned/ds',
        'extra_params': {'build_docker_images': False},
        'dataset_args': {'action_hint': 'x'},
        'generation_config': {'temperature': 0.7},
        'eval_batch_size': 1,
        'agent_config': {'mode': 'native', 'max_steps': 250},
    }
    cfg = bench.build_task_config(dataset='swe_bench_verified_agentic', spec=spec, profile='full', **common)
    args = cfg['dataset_args']['swe_bench_verified_agentic']
    assert cfg['limit'] == 1
    assert args['local_path'] == str(data_root / 'pinned' / 'ds')
    assert args['extra_params']['build_docker_images'] is False
    assert args['action_hint'] == 'x'
    assert cfg['generation_config']['temperature'] == 0.7
    assert cfg['generation_config']['max_tokens'] == 32768
    assert cfg['eval_batch_size'] == 1
    assert cfg['agent_config'] == {'mode': 'native', 'max_steps': 250}


def test_metric_rows():
    report = {
        'schema_version': 2,
        'dataset_name': 'gpqa_diamond',
        'metrics': [{
            'identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
            'num': 20, 'score': 0.35, 'macro_score': 0.35,
            'categories': [{
                'name': ['physics'], 'num': 10, 'score': 0.4, 'macro_score': 0.4,
                'subsets': [{'name': 'default', 'num': 10, 'score': 0.4, 'is_aggregate': False}],
            }],
        }],
    }
    rows = bench.metric_rows(report)
    assert len(rows) == 3
    top = rows[0]
    assert top['metric'] == 'accuracy' and top['category'] == '' and top['score'] == 0.35
    assert rows[1]['category'] == 'physics'
    assert rows[2]['subset'] == 'default'


def test_db_roundtrip_and_summary():
    data_root = Path(tempfile.mkdtemp())
    conn = bench.connect_db(data_root / 'results.db')
    bench.init_db(conn)
    bench.db_start_run(conn, {
        'run_id': 'r1', 'run_group': 'g', 'model': 'minimax', 'suite': 'knowledge',
        'profile': 'smoke', 'dataset': 'gpqa_diamond', 'started_at': '2026-01-01T00:00:00',
        'evalscope_version': '1.12.0', 'git_commit': 'abc', 'output_dir': '/tmp/out',
        'config_json': '{}',
    })
    bench.db_record_metrics(conn, 'r1', 'gpqa_diamond', [
        {'metric': 'accuracy', 'category': '', 'subset': '', 'num': 20, 'score': 0.35, 'macro_score': 0.35},
    ])
    bench.db_record_samples(conn, 'r1', 'gpqa_diamond', {'default': ['0', '1']})
    bench.db_finish_run(conn, 'r1', status='success', error=None,
                        execution_summary={'requested': 20, 'succeeded': 20, 'errored': 0}, perf_metrics=None)
    bench.db_start_run(conn, {
        'run_id': 'r2', 'run_group': 'g', 'model': 'minimax', 'suite': 'swe',
        'profile': 'smoke', 'dataset': 'live_code_bench', 'started_at': '2026-01-01T00:00:01',
        'evalscope_version': '1.12.0', 'git_commit': 'abc', 'output_dir': '/tmp/out2',
        'config_json': '{}',
    })
    bench.db_finish_run(conn, 'r2', status='failed', error='boom', execution_summary=None, perf_metrics=None)

    summary = bench.build_summary(conn, data_root)
    assert 'gpqa_diamond' in summary
    assert '0.3500' in summary
    assert 'live_code_bench' in summary and 'failed' in summary
    samples = conn.execute('SELECT sample_ids FROM run_samples WHERE run_id = ?', ('r1',)).fetchone()
    assert json.loads(samples['sample_ids']) == {'default': ['0', '1']}


def test_sample_ids_from_outputs():
    out = Path(tempfile.mkdtemp())
    reviews = out / 'reviews' / 'model'
    reviews.mkdir(parents=True)
    (reviews / 'gpqa_diamond_default.jsonl').write_text(
        json.dumps({'sample_id': 3}) + '\n' + json.dumps({'sample_id': 1}) + '\n'
    )
    (reviews / 'gpqa_diamond_other.jsonl').write_text(
        json.dumps({'index': 7}) + '\n'
    )
    assert bench.sample_ids_from_outputs(out, 'gpqa_diamond') == {'default': ['1', '3'], 'other': ['7']}


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'ok  {name}')
            except Exception as exc:
                failures += 1
                print(f'FAIL {name}: {exc}')
    sys.exit(1 if failures else 0)
