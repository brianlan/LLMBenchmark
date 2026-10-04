import json
from pathlib import Path

from llmbench.cli import main
from llmbench.store import Store
from llmbench.summary import TOP_CATEGORY_KEY

REPO = Path(__file__).resolve().parents[1]
CONFIGS = str(REPO / 'configs')


def test_dry_run_has_zero_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('MINIMAX_API_KEY', raising=False)
    exit_code = main([
        'run', '--dry-run', '--suite', 'knowledge', '--profile', 'smoke',
        '--data-root', str(tmp_path), '--config-dir', CONFIGS,
    ])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['dry_run'] is True
    assert payload['entries']
    assert not (tmp_path / 'results.db').exists()
    assert not (tmp_path / 'outputs').exists()


def test_dry_run_explicit_datasets_override_core_filter(tmp_path, capsys):
    exit_code = main([
        'run', '--dry-run', '--suite', 'vision', '--datasets', 'chartqa',
        '--data-root', str(tmp_path), '--config-dir', CONFIGS,
    ])
    assert exit_code == 0
    datasets = [entry['dataset'] for entry in json.loads(capsys.readouterr().out)['entries']]
    assert datasets == ['chartqa']


def test_wrong_suite_dataset_exits_2(tmp_path, capsys):
    exit_code = main([
        'run', '--dry-run', '--suite', 'knowledge', '--datasets', 'chartqa',
        '--data-root', str(tmp_path), '--config-dir', CONFIGS,
    ])
    assert exit_code == 2
    assert 'belongs to suite' in capsys.readouterr().err


def test_unknown_dataset_exits_2(tmp_path, capsys):
    exit_code = main([
        'run', '--dry-run', '--suite', 'knowledge', '--datasets', 'not-a-dataset',
        '--data-root', str(tmp_path), '--config-dir', CONFIGS,
    ])
    assert exit_code == 2
    assert 'unknown dataset' in capsys.readouterr().err


def test_invalid_limit_exits_2(tmp_path, capsys):
    exit_code = main([
        'run', '--dry-run', '--suite', 'knowledge', '--limit', '0',
        '--data-root', str(tmp_path), '--config-dir', CONFIGS,
    ])
    assert exit_code == 2
    assert '--limit' in capsys.readouterr().err


def test_missing_api_key_fails_preflight_without_creating_db(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('MINIMAX_API_KEY', raising=False)
    exit_code = main([
        'run', '--suite', 'knowledge', '--profile', 'smoke', '--datasets', 'gpqa_diamond',
        '--data-root', str(tmp_path), '--config-dir', CONFIGS,
    ])
    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert 'preflight failed' in stderr
    assert 'MINIMAX_API_KEY' in stderr
    assert not (tmp_path / 'results.db').exists()
    assert not (tmp_path / 'outputs').exists()


def test_report_command_writes_summary_from_db(tmp_path):
    db = tmp_path / 'results.db'
    store = Store(db)
    store.start_attempt({
        'run_id': 'seed', 'run_group': 'g', 'model_alias': 'minimax', 'model_id': 'm',
        'api_url': 'u', 'model_config_identity': 'mc', 'suite': 'knowledge', 'profile': 'lite',
        'dataset': 'gpqa_diamond', 'execution_status': 'running', 'started_at': 't',
        'evalscope_version': '1.12.0', 'tool_version': '0.2.0', 'git_commit': 'g',
        'output_dir': 'o', 'config_json': '{}', 'protocol_identity': 'p',
        'sample_manifest_identity': 's',
    })
    store.finish_attempt('seed', {
        'execution_status': 'completed', 'validity_status': 'complete', 'status_reason': 'complete',
        'phase': 'persist', 'comparability': 'verified', 'finished_at': 't2',
        'num_requested': 5, 'num_succeeded': 5, 'num_errored': 0, 'incomplete': False,
        'report_path': 'p', 'diagnostics': None,
        'perf_metrics': None, 'protocol_identity': 'p', 'sample_manifest_identity': 's',
        'metrics': [{
            'metric_key': 'mk', 'metric_name': 'accuracy', 'aggregation': 'mean',
            'dimensions': {}, 'category_key': TOP_CATEGORY_KEY, 'category': [], 'subset': '',
            'num': 5, 'score': 1.0, 'macro_score': None, 'semantics_kind': 'quality',
            'display_kind': 'percent', 'direction': 'higher', 'unit': '%', 'is_primary': True,
        }],
        'sample_manifest': [],
    })
    store.close()

    out = tmp_path / 'summary.md'
    exit_code = main(['report', '--data-root', str(tmp_path), '--db', str(db), '--out', str(out)])
    assert exit_code == 0
    text = out.read_text(encoding='utf-8')
    assert 'seed' in text


def test_import_without_manifest_exits_1(tmp_path, capsys):
    empty = tmp_path / 'no-such-attempt'
    empty.mkdir()
    exit_code = main([
        'import', '--output-dir', str(empty), '--data-root', str(tmp_path),
        '--db', str(tmp_path / 'results.db'),
    ])
    assert exit_code == 1
    assert 'left untouched' in capsys.readouterr().err


def test_rejected_import_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('llmbench.cli.import_output', lambda *a, **k: {
        'status': 'rejected', 'committed': False, 'run_id': 'r1', 'reason': 'no_metrics'})
    exit_code = main([
        'import', '--output-dir', str(tmp_path), '--data-root', str(tmp_path),
        '--db', str(tmp_path / 'results.db'),
    ])
    assert exit_code == 1
    assert 'import rejected' in capsys.readouterr().err


def test_summary_write_failure_returns_exit_1(tmp_path, monkeypatch, capsys):
    from llmbench.runner import AttemptResult

    monkeypatch.setenv('MINIMAX_API_KEY', 'test-key')
    monkeypatch.setattr('llmbench.cli._preflight_errors', lambda plan, data_root: [])
    monkeypatch.setattr('llmbench.cli.resolve_task_config', lambda raw: {})
    ok = AttemptResult('r1', tmp_path / 'out', 'completed', 'complete', 'complete', True)
    monkeypatch.setattr('llmbench.cli.run_plan', lambda plan, store, repo_dir: [ok])

    def boom(*args, **kwargs):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr('llmbench.cli.write_summary', boom)
    exit_code = main([
        'run', '--suite', 'knowledge', '--profile', 'smoke', '--datasets', 'gpqa_diamond',
        '--data-root', str(tmp_path), '--config-dir', CONFIGS,
    ])
    assert exit_code == 1
    assert 'summary generation failed' in capsys.readouterr().err
