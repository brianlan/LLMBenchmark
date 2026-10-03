import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

import llmbench.runner as runner
from llmbench.config import ConfigError, Plan, PlanEntry, build_plan
from llmbench.evidence import MetricParseError, ReportError
from llmbench.store import Store
from llmbench.util import read_json

MODEL_ID = 'MiniMax-M3.1-Flash-Preview'
MODEL_CFG = {
    'model_id': MODEL_ID,
    'report_id': MODEL_ID,
    'api_url': 'https://api.example/v1',
    'api_key': 'test-key',
    'generation_config': {'temperature': 0.0},
}


def entry(dataset='gpqa_diamond', **overrides):
    values = dict(
        model_alias='minimax', model_cfg=MODEL_CFG, suite='knowledge', dataset=dataset,
        spec={}, profile='lite', limit=5, batch_size=4, pinned=False, status='tested',
    )
    values.update(overrides)
    return PlanEntry(**values)


def write_report(output_dir: Path, *, dataset='gpqa_diamond', score=0.5, report_id=MODEL_ID,
                 execution=None, raw=None):
    path = output_dir / 'reports' / report_id / f'{dataset}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'schema_version': 2, 'dataset_name': dataset, 'model_name': report_id,
        'metrics': [{
            'identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
            'num': 5, 'score': score, 'macro_score': score,
            'categories': [{
                'name': ['default'], 'num': 5, 'score': score, 'macro_score': score,
                'subsets': [{'name': 'default', 'score': score, 'num': 5, 'is_aggregate': False}],
            }],
            'semantics': {'kind': 'quality', 'display_kind': 'percent',
                          'direction': 'higher_is_better', 'display_unit': '%'},
        }],
        'primary_metric_identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
        'execution_summary': execution or {'requested': 5, 'succeeded': 5, 'errored': 0, 'incomplete': False},
    }
    path.write_text(json.dumps(raw if raw is not None else report), encoding='utf-8')
    return path


def write_jsonl(output_dir: Path, kind, dataset='gpqa_diamond', report_id=MODEL_ID, count=5):
    path = output_dir / kind / report_id / f'{dataset}_default.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({
        'index': index,
        'model_output': {'choices': [{'stop_reason': 'stop'}],
                         'usage': {'input_tokens': 10, 'output_tokens': 5}},
    }) for index in range(count)]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


@contextmanager
def fake_capture(callback):
    samples = [type('Sample', (), {'id': index, 'input': f'What is 2+{index}?'})() for index in range(5)]
    callback(None, {'default': samples})
    yield {}


def stub_evalscope(monkeypatch, run_task):
    module = types.ModuleType('evalscope')
    module.__version__ = '1.12.0'
    module.run_task = run_task
    monkeypatch.setitem(sys.modules, 'evalscope', module)
    monkeypatch.setattr(runner, 'capture_loaded_dataset', fake_capture)
    monkeypatch.setattr(runner, 'resolve_task_config', lambda raw: {'model': raw['model']})
    monkeypatch.setattr(runner, 'apply_ocr_compat', lambda: {'applied': False, 'reason': 'test'})
    monkeypatch.setattr(runner, 'report_patch_status', lambda status: None)
    return module


def fake_run_task(*, dataset='gpqa_diamond', score=0.5, fail=False, interrupt=False,
                  write=True, report_raw=None, execution=None):
    def task(cfg):
        if interrupt:
            raise KeyboardInterrupt()
        if fail:
            raise RuntimeError('model endpoint exploded')
        if not write:
            return
        output_dir = Path(cfg['work_dir'])
        write_report(output_dir, dataset=dataset, score=score, raw=report_raw,
                     execution=execution)
        write_jsonl(output_dir, 'predictions', dataset=dataset)
        write_jsonl(output_dir, 'reviews', dataset=dataset)
    return task


def execute(tmp_path, monkeypatch, task, *, entry_obj=None, store=None):
    stub_evalscope(monkeypatch, task)
    entry_obj = entry_obj or entry()
    store = store or Store(tmp_path / 'results.db')
    output_dir = tmp_path / 'outputs' / 'attempt'
    raw_cfg = {
        'model': MODEL_ID, 'model_id': MODEL_ID, 'api_url': MODEL_CFG['api_url'],
        'api_key': 'test-key', 'datasets': [entry_obj.dataset], 'work_dir': str(output_dir),
    }
    result = runner.execute_attempt(
        entry_obj, store=store, attempt_id='attempt-1', run_group='group',
        output_dir=output_dir, raw_cfg=raw_cfg, resolved_cfg={'model': MODEL_ID},
        repo_dir=tmp_path, data_root=tmp_path,
    )
    return result, store, output_dir


def test_complete_attempt_is_persisted_with_evidence(tmp_path, monkeypatch):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task())
    assert (result.execution_status, result.validity_status) == ('completed', 'complete')
    assert result.persisted
    row = store.get_attempt('attempt-1')
    assert row['validity_status'] == 'complete'
    assert row['comparability'] == 'verified'
    assert store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n'] >= 1
    assert store.latest_extra('attempt-1')[0]['selected'] == 5
    outcome = read_json(output_dir / 'run_outcome.json')
    assert outcome['execution_status'] == 'completed'
    manifest = read_json(output_dir / 'run_manifest.json')
    assert manifest['sample_manifest_identity']
    assert manifest['model_config_identity']
    assert 'test-key' not in json.dumps(manifest)
    store.close()


def test_zero_score_complete_run_is_valid(tmp_path, monkeypatch):
    result, store, _ = execute(tmp_path, monkeypatch, fake_run_task(score=0.0))
    assert result.validity_status == 'complete'
    assert store.conn.execute('SELECT score FROM metrics WHERE is_primary=1').fetchone()['score'] == 0.0
    store.close()


def test_missing_report_is_invalid_not_success(tmp_path, monkeypatch):
    result, store, _ = execute(tmp_path, monkeypatch, fake_run_task(write=False))
    assert result.execution_status == 'completed'
    assert result.validity_status == 'invalid'
    assert 'report_missing' in result.status_reason
    store.close()


def test_corrupt_report_is_invalid_with_report_parse_error(tmp_path, monkeypatch):
    def task(cfg):
        output_dir = Path(cfg['work_dir'])
        path = output_dir / 'reports' / MODEL_ID / 'gpqa_diamond.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{broken', encoding='utf-8')

    result, store, _ = execute(tmp_path, monkeypatch, task)
    assert result.validity_status == 'invalid'
    assert 'report_parse_error' in result.status_reason
    assert store.get_attempt('attempt-1')['validity_status'] == 'invalid'
    store.close()


def test_task_exception_is_failed_and_redacted(tmp_path, monkeypatch):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(fail=True))
    assert result.execution_status == 'failed'
    assert result.validity_status == 'invalid'
    assert 'endpoint exploded' in result.status_reason or 'task_error' in result.status_reason
    assert store.get_attempt('attempt-1')['execution_status'] == 'failed'
    store.close()


def test_persist_failure_keeps_evidence_and_is_not_accepted(tmp_path, monkeypatch, capsys):
    def boom(run_id, outcome):
        raise sqlite_error()

    def sqlite_error():
        import sqlite3
        return sqlite3.OperationalError('disk I/O error')

    store = Store(tmp_path / 'results.db')
    monkeypatch.setattr(store, 'finish_attempt', boom)
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(), store=store)
    assert result.persisted is False
    assert runner.batch_exit_code([result]) == 1
    assert (output_dir / 'run_outcome.json').exists()
    captured = capsys.readouterr()
    assert 'import --output-dir' in captured.err
    store.close()


def test_import_is_idempotent_and_does_not_erase_scores_on_failure(tmp_path, monkeypatch):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(score=0.7))
    count = store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n']
    assert count >= 1

    first = runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    second = runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert first['run_id'] == second['run_id']
    assert store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n'] == count

    # a broken report must fail the import and leave the previous scores intact
    report_path = output_dir / 'reports' / MODEL_ID / 'gpqa_diamond.json'
    report_path.write_text('{broken', encoding='utf-8')
    with pytest.raises(ReportError):
        runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n'] == count
    store.close()


def test_batch_continues_after_failure_and_reports_nonzero_exit(tmp_path, monkeypatch):
    calls = []

    def run_task(cfg):
        calls.append(cfg['datasets'][0])
        if len(calls) == 1:
            raise RuntimeError('first dataset failed')
        write_report(Path(cfg['work_dir']), dataset=cfg['datasets'][0])
        write_jsonl(Path(cfg['work_dir']), 'predictions', dataset=cfg['datasets'][0])
        write_jsonl(Path(cfg['work_dir']), 'reviews', dataset=cfg['datasets'][0])

    stub_evalscope(monkeypatch, run_task)
    plan = Plan(data_root=tmp_path, profile='lite', entries=[
        entry('gpqa_diamond'), entry('math_500'),
    ])
    store = Store(tmp_path / 'results.db')
    results = runner.run_plan(plan, store=store, repo_dir=tmp_path)
    assert len(results) == 2
    assert results[0].validity_status == 'invalid'
    assert results[1].validity_status == 'complete'
    assert runner.batch_exit_code(results) == 1
    assert runner.batch_exit_code(results, allow_partial=True) == 1
    store.close()


def test_keyboard_interrupt_marks_interrupted_and_aborts_batch(tmp_path, monkeypatch):
    calls = []

    def run_task(cfg):
        calls.append(cfg['datasets'][0])
        raise KeyboardInterrupt()

    stub_evalscope(monkeypatch, run_task)
    plan = Plan(data_root=tmp_path, profile='lite', entries=[
        entry('gpqa_diamond'), entry('math_500'),
    ])
    store = Store(tmp_path / 'results.db')
    with pytest.raises(runner.BatchInterrupted):
        runner.run_plan(plan, store=store, repo_dir=tmp_path)
    assert calls == ['gpqa_diamond']  # the second entry is never started
    row = store.conn.execute('SELECT execution_status FROM runs').fetchone()
    assert row['execution_status'] == 'interrupted'
    store.close()


def test_duplicate_attempt_id_is_reported_not_overwritten(tmp_path, monkeypatch):
    result, store, _ = execute(tmp_path, monkeypatch, fake_run_task())
    assert result.persisted
    stub_evalscope(monkeypatch, fake_run_task())
    second = runner.execute_attempt(
        entry(), store=store, attempt_id='attempt-1', run_group='group',
        output_dir=tmp_path / 'outputs' / 'attempt2',
        raw_cfg={'model': MODEL_ID, 'model_id': MODEL_ID, 'api_key': 'k'},
        resolved_cfg={}, repo_dir=tmp_path, data_root=tmp_path,
    )
    assert second.persisted is False
    assert 'duplicate' in second.status_reason
    store.close()


def test_plan_identity_errors_are_raised_before_running(tmp_path):
    from llmbench.config import load_models, load_suites

    repo = Path(__file__).resolve().parents[1]
    models = load_models(repo / 'configs')
    suites = load_suites(repo / 'configs')
    with pytest.raises(ConfigError):
        build_plan(models_cfg=models, suites_doc=suites, data_root=tmp_path,
                   model_names=['nope'], suite_names=['knowledge'], profile='lite')
