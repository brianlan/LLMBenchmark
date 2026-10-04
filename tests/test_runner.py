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


@contextmanager
def stub_ocr_patch():
    yield {'applied': True, 'reason': 'test', 'target': 'nltk.edit_distance'}


def stub_evalscope(monkeypatch, run_task):
    module = types.ModuleType('evalscope')
    module.__version__ = '1.12.0'
    module.run_task = run_task
    monkeypatch.setitem(sys.modules, 'evalscope', module)
    monkeypatch.setattr(runner, 'capture_loaded_dataset', fake_capture)
    monkeypatch.setattr(runner, 'resolve_task_config', lambda raw: {'model': raw['model']})
    monkeypatch.setattr(runner, 'ocr_compat_patch', stub_ocr_patch)
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


def execute(tmp_path, monkeypatch, task, *, entry_obj=None, store=None, api_key='test-key'):
    stub_evalscope(monkeypatch, task)
    entry_obj = entry_obj or entry()
    store = store or Store(tmp_path / 'results.db')
    output_dir = tmp_path / 'outputs' / 'attempt'
    raw_cfg = {
        'model': MODEL_ID, 'model_id': MODEL_ID, 'api_url': MODEL_CFG['api_url'],
        'api_key': api_key, 'datasets': [entry_obj.dataset], 'work_dir': str(output_dir),
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


SENTINEL = 'sk-review-SENTINEL-not-a-real-key'


def _leaks(text: str) -> bool:
    return SENTINEL in (text or '')


def _secret_entry():
    entry_obj = entry()
    entry_obj.model_cfg = {**MODEL_CFG, 'api_key': SENTINEL}
    return entry_obj


def _assert_no_sentinel(result, store, output_dir, tmp_path, capsys):
    captured = capsys.readouterr()
    row = store.get_attempt('attempt-1')
    outcome = read_json(output_dir / 'run_outcome.json')
    from llmbench.summary import render
    summary = render(store, tmp_path)
    assert not _leaks(result.status_reason)
    assert not _leaks(captured.out) and not _leaks(captured.err)
    assert not _leaks(row['status_reason']) and not _leaks(row['error'])
    assert not _leaks(row['config_json']) and not _leaks(row['diagnostics_json'] or '')
    assert not _leaks(json.dumps(outcome))
    assert not _leaks(summary)


def test_b1_secret_from_task_exception_never_reaches_outputs(tmp_path, monkeypatch, capsys):
    def task(cfg):
        raise RuntimeError(f'endpoint rejected {SENTINEL}')

    result, store, output_dir = execute(tmp_path, monkeypatch, task, entry_obj=_secret_entry(),
                                        api_key=SENTINEL)
    _assert_no_sentinel(result, store, output_dir, tmp_path, capsys)
    store.close()


def test_b1_secret_from_report_parse_error_is_redacted(tmp_path, monkeypatch, capsys):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(score=SENTINEL),
                                        entry_obj=_secret_entry(), api_key=SENTINEL)
    assert result.execution_status == 'completed'
    assert 'report_metric_parse_error' in result.status_reason
    _assert_no_sentinel(result, store, output_dir, tmp_path, capsys)
    store.close()


def test_b1_secret_from_diagnostics_error_is_redacted(tmp_path, monkeypatch, capsys):
    def boom(output_dir, dataset, *, report_id=None):
        raise RuntimeError(f'diagnostics failed with {SENTINEL}')

    monkeypatch.setattr(runner, 'collect_output_evidence', boom)
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(),
                                        entry_obj=_secret_entry(), api_key=SENTINEL)
    assert result.validity_status == 'invalid'
    _assert_no_sentinel(result, store, output_dir, tmp_path, capsys)
    store.close()


def test_b1_secret_from_store_failure_is_redacted(tmp_path, monkeypatch, capsys):
    class BoomStore(Store):
        def finish_attempt(self, run_id, outcome):
            raise RuntimeError(f'database write failed with {SENTINEL}')

    store = BoomStore(tmp_path / 'results.db')
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(), entry_obj=_secret_entry(),
                                        store=store, api_key=SENTINEL)
    captured = capsys.readouterr()
    assert result.persisted is False
    assert not _leaks(result.error)
    assert not _leaks(captured.err)
    store.close()


def test_b3_primary_metric_without_score_is_unverified_and_visible(tmp_path, monkeypatch):
    result, store, _ = execute(tmp_path, monkeypatch, fake_run_task(score=None))
    assert result.validity_status == 'unverified'
    assert 'primary_metric_missing_score' in result.status_reason
    assert store.get_attempt('attempt-1')['validity_status'] == 'unverified'
    from llmbench.summary import render
    text = render(store, tmp_path)
    assert 'attempt-1' in text.split('## Diagnostics')[1]
    assert 'attempt-1' not in text.split('## Diagnostics')[0]
    store.close()


def test_b4_outcome_write_failure_terminates_attempt_and_continues(tmp_path, monkeypatch, capsys):
    calls = []

    def run_task(cfg):
        dataset = cfg['datasets'][0]
        calls.append(dataset)
        write_report(Path(cfg['work_dir']), dataset=dataset, score=0.5)
        write_jsonl(Path(cfg['work_dir']), 'predictions', dataset=dataset)
        write_jsonl(Path(cfg['work_dir']), 'reviews', dataset=dataset)

    stub_evalscope(monkeypatch, run_task)
    real_write = runner.atomic_write_json

    def flaky_write(path, payload):
        if Path(path).name == 'run_outcome.json':
            raise OSError(28, 'No space left on device')
        return real_write(path, payload)

    monkeypatch.setattr(runner, 'atomic_write_json', flaky_write)
    plan = Plan(data_root=tmp_path, profile='lite', entries=[entry('gpqa_diamond'), entry('math_500')])
    store = Store(tmp_path / 'results.db')
    results = runner.run_plan(plan, store=store, repo_dir=tmp_path)

    assert calls == ['gpqa_diamond', 'math_500']  # the next attempt still runs
    assert len(results) == 2
    first = results[0]
    assert first.execution_status == 'failed' and first.validity_status == 'invalid'
    assert 'evidence_write_error' in first.status_reason
    row = store.get_attempt(first.run_id)
    assert row['execution_status'] == 'failed'
    assert row['validity_status'] == 'invalid'
    assert row['phase'] == 'evidence'
    assert runner.batch_exit_code(results) == 1
    store.close()


def test_b4_manifest_write_failure_does_not_call_model(tmp_path, monkeypatch, capsys):
    calls = []

    def run_task(cfg):
        calls.append(cfg['datasets'][0])

    stub_evalscope(monkeypatch, run_task)
    import llmbench.evidence as evidence_module
    real_write = evidence_module.atomic_write_json

    def flaky_write(path, payload):
        if Path(path).name == 'run_manifest.json':
            raise OSError(28, 'No space left on device')
        return real_write(path, payload)

    monkeypatch.setattr(evidence_module, 'atomic_write_json', flaky_write)
    result, store, _ = execute(tmp_path, monkeypatch, run_task)

    assert calls == []  # no API spend after the evidence path failed
    assert result.execution_status == 'failed' and result.validity_status == 'invalid'
    row = store.get_attempt('attempt-1')
    assert row['execution_status'] == 'failed'
    diagnostics = json.loads(row['diagnostics_json'])
    assert diagnostics['attempt_error_phase'] == 'inference'
    store.close()


def test_b5_import_of_semantically_invalid_report_is_rejected(tmp_path, monkeypatch):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(score=0.7))
    before = store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n']
    report = output_dir / 'reports' / MODEL_ID / 'gpqa_diamond.json'
    payload = json.loads(report.read_text(encoding='utf-8'))
    payload['metrics'] = []
    report.write_text(json.dumps(payload), encoding='utf-8')

    imported = runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert imported['status'] == 'rejected' and imported['committed'] is False
    assert imported['metrics_kept'] == before
    assert store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n'] == before
    assert read_json(output_dir / 'run_import_rejected.json')['kept_metrics'] == before
    assert store.get_attempt('attempt-1')['validity_status'] == 'complete'
    store.close()


def test_b5_import_preserves_evaluation_finished_at_and_latest_order(tmp_path, monkeypatch):
    old_result, store, old_dir = execute(tmp_path, monkeypatch, fake_run_task(score=0.4))
    # pin the first attempt to a clearly older evaluation time
    store.conn.execute('UPDATE runs SET finished_at=? WHERE run_id=?',
                       ('2026-01-01T00:00:00+00:00', old_result.run_id))
    store.conn.commit()
    old_finished = store.get_attempt(old_result.run_id)['finished_at']

    second_dir = tmp_path / 'outputs' / 'attempt2'
    raw_cfg = {'model': MODEL_ID, 'model_id': MODEL_ID, 'api_url': MODEL_CFG['api_url'],
               'api_key': 'test-key', 'datasets': ['gpqa_diamond'], 'work_dir': str(second_dir)}
    new_result = runner.execute_attempt(
        entry(), store=store, attempt_id='attempt-2', run_group='group',
        output_dir=second_dir, raw_cfg=raw_cfg, resolved_cfg={'model': MODEL_ID},
        repo_dir=tmp_path, data_root=tmp_path,
    )
    new_finished = store.get_attempt(new_result.run_id)['finished_at']
    assert new_finished and new_finished >= old_finished

    imported = runner.import_output(old_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert imported['status'] == 'committed'
    row = store.get_attempt(old_result.run_id)
    assert row['finished_at'] == old_finished  # import does not rewrite history
    assert row['imported_at'] is not None

    from llmbench.summary import render
    formal = render(store, tmp_path).split('## Diagnostics')[0]
    assert new_result.run_id in formal
    assert old_result.run_id not in formal  # the newer 90% run still wins
    store.close()


def test_c1_generation_config_is_deep_merged(tmp_path):
    entry_obj = entry()
    entry_obj.model_cfg = {
        **MODEL_CFG,
        'generation_config': {'max_tokens': 100, 'extra_body': {'reasoning': True, 'top_p': 0.9}},
    }
    entry_obj.spec = {'generation_config': {'extra_body': {'top_p': 0.5}},
                      'extra_params': {'build_docker_images': True, 'nested': {'a': 1}}}
    raw = runner.build_raw_task_config(entry_obj, data_root=tmp_path,
                                       output_dir=tmp_path / 'out', api_key='k')
    assert raw['generation_config'] == {'max_tokens': 100, 'extra_body': {'reasoning': True, 'top_p': 0.5}}
    assert raw['dataset_args']['gpqa_diamond']['extra_params'] == {'build_docker_images': True, 'nested': {'a': 1}}
