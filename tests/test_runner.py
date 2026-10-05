import json
import shutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

import llmbench.runner as runner
from llmbench.config import ConfigError, Plan, PlanEntry, build_plan
from llmbench.evidence import MetricParseError, ReportError
from llmbench.store import Store
from llmbench.util import digest, read_json

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
    failed = runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert failed['status'] == 'failed' and failed['committed'] is False
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


def test_b4_malformed_execution_summary_terminates_and_continues(tmp_path, monkeypatch):
    def run_task(cfg):
        dataset = cfg['datasets'][0]
        execution = ({'requested': '5', 'succeeded': 5, 'errored': 0, 'incomplete': False}
                     if dataset == 'gpqa_diamond' else
                     {'requested': 5, 'succeeded': 5, 'errored': 0, 'incomplete': False})
        write_report(Path(cfg['work_dir']), dataset=dataset, execution=execution)
        write_jsonl(Path(cfg['work_dir']), 'predictions', dataset=dataset)
        write_jsonl(Path(cfg['work_dir']), 'reviews', dataset=dataset)

    stub_evalscope(monkeypatch, run_task)
    plan = Plan(data_root=tmp_path, profile='lite', entries=[entry('gpqa_diamond'), entry('math_500')])
    store = Store(tmp_path / 'results.db')
    results = runner.run_plan(plan, store=store, repo_dir=tmp_path)

    assert len(results) == 2
    assert results[0].validity_status == 'unverified'
    assert 'invalid_execution_summary' in results[0].status_reason
    assert results[1].validity_status == 'complete'
    assert store.conn.execute("SELECT COUNT(*) AS n FROM runs WHERE execution_status='running'").fetchone()['n'] == 0
    assert runner.batch_exit_code(results) == 1
    store.close()


def test_b5_import_from_verified_to_unknown_is_rejected(tmp_path, monkeypatch):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(score=0.8))
    before = store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n']
    shutil.rmtree(output_dir / 'reviews')

    imported = runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert imported['status'] == 'rejected' and imported['committed'] is False
    assert imported['new_comparability'] == 'unknown'
    row = store.get_attempt('attempt-1')
    assert row['validity_status'] == 'complete' and row['comparability'] == 'verified'
    assert store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n'] == before

    from llmbench.summary import render
    formal = render(store, tmp_path).split('## Diagnostics')[0]
    assert 'attempt-1' in formal
    audit = read_json(output_dir / 'run_import_rejected.json')
    assert audit['kept_comparability'] == 'verified'
    store.close()


def test_b5_import_evidence_write_failure_does_not_commit(tmp_path, monkeypatch):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(score=0.8))
    before = store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n']
    real_write = runner.atomic_write_json

    def flaky_write(path, payload):
        if Path(path).name == 'run_outcome.json':
            raise OSError(28, 'No space left on device')
        return real_write(path, payload)

    monkeypatch.setattr(runner, 'atomic_write_json', flaky_write)
    imported = runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert imported['status'] == 'failed'
    assert imported['committed'] is False
    assert imported['evidence_file_updated'] is False
    assert store.conn.execute('SELECT COUNT(*) AS n FROM metrics').fetchone()['n'] == before
    store.close()


def test_b5_import_database_failure_reports_evidence_updated(tmp_path, monkeypatch):
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(score=0.8))

    def boom(run_id, outcome):
        raise RuntimeError('database is read-only')

    monkeypatch.setattr(store, 'finish_attempt', boom)
    imported = runner.import_output(output_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert imported['status'] == 'failed'
    assert imported['committed'] is False
    assert imported['evidence_file_updated'] is True
    assert 'database_error' in imported['reason']
    store.close()


def test_b4_assessment_exception_terminates_attempt_and_continues(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('assessment crashed')

    monkeypatch.setattr(runner, 'apply_coverage', boom)
    stub_evalscope(monkeypatch, fake_run_task())
    plan = Plan(data_root=tmp_path, profile='lite', entries=[entry('gpqa_diamond')])
    store = Store(tmp_path / 'results.db')
    results = runner.run_plan(plan, store=store, repo_dir=tmp_path)
    assert len(results) == 1
    assert results[0].execution_status == 'failed' and results[0].validity_status == 'invalid'
    row = store.get_attempt(results[0].run_id)
    assert row['execution_status'] == 'failed'
    assert store.conn.execute("SELECT COUNT(*) AS n FROM runs WHERE execution_status='running'").fetchone()['n'] == 0
    diagnostics = json.loads(row['diagnostics_json'])
    assert diagnostics['attempt_error_phase'] == 'assessment'
    store.close()


def test_sandbox_attempt_records_prerun_image_inventory(tmp_path, monkeypatch):
    entry_obj = entry()
    entry_obj.spec = {'requires_sandbox': True, 'sandbox': {'enabled': True, 'engine': 'docker'}}
    monkeypatch.setattr(runner, '_swe_image_snapshot', lambda: ['swebench/old:latest'])
    result, store, output_dir = execute(tmp_path, monkeypatch, fake_run_task(), entry_obj=entry_obj)
    assert result.validity_status == 'complete'
    manifest = read_json(output_dir / 'run_manifest.json')
    assert manifest['baseline_swe_images'] == ['swebench/old:latest']
    store.close()


OPAQUE_SECRET = 'provider_opaque_secret_6c177eaa'


def _standalone_manifest(attempt_id, model_id='Model-A', dataset='gpqa_diamond',
                         api_key=OPAQUE_SECRET, protocol='proto-standalone',
                         manifest_identity='sm-standalone'):
    return {
        'attempt_id': attempt_id, 'run_group': 'g', 'dataset': dataset,
        'model_alias': 'minimax', 'model_id': model_id, 'report_id': model_id,
        'suite': 'knowledge', 'profile': 'lite', 'created_at': '2026-01-01T00:00:00+00:00',
        'model_config_identity': f'mci-{model_id}', 'protocol_identity': protocol,
        'sample_manifest_identity': manifest_identity, 'task_config': {'api_key': api_key},
    }


def _write_standalone_attempt(root, attempt_id, *, model_id='Model-A', dataset='gpqa_diamond',
                              score=0.8, api_key=OPAQUE_SECRET, metrics=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'run_manifest.json').write_text(json.dumps(
        _standalone_manifest(attempt_id, model_id=model_id, dataset=dataset, api_key=api_key),
        ensure_ascii=False), encoding='utf-8')
    report_dir = root / 'reports' / model_id
    report_dir.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        metrics = [{
            'identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
            'score': score, 'num': 1, 'categories': [], 'semantics': {'kind': 'quality'},
        }]
    (report_dir / f'{dataset}.json').write_text(json.dumps({
        'dataset_name': dataset, 'model_name': model_id, 'metrics': metrics,
        'primary_metric_identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
        'execution_summary': {'requested': 1, 'succeeded': 1, 'errored': 0, 'incomplete': False},
    }), encoding='utf-8')


def test_p1_import_identity_conflict_is_rejected(tmp_path):
    a_dir = tmp_path / 'a'
    b_dir = tmp_path / 'b'
    _write_standalone_attempt(a_dir, 'run-1', model_id='Model-A', score=0.8)
    _write_standalone_attempt(b_dir, 'run-1', model_id='Model-B', score=0.2)
    store = Store(tmp_path / 'results.db')

    first = runner.import_output(a_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert first['status'] == 'committed'
    finished_before = store.get_attempt('run-1')['finished_at']

    second = runner.import_output(b_dir, store, repo_dir=tmp_path, data_root=tmp_path)
    assert second['status'] == 'rejected'
    assert second['committed'] is False
    assert 'identity_conflict' in second['reason']
    assert any('model_id' in conflict for conflict in second['conflicts'])

    row = store.get_attempt('run-1')
    assert row['model_id'] == 'Model-A'
    assert row['finished_at'] == finished_before
    score = store.conn.execute(
        'SELECT score FROM metrics WHERE run_id=? AND is_primary=1', ('run-1',)).fetchone()['score']
    assert score == 0.8
    audit = read_json(b_dir / 'run_import_rejected.json')
    assert audit['kind'] == 'identity_conflict'
    store.close()


def test_p1_import_redacts_known_secret_without_pattern(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempt'
    _write_standalone_attempt(attempt, 'run-secret')
    monkeypatch.setattr(runner, 'metric_rows', lambda report: (_ for _ in ()).throw(
        RuntimeError(f'provider returned {OPAQUE_SECRET} while scoring')))
    store = Store(tmp_path / 'results.db')
    result = runner.import_output(attempt, store, repo_dir=tmp_path, data_root=tmp_path)
    assert result['status'] == 'failed'
    assert OPAQUE_SECRET not in result['reason']
    assert '***' in result['reason']
    assert not store.attempt_exists('run-secret')
    store.close()


def test_p1_import_cli_redacts_known_secret(tmp_path, monkeypatch, capsys):
    from llmbench.cli import main

    attempt = tmp_path / 'attempt'
    _write_standalone_attempt(attempt, 'run-secret-cli')
    monkeypatch.setattr(runner, 'metric_rows', lambda report: (_ for _ in ()).throw(
        RuntimeError(f'provider returned {OPAQUE_SECRET} while scoring')))
    exit_code = main([
        'import', '--output-dir', str(attempt), '--data-root', str(tmp_path),
        '--db', str(tmp_path / 'results.db'),
    ])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert OPAQUE_SECRET not in captured.err and OPAQUE_SECRET not in captured.out
    assert '***' in captured.err


def test_p2_first_import_is_atomic_and_leaves_no_running_row(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempt'
    _write_standalone_attempt(attempt, 'run-atomic')
    store = Store(tmp_path / 'results.db')

    def boom(run_id, outcome):
        raise RuntimeError('metrics write failed')

    monkeypatch.setattr(store, '_write_outcome', boom)
    result = runner.import_output(attempt, store, repo_dir=tmp_path, data_root=tmp_path)
    assert result['status'] == 'failed' and result['committed'] is False
    assert store.attempt_exists('run-atomic') is False
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM runs WHERE execution_status='running'").fetchone()['n'] == 0
    store.close()


def test_p2_duplicate_metric_identity_is_rejected_before_persistence(tmp_path):
    duplicate = {
        'identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
        'score': 0.5, 'num': 1, 'categories': [], 'semantics': {'kind': 'quality'},
    }
    attempt = tmp_path / 'attempt'
    _write_standalone_attempt(attempt, 'run-dup', metrics=[duplicate, dict(duplicate)])
    store = Store(tmp_path / 'results.db')
    result = runner.import_output(attempt, store, repo_dir=tmp_path, data_root=tmp_path)
    assert result['status'] == 'failed'
    assert 'duplicate metric identity' in result['reason']
    assert store.attempt_exists('run-dup') is False
    store.close()


def test_p2_duplicate_metric_identity_marks_run_invalid(tmp_path, monkeypatch):
    duplicate = {
        'identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
        'score': 0.5, 'num': 5, 'categories': [], 'semantics': {'kind': 'quality'},
    }

    def run_task(cfg):
        dataset = cfg['datasets'][0]
        write_report(Path(cfg['work_dir']), dataset=dataset,
                     raw={'dataset_name': dataset, 'model_name': MODEL_ID,
                          'metrics': [duplicate, dict(duplicate)],
                          'execution_summary': {'requested': 5, 'succeeded': 5,
                                                'errored': 0, 'incomplete': False}})

    result, store, _ = execute(tmp_path, monkeypatch, run_task)
    assert result.execution_status == 'completed'
    assert result.validity_status == 'invalid'
    assert 'report_metric_parse_error' in result.status_reason
    store.close()


def test_p2_interrupt_during_assessment_terminates_attempt(tmp_path, monkeypatch):
    calls = []

    def run_task(cfg):
        dataset = cfg['datasets'][0]
        calls.append(dataset)
        write_report(Path(cfg['work_dir']), dataset=dataset)
        write_jsonl(Path(cfg['work_dir']), 'predictions', dataset=dataset)
        write_jsonl(Path(cfg['work_dir']), 'reviews', dataset=dataset)

    monkeypatch.setattr(runner, 'apply_coverage', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    stub_evalscope(monkeypatch, run_task)
    plan = Plan(data_root=tmp_path, profile='lite', entries=[entry('gpqa_diamond'), entry('math_500')])
    store = Store(tmp_path / 'results.db')
    with pytest.raises(runner.BatchInterrupted):
        runner.run_plan(plan, store=store, repo_dir=tmp_path)
    assert calls == ['gpqa_diamond']  # the second attempt is never started
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM runs WHERE execution_status='running'").fetchone()['n'] == 0
    row = store.conn.execute('SELECT execution_status, validity_status FROM runs').fetchone()
    assert row['execution_status'] == 'interrupted' and row['validity_status'] == 'invalid'
    store.close()


def _mutate_manifest(attempt_dir, **changes):
    path = attempt_dir / 'run_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for key, value in changes.items():
        manifest[key] = value
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def _primary_score(store, run_id):
    return store.conn.execute(
        'SELECT score FROM metrics WHERE run_id=? AND is_primary=1', (run_id,)).fetchone()['score']


@pytest.mark.parametrize('mode', ['missing', 'none', 'empty'])
def test_incomplete_model_identity_cannot_replace_existing_run(tmp_path, mode):
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    _write_standalone_attempt(a_dir, 'run-1', model_id='Model-A', score=0.8)
    _write_standalone_attempt(b_dir, 'run-1', model_id='Model-B', score=0.2)
    manifest = json.loads((b_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest.pop('model_id')
    manifest.pop('model_config_identity')
    if mode in ('none', 'empty'):
        blank = None if mode == 'none' else ''
        manifest['model_id'] = blank
        manifest['model_config_identity'] = blank
    (b_dir / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    store = Store(tmp_path / 'results.db')
    try:
        first = runner.import_output(a_dir, store, repo_dir=tmp_path, data_root=tmp_path)
        assert first['status'] == 'committed'

        second = runner.import_output(b_dir, store, repo_dir=tmp_path, data_root=tmp_path)
        assert second['status'] == 'rejected' and second['committed'] is False
        assert 'model_id' in second['reason']
        assert store.get_attempt('run-1')['model_id'] == 'Model-A'
        assert _primary_score(store, 'run-1') == 0.8
    finally:
        store.close()


def test_recomputed_sample_identity_conflict_is_rejected(tmp_path):
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    _write_standalone_attempt(a_dir, 'run-s', model_id='Model-A', score=0.8)
    _write_standalone_attempt(b_dir, 'run-s', model_id='Model-A', score=0.2)
    rows_a = [{'subset': 'gpqa_diamond', 'sample_ids': ['1', '2']}]
    rows_b = [{'subset': 'gpqa_diamond', 'sample_ids': ['3', '4']}]
    _mutate_manifest(a_dir, sample_manifest=rows_a, sample_manifest_identity=digest(rows_a))
    manifest_b = json.loads((b_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest_b.pop('sample_manifest_identity')
    manifest_b['sample_manifest'] = rows_b
    (b_dir / 'run_manifest.json').write_text(json.dumps(manifest_b), encoding='utf-8')

    store = Store(tmp_path / 'results.db')
    try:
        assert runner.import_output(a_dir, store, repo_dir=tmp_path,
                                    data_root=tmp_path)['status'] == 'committed'
        stored_identity = store.get_attempt('run-s')['sample_manifest_identity']
        second = runner.import_output(b_dir, store, repo_dir=tmp_path, data_root=tmp_path)
        assert second['status'] == 'rejected'
        assert 'sample_manifest_identity' in second['reason']
        assert store.get_attempt('run-s')['sample_manifest_identity'] == stored_identity
        assert _primary_score(store, 'run-s') == 0.8
    finally:
        store.close()


def test_declared_sample_identity_must_match_content(tmp_path):
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    _write_standalone_attempt(a_dir, 'run-d', model_id='Model-A', score=0.8)
    _write_standalone_attempt(b_dir, 'run-d', model_id='Model-A', score=0.2)
    rows_a = [{'subset': 'gpqa_diamond', 'sample_ids': ['1', '2']}]
    rows_b = [{'subset': 'gpqa_diamond', 'sample_ids': ['3', '4']}]
    _mutate_manifest(a_dir, sample_manifest=rows_a, sample_manifest_identity=digest(rows_a))
    # same declared identity as the stored run, but the content it claims to cover is different
    _mutate_manifest(b_dir, sample_manifest=rows_b, sample_manifest_identity=digest(rows_a))

    store = Store(tmp_path / 'results.db')
    try:
        assert runner.import_output(a_dir, store, repo_dir=tmp_path,
                                    data_root=tmp_path)['status'] == 'committed'
        second = runner.import_output(b_dir, store, repo_dir=tmp_path, data_root=tmp_path)
        assert second['status'] == 'rejected'
        assert 'declared=' in second['reason']
        assert _primary_score(store, 'run-d') == 0.8
    finally:
        store.close()


def test_moved_output_directory_can_be_reimported(tmp_path):
    a_dir = tmp_path / 'a'
    _write_standalone_attempt(a_dir, 'run-m', model_id='Model-A', score=0.8)
    moved = tmp_path / 'moved'
    shutil.copytree(a_dir, moved)

    store = Store(tmp_path / 'results.db')
    try:
        assert runner.import_output(a_dir, store, repo_dir=tmp_path,
                                    data_root=tmp_path)['status'] == 'committed'
        again = runner.import_output(moved, store, repo_dir=tmp_path, data_root=tmp_path)
        assert again['status'] == 'committed'
        assert again['committed'] is True
        assert _primary_score(store, 'run-m') == 0.8
    finally:
        store.close()


def test_identity_conflict_rejection_is_redacted(tmp_path):
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    _write_standalone_attempt(a_dir, 'run-r', model_id='Model-A', score=0.8)
    # the conflicting identity itself carries the opaque secret; the manifest also
    # declares it as the api key so the boundary knows the value
    _write_standalone_attempt(b_dir, 'run-r', model_id=OPAQUE_SECRET, score=0.2)

    store = Store(tmp_path / 'results.db')
    try:
        assert runner.import_output(a_dir, store, repo_dir=tmp_path,
                                    data_root=tmp_path)['status'] == 'committed'
        second = runner.import_output(b_dir, store, repo_dir=tmp_path, data_root=tmp_path)
        assert second['status'] == 'rejected'
        assert OPAQUE_SECRET not in json.dumps(second, ensure_ascii=False)
        assert '***' in second['reason']
        assert OPAQUE_SECRET not in (b_dir / 'run_import_rejected.json').read_text(encoding='utf-8')
        assert _primary_score(store, 'run-r') == 0.8
    finally:
        store.close()


def test_identity_conflict_cli_output_is_redacted(tmp_path, capsys):
    from llmbench.cli import main

    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    _write_standalone_attempt(a_dir, 'run-r-cli', model_id='Model-A', score=0.8)
    _write_standalone_attempt(b_dir, 'run-r-cli', model_id=OPAQUE_SECRET, score=0.2)
    db = tmp_path / 'results.db'

    assert main(['import', '--output-dir', str(a_dir), '--data-root', str(tmp_path),
                 '--db', str(db)]) == 0
    capsys.readouterr()
    exit_code = main(['import', '--output-dir', str(b_dir), '--data-root', str(tmp_path),
                      '--db', str(db)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert OPAQUE_SECRET not in captured.out and OPAQUE_SECRET not in captured.err
    assert 'import rejected' in captured.err


def test_first_import_rejects_declared_sample_identity_mismatch(tmp_path):
    from llmbench.summary import render

    attempt = tmp_path / 'attempt'
    _write_standalone_attempt(attempt, 'run-first-bad', model_id='Model-A', score=0.8)
    rows = [{'subset': 'gpqa_diamond', 'sample_ids': ['1', '2']}]
    _mutate_manifest(attempt, sample_manifest=rows, sample_manifest_identity='0' * 64)

    store = Store(tmp_path / 'results.db')
    try:
        first = runner.import_output(attempt, store, repo_dir=tmp_path, data_root=tmp_path)
        assert first['status'] == 'rejected' and first['committed'] is False
        assert 'declared=' in first['reason']
        # the same unchanged artifact must not become certifiable on a later try either
        second = runner.import_output(attempt, store, repo_dir=tmp_path, data_root=tmp_path)
        assert second['status'] == 'rejected'
        assert store.attempt_exists('run-first-bad') is False
        assert store.conn.execute('SELECT COUNT(*) AS n FROM runs').fetchone()['n'] == 0
        assert 'run-first-bad' not in render(store, tmp_path)
        audit = json.loads((attempt / 'run_import_rejected.json').read_text(encoding='utf-8'))
        assert audit['kind'] == 'identity_conflict' and audit['kept_validity_status'] is None
    finally:
        store.close()


@pytest.mark.parametrize('blank_state', [
    'missing', 'none', 'empty', 'empty_model_id', 'empty_config_identity',
    'blank', 'tab',
])
def test_first_import_without_model_identity_is_not_certified(tmp_path, blank_state):
    from llmbench.summary import render

    attempt = tmp_path / 'attempt'
    _write_standalone_attempt(attempt, 'run-first-noid', model_id='Model-A', score=0.8)
    manifest = json.loads((attempt / 'run_manifest.json').read_text(encoding='utf-8'))
    if blank_state == 'missing':
        manifest.pop('model_id')
        manifest.pop('model_config_identity')
    elif blank_state == 'none':
        manifest['model_id'] = None
        manifest['model_config_identity'] = None
    elif blank_state == 'empty':
        manifest['model_id'] = ''
        manifest['model_config_identity'] = ''
    elif blank_state == 'blank':
        manifest['model_id'] = '   '
        manifest['model_config_identity'] = '   '
    elif blank_state == 'tab':
        manifest['model_id'] = '\t'
        manifest['model_config_identity'] = '\t'
    elif blank_state == 'empty_model_id':
        manifest['model_id'] = ''
    else:
        manifest['model_config_identity'] = ''
    manifest['sample_manifest'] = [{
        'subset': 'default', 'selected': 1, 'sample_ids': ['0'],
        'question_digest': 'qd', 'media_digest': 'md', 'input_digest': 'id',
        'media_evidence': True,
    }]
    manifest['sample_manifest_identity'] = digest(manifest['sample_manifest'])
    (attempt / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    store = Store(tmp_path / 'results.db')
    try:
        result = runner.import_output(attempt, store, repo_dir=tmp_path, data_root=tmp_path)
        assert result['status'] == 'committed'
        assert result['validity_status'] == 'unverified'
        row = store.get_attempt('run-first-noid')
        if blank_state in ('missing', 'none', 'empty', 'blank', 'tab', 'empty_model_id'):
            assert row['model_id'] is None
        if blank_state in ('missing', 'none', 'empty', 'blank', 'tab', 'empty_config_identity'):
            assert row['model_config_identity'] is None
        assert 'import_identity_unverifiable' in row['status_reason']
        summary = render(store, tmp_path)
        assert 'run-first-noid' in summary.split('## Diagnostics')[1]

        # a later, complete artifact repairs the same run instead of staying
        # permanently uncertifiable; the repaired identity must be persisted
        good = tmp_path / 'good'
        _write_standalone_attempt(good, 'run-first-noid', model_id='Model-A', score=0.8)
        _mutate_manifest(good, sample_manifest=manifest['sample_manifest'],
                         sample_manifest_identity=manifest['sample_manifest_identity'])
        write_jsonl(good, 'predictions', dataset='gpqa_diamond', report_id='Model-A', count=1)
        write_jsonl(good, 'reviews', dataset='gpqa_diamond', report_id='Model-A', count=1)
        repaired = runner.import_output(good, store, repo_dir=tmp_path, data_root=tmp_path)
        assert repaired['status'] == 'committed'
        row = store.get_attempt('run-first-noid')
        assert row['model_id'] == 'Model-A'
        assert row['model_config_identity'] == 'mci-Model-A'
        assert (row['validity_status'], row['comparability']) == ('complete', 'verified')
        assert 'run-first-noid' in render(store, tmp_path).split('## Diagnostics')[0]

        # the repaired identity now protects the run from a different model
        other = tmp_path / 'other'
        _write_standalone_attempt(other, 'run-first-noid', model_id='Model-B', score=0.2)
        _mutate_manifest(other, sample_manifest=manifest['sample_manifest'],
                         sample_manifest_identity=manifest['sample_manifest_identity'])
        write_jsonl(other, 'predictions', dataset='gpqa_diamond', report_id='Model-B', count=1)
        write_jsonl(other, 'reviews', dataset='gpqa_diamond', report_id='Model-B', count=1)
        conflict = runner.import_output(other, store, repo_dir=tmp_path, data_root=tmp_path)
        assert conflict['status'] == 'rejected'
        assert store.get_attempt('run-first-noid')['model_id'] == 'Model-A'
        assert _primary_score(store, 'run-first-noid') == 0.8
    finally:
        store.close()


def test_first_import_with_consistent_identity_is_certified(tmp_path):
    attempt = tmp_path / 'attempt'
    _write_standalone_attempt(attempt, 'run-first-ok', model_id='Model-A', score=0.8)
    rows = [{'subset': 'gpqa_diamond', 'selected': 1, 'sample_ids': ['1']}]
    _mutate_manifest(attempt, sample_manifest=rows, sample_manifest_identity=digest(rows))

    store = Store(tmp_path / 'results.db')
    try:
        result = runner.import_output(attempt, store, repo_dir=tmp_path, data_root=tmp_path)
        assert result['status'] == 'committed' and result['validity_status'] == 'complete'
        row = store.get_attempt('run-first-ok')
        assert row['sample_manifest_identity'] == digest(rows)
        assert 'unverifiable' not in (row['status_reason'] or '')
    finally:
        store.close()
