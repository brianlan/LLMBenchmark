import json

from llmbench.store import Store
from llmbench.summary import TOP_CATEGORY_KEY, render


def seed_run(store, run_id, *, model_alias='minimax', model_config='mc1', dataset='gpqa_diamond',
             profile='full', validity='complete', comparability='verified', score=0.8,
             metric='accuracy', finished='2026-01-01T00:00:01', protocol='proto-1',
             samples='samples-1', exec_status='completed', reason='complete',
             eval_version='1.12.0', display_kind='percent', unit='%', metrics=None):
    store.start_attempt({
        'run_id': run_id, 'run_group': 'g', 'model_alias': model_alias, 'model_id': 'm',
        'api_url': 'https://api.example/v1', 'model_config_identity': model_config,
        'suite': 'knowledge', 'profile': profile, 'dataset': dataset,
        'execution_status': 'running', 'validity_status': None, 'status_reason': None,
        'phase': 'preflight', 'comparability': 'unknown', 'started_at': '2026-01-01T00:00:00',
        'evalscope_version': eval_version, 'tool_version': '0.2.0', 'git_commit': 'abc',
        'output_dir': f'/tmp/{run_id}', 'config_json': '{}', 'protocol_identity': protocol,
        'sample_manifest_identity': samples,
    })
    store.finish_attempt(run_id, {
        'execution_status': exec_status, 'validity_status': validity, 'status_reason': reason,
        'phase': 'persist', 'comparability': comparability, 'finished_at': finished,
        'num_requested': 20, 'num_succeeded': 20, 'num_errored': 0, 'incomplete': False,
        'report_path': 'p', 'diagnostics': {'generation_calls': 20},
        'perf_metrics': None, 'primary_metric_identity': {'name': metric},
        'protocol_identity': protocol, 'sample_manifest_identity': samples,
        'metrics': metrics if metrics is not None else [{
            'metric_key': f'mk-{metric}', 'metric_name': metric, 'aggregation': 'mean',
            'dimensions': {}, 'category_key': TOP_CATEGORY_KEY, 'category': [], 'subset': '',
            'num': 20, 'score': score, 'macro_score': None, 'semantics_kind': 'quality',
            'display_kind': display_kind, 'direction': 'higher', 'unit': unit, 'is_primary': True,
        }],
        'sample_manifest': [{
            'subset': 'default', 'selected': 20, 'predicted': 20, 'reviewed': 20,
            'sample_ids': ['0'], 'question_digest': 'q', 'media_digest': 'm',
            'input_digest': 'i', 'media_evidence': True,
        }],
    })


def test_smoke_never_replaces_full_in_formal_table(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'full-run', profile='full', score=0.80, finished='2026-01-01T00:00:01')
    seed_run(store, 'smoke-run', profile='smoke', score=1.00, finished='2026-01-02T00:00:01')
    text = render(store, tmp_path)
    formal = text.split('## Diagnostics')[0]
    assert 'full-run' in formal
    assert 'smoke-run' not in formal
    diagnostics = text.split('## Diagnostics')[1]
    assert 'smoke-run' in diagnostics
    store.close()


def test_partial_does_not_replace_complete(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'complete-run', validity='complete', score=0.80, finished='2026-01-01T00:00:01')
    seed_run(store, 'partial-run', validity='partial', score=0.20, finished='2026-01-02T00:00:01',
             reason='partial_completion: succeeded=1/20')
    text = render(store, tmp_path)
    formal = text.split('## Diagnostics')[0]
    assert 'complete-run' in formal and 'partial-run' not in formal
    assert 'partial-run' in text.split('## Diagnostics')[1]
    store.close()


def test_allow_partial_shows_partial(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'partial-run', validity='partial', score=0.20, reason='partial_completion')
    text = render(store, tmp_path, include_partial=True)
    assert 'partial-run' in text.split('## Diagnostics')[0]
    store.close()


def test_same_alias_with_different_model_config_is_not_merged(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'old-endpoint', model_config='mc-old', score=0.10)
    seed_run(store, 'new-endpoint', model_config='mc-new', score=0.90)
    text = render(store, tmp_path)
    formal = text.split('## Diagnostics')[0]
    assert 'old-endpoint' in formal and 'new-endpoint' in formal
    assert '| `mc-old`' in formal or 'mc-old' in formal
    assert 'mc-new' in formal
    store.close()


def test_same_protocol_two_models_share_one_group(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'run-a', model_alias='alpha', model_config='mc-a', score=0.5)
    seed_run(store, 'run-b', model_alias='beta', model_config='mc-b', score=0.7)
    text = render(store, tmp_path)
    formal = text.split('## Diagnostics')[0]
    assert '### protocol `proto-1`' in formal
    assert 'alpha' in formal and 'beta' in formal
    store.close()


def test_different_protocol_budget_makes_separate_groups(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'budget-250', protocol='proto-250', samples='samples-a', score=0.9)
    seed_run(store, 'budget-100', protocol='proto-100', samples='samples-b', score=0.5)
    text = render(store, tmp_path)
    formal = text.split('## Diagnostics')[0]
    assert '### protocol `proto-250`' in formal
    assert '### protocol `proto-100`' in formal
    store.close()


def test_same_finished_at_is_stable_and_explicit(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'run-x', score=0.1, finished='2026-01-01T00:00:01')
    seed_run(store, 'run-y', score=0.2, finished='2026-01-01T00:00:01')
    first = render(store, tmp_path)
    second = render(store, tmp_path)
    assert first == second
    store.close()


def test_null_score_displays_na_and_versions_are_visible(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'null-score', score=None, eval_version='1.12.0')
    text = render(store, tmp_path)
    assert 'N/A' in text
    assert '1.12.0' in text
    assert 'tool version is the renderer' in text.lower() or 'Tool version is the renderer' in text
    store.close()


def test_verified_run_without_primary_metric_is_still_visible(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'no-primary-metric', metrics=[], reason='no_metrics')
    text = render(store, tmp_path)
    assert 'no-primary-metric' not in text.split('## Diagnostics')[0]
    assert 'no-primary-metric' in text.split('## Diagnostics')[1]
    store.close()


def test_legacy_runs_only_in_diagnostics(tmp_path):
    store = Store(tmp_path / 'results.db')
    seed_run(store, 'legacy-run', validity='legacy', comparability='legacy', reason='migrated from v1')
    text = render(store, tmp_path)
    assert 'legacy-run' not in text.split('## Diagnostics')[0]
    assert 'legacy-run' in text.split('## Diagnostics')[1]
    store.close()
