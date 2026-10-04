import json
from pathlib import Path

import pytest

from llmbench.config import (
    ConfigError,
    PlanEntry,
    build_plan,
    check_entry_requirements,
    endpoint_identity,
    model_config_identity,
    preflight,
    protocol_identity,
)

MODELS = {
    'minimax': {
        'model_id': 'MiniMax-M3.1-Flash-Preview',
        'api_url': 'https://api.minimax.cn/v1',
        'api_key_env': 'LLMBENCH_TEST_KEY',
        'generation_config': {'temperature': 0.0, 'max_tokens': 32768},
    },
}

SUITES = {
    'profiles': {'smoke': {'limit': 5}, 'lite': {'limit': 20}, 'full': {'limit': None}},
    'defaults': {'eval_batch_size': 4},
    'suites': {
        'knowledge': {
            'datasets': {
                'gpqa_diamond': {'core': True, 'status': 'tested'},
                'mmlu_pro': {'core': False, 'full_per_subset_limit': 500},
            },
        },
        'swe': {
            'eval_batch_size': 1,
            'sandbox': {'enabled': True, 'engine': 'docker'},
            'datasets': {
                'live_code_bench': {
                    'core': True, 'requires_sandbox': True, 'requires': ['ms_enclave'],
                },
                'swe_bench_pro': {'core': False, 'requires_sandbox': True},
                'swe_bench_verified_agentic': {
                    'core': True, 'pinned': True, 'requires_sandbox': True,
                    'pinned_instance_id': 'django__django-10097',
                    'local_path': '${DATA_ROOT}/pinned/ds',
                },
            },
        },
        'vision': {
            'datasets': {
                'docvqa': {'core': True, 'status': 'tested'},
                'chartqa': {'core': False, 'status': 'configured'},
            },
        },
    },
}


def plan(tmp_path, **kwargs):
    defaults = dict(
        models_cfg=MODELS, suites_doc=SUITES, data_root=tmp_path,
        model_names=['minimax'], suite_names=['knowledge'], profile='lite',
    )
    defaults.update(kwargs)
    return build_plan(**defaults)


def test_limit_resolution_smoke_lite_full_cap(tmp_path):
    assert plan(tmp_path, profile='smoke').entries[0].limit == 5
    assert plan(tmp_path, profile='lite').entries[0].limit == 20
    assert plan(tmp_path, suite_names=['knowledge'], profile='full').entries[0].limit is None
    capped = plan(tmp_path, suite_names=['knowledge'], profile='full', include_extensions=True)
    mmlu = [entry for entry in capped.entries if entry.dataset == 'mmlu_pro'][0]
    assert mmlu.limit == 500


def test_cli_limit_overrides_profile(tmp_path):
    assert plan(tmp_path, limit=3).entries[0].limit == 3


def test_pinned_limit_is_one(tmp_path):
    entries = plan(tmp_path, suite_names=['swe'], profile='full', include_extensions=False).entries
    pinned = [entry for entry in entries if entry.dataset == 'swe_bench_verified_agentic'][0]
    assert pinned.limit == 1
    assert pinned.local_path == tmp_path / 'pinned' / 'ds'
    assert pinned.pinned_instance == 'django__django-10097'


def test_explicit_datasets_override_core_filter(tmp_path):
    entries = plan(tmp_path, suite_names=['vision'], datasets=['chartqa'], include_extensions=False).entries
    assert [entry.dataset for entry in entries] == ['chartqa']


def test_dataset_from_wrong_suite_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match='belongs to suite'):
        plan(tmp_path, suite_names=['knowledge'], datasets=['chartqa'])


def test_unknown_dataset_rejected(tmp_path):
    with pytest.raises(ConfigError, match='unknown dataset'):
        plan(tmp_path, datasets=['nope'])


def test_empty_selection_rejected(tmp_path):
    # only non-core datasets in a one-dataset suite
    suites = json.loads(json.dumps(SUITES))
    suites['suites']['vision']['datasets']['docvqa']['core'] = False
    with pytest.raises(ConfigError, match='empty'):
        plan(tmp_path, suites_doc=suites, suite_names=['vision'])


def test_invalid_limit_and_batch_rejected(tmp_path):
    with pytest.raises(ConfigError, match='--limit'):
        plan(tmp_path, limit=0)
    with pytest.raises(ConfigError, match='--batch-size'):
        plan(tmp_path, batch_size=0)


def test_duplicate_model_selection_is_deduplicated(tmp_path):
    entries = plan(tmp_path, model_names=['minimax', 'minimax']).entries
    assert [entry.model_alias for entry in entries] == ['minimax']


def test_model_identity_tracks_endpoint_and_id():
    base = model_config_identity('minimax', MODELS['minimax'])
    assert base == model_config_identity('minimax', MODELS['minimax'])
    changed_id = dict(MODELS['minimax'], model_id='other')
    changed_url = dict(MODELS['minimax'], api_url='https://other.example/v1')
    assert base != model_config_identity('minimax', changed_id)
    assert base != model_config_identity('minimax', changed_url)
    # credentials never change the identity
    assert base == model_config_identity('minimax', dict(MODELS['minimax'], api_url='https://u:p@api.minimax.cn/v1'))


def test_endpoint_identity_hides_credentials():
    assert 'u:p' not in endpoint_identity('https://u:p@api.example.com/v1?x=1')
    assert endpoint_identity('https://u:p@api.example.com/v1?x=1') == 'https://api.example.com/v1'


def test_protocol_identity_axes():
    resolved = {
        'datasets': ['gpqa_diamond'],
        'generation_config': {'temperature': 0.0},
        'agent_config': {'mode': 'native', 'max_steps': 250},
        'dataset_args': {'swe_bench_verified_agentic': {'local_path': '/data/a'}},
        'model': 'whatever', 'api_key': 'secret', 'work_dir': '/tmp/x', 'limit': 20,
    }
    base = protocol_identity(resolved)
    # model identity and local data location are not protocol axes
    assert base == protocol_identity({**resolved, 'model': 'other', 'api_key': 'other',
                                      'work_dir': '/tmp/y',
                                      'dataset_args': {'swe_bench_verified_agentic': {'local_path': '/data/b'}}})
    assert base != protocol_identity({**resolved, 'generation_config': {'temperature': 0.7}})
    assert base != protocol_identity({**resolved, 'agent_config': {'mode': 'native', 'max_steps': 100}})
    # sample count lives in the manifest identity, not the protocol identity
    assert base == protocol_identity({**resolved, 'limit': 5})
    # pure storage locations are not part of the protocol
    assert base == protocol_identity({**resolved, 'dataset_dir': '/disk-a/datasets'})
    assert protocol_identity({**resolved, 'dataset_dir': '/disk-a/datasets'}) == protocol_identity(
        {**resolved, 'dataset_dir': '/disk-b/datasets'})
    # dataset behavior knobs are protocol
    assert base != protocol_identity({**resolved, 'dataset_args': {
        'gpqa_diamond': {'extra_params': {'x': 1}}}})


def test_preflight_rejects_pinned_parquet_with_wrong_instance(tmp_path, monkeypatch):
    pyarrow = pytest.importorskip('pyarrow')
    import hashlib
    import json as json_module
    import pyarrow.parquet as pq

    pinned = tmp_path / 'pinned' / 'ds'
    pinned.mkdir(parents=True)
    parquet = pinned / 'test-00000-of-00001.parquet'
    pq.write_table(pyarrow.table({'instance_id': ['wrong-instance']}), parquet)
    sha = hashlib.sha256(parquet.read_bytes()).hexdigest()
    (pinned / 'source.json').write_text(json_module.dumps({
        'instance_ids': ['django__django-10097'], 'revision': 'a' * 40, 'parquet_sha256': sha,
    }))

    entry = PlanEntry(
        model_alias='minimax', model_cfg=MODELS['minimax'], suite='swe',
        dataset='swe_bench_verified_agentic', spec={}, profile='smoke', limit=1,
        batch_size=1, pinned=True, local_path=pinned, status='tested',
        pinned_instance='django__django-10097',
    )
    errors = check_entry_requirements(entry, check_keys=False, check_docker=False, check_pinned=True)
    assert any('instance_id' in error for error in errors)


def test_preflight_requires_pinned_instance_id_in_config(tmp_path):
    entry = PlanEntry(
        model_alias='minimax', model_cfg=MODELS['minimax'], suite='swe',
        dataset='swe_bench_verified_agentic', spec={}, profile='smoke', limit=1,
        batch_size=1, pinned=True, local_path=tmp_path / 'pinned', status='tested',
    )
    errors = check_entry_requirements(entry, check_keys=False, check_docker=False, check_pinned=True)
    assert any('pinned_instance_id' in error for error in errors)


def test_suite_sandbox_policy_is_inherited_and_satisfied(tmp_path, monkeypatch):
    entries = plan(tmp_path, suite_names=['swe'], include_extensions=False).entries
    sandbox_entry = [entry for entry in entries if entry.dataset == 'live_code_bench'][0]
    assert sandbox_entry.spec['sandbox'] == {'enabled': True, 'engine': 'docker'}
    monkeypatch.setenv('LLMBENCH_TEST_KEY', 'x')
    monkeypatch.setattr('llmbench.config.docker_available', lambda: True)
    monkeypatch.setattr('llmbench.config.module_available', lambda name: True)
    assert check_entry_requirements(
        sandbox_entry, check_keys=True, check_docker=True, check_pinned=False) == []


def test_preflight_checks_keys_modules_sandbox_and_pinned(tmp_path, monkeypatch):
    entries = plan(tmp_path, suite_names=['swe'], include_extensions=False).entries
    sandbox_entry = [entry for entry in entries if entry.dataset == 'live_code_bench'][0]

    monkeypatch.delenv('LLMBENCH_TEST_KEY', raising=False)
    assert any('LLMBENCH_TEST_KEY' in error for error in check_entry_requirements(
        sandbox_entry, check_keys=True, check_docker=True, check_pinned=True))

    monkeypatch.setattr('llmbench.config.module_available', lambda name: False)
    assert any('ms_enclave' in error for error in check_entry_requirements(
        sandbox_entry, check_keys=False, check_docker=True, check_pinned=True))

    monkeypatch.setattr('llmbench.config.docker_available', lambda: False)
    errors = check_entry_requirements(sandbox_entry, check_keys=False, check_docker=True, check_pinned=True)
    assert any('docker' in error for error in errors)

    pinned_entry = [entry for entry in entries if entry.dataset == 'swe_bench_verified_agentic'][0]
    errors = check_entry_requirements(pinned_entry, check_keys=False, check_docker=True, check_pinned=True)
    assert any('prepare' in error for error in errors)


def test_preflight_rejects_pinned_parquet_that_is_not_readable(tmp_path, monkeypatch):
    import hashlib

    pinned = tmp_path / 'pinned' / 'ds'
    pinned.mkdir(parents=True)
    parquet = pinned / 'test-00000-of-00001.parquet'
    parquet.write_bytes(b'not-parquet')
    (pinned / 'source.json').write_text(json.dumps({
        'instance_ids': ['django__django-10097'], 'revision': 'a' * 40,
        'parquet_sha256': hashlib.sha256(b'not-parquet').hexdigest(),
    }))
    from llmbench.config import validate_pinned_dir
    assert validate_pinned_dir(pinned, 'django__django-10097')[0] is True  # hash matches

    entry = PlanEntry(
        model_alias='minimax', model_cfg=MODELS['minimax'], suite='swe',
        dataset='swe_bench_verified_agentic', spec={}, profile='smoke', limit=1,
        batch_size=1, pinned=True, local_path=pinned, status='tested',
        pinned_instance='django__django-10097',
    )
    errors = check_entry_requirements(entry, check_keys=False, check_docker=False, check_pinned=True)
    assert any('parquet' in error or 'pyarrow' in error for error in errors)


def test_sandbox_disabled_refuses_host_fallback(tmp_path):
    entry = plan(tmp_path, suite_names=['swe'], include_extensions=False).entries[0]
    entry.spec = {**entry.spec, 'sandbox': {'enabled': False}}
    errors = check_entry_requirements(entry, check_keys=False, check_docker=True, check_pinned=False)
    assert any('sandbox.enabled=true' in error for error in errors)


def test_preflight_aggregates_errors(tmp_path, monkeypatch):
    monkeypatch.delenv('LLMBENCH_TEST_KEY', raising=False)
    errors = preflight(plan(tmp_path, suite_names=['knowledge']))
    assert errors
