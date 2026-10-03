import json
import math

import pytest

from llmbench.util import (
    atomic_write_text,
    canonical_json,
    collect_secrets,
    digest,
    is_secret_key,
    redact,
    redact_text,
    safe_slug,
)


def test_canonical_json_is_stable():
    assert canonical_json({'b': 1, 'a': [1, 2]}) == '{"a":[1,2],"b":1}'
    assert digest({'a': 1, 'b': 2}) == digest({'b': 2, 'a': 1})


def test_canonical_json_rejects_non_finite():
    with pytest.raises(ValueError):
        canonical_json({'x': math.nan})
    with pytest.raises(ValueError):
        canonical_json({'x': math.inf})


def test_redact_hides_nested_secrets_and_known_values():
    payload = {
        'api_key': 'sk-live-123',
        'nested': {'authorization': 'Bearer xyz', 'keep': 'ok'},
        'list': [{'token': 'abc'}],
        'text': 'prefix sk-live-123 suffix',
    }
    redacted = redact(payload, secrets=['sk-live-123'])
    assert redacted['api_key'] == '***'
    assert redacted['nested']['authorization'] == '***'
    assert redacted['nested']['keep'] == 'ok'
    assert redacted['list'][0]['token'] == '***'
    assert 'sk-live-123' not in redacted['text']
    assert '***' in redacted['text']


def test_redact_text_strips_url_credentials():
    assert 'user:pass' not in redact_text('https://user:pass@api.example.com/v1')
    assert '***@api.example.com' in redact_text('https://user:pass@api.example.com/v1')


def test_secret_key_detection():
    assert is_secret_key('api_key')
    assert is_secret_key('Authorization')
    assert is_secret_key('api-key')
    assert is_secret_key('refresh_token')
    assert not is_secret_key('display_kind')
    # counters must never be redacted
    assert not is_secret_key('input_tokens')
    assert not is_secret_key('output_tokens')
    assert not is_secret_key('total_tokens')
    assert not is_secret_key('max_tokens')


def test_redact_keeps_usage_counts_and_generation_limits():
    payload = {
        'usage_totals': {'input_tokens': 1234, 'output_tokens': 56, 'total_tokens': 1290},
        'generation_config': {'max_tokens': 32768, 'temperature': 0.0},
    }
    assert redact(payload, ['sk-live-123']) == payload


def test_collect_secrets_reads_env(monkeypatch):
    monkeypatch.setenv('LLMBENCH_TEST_KEY', 'env-secret')
    assert 'env-secret' in collect_secrets({'api_key_env': 'LLMBENCH_TEST_KEY'})
    assert collect_secrets({'api_key': 'literal'}) == ['literal']


def test_atomic_write_text(tmp_path):
    target = tmp_path / 'nested' / 'file.txt'
    atomic_write_text(target, 'hello')
    assert target.read_text() == 'hello'
    assert list(target.parent.glob('.*.tmp')) == []


def test_safe_slug():
    assert safe_slug('a/b c') == 'a-b-c'
