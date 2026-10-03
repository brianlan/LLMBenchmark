import json
import sys
import types
from pathlib import Path

import pytest

from llmbench.config import validate_pinned_dir
from llmbench.prepare import DJANGO_INSTANCE_ID, prepare_django_dataset

REVISION = 'a' * 40


class FakePinned:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def to_parquet(self, path):
        Path(path).write_bytes(b'parquet-bytes')


class FakeDataset:
    cache_files = [{'filename': f'/cache/datasets--p--s/snapshots/{REVISION}/data.arrow'}]

    def __init__(self, records):
        self.records = records

    def filter(self, predicate):
        return FakePinned([record for record in self.records if predicate(record)])


def stub_datasets(monkeypatch, records):
    module = types.ModuleType('datasets')
    module.load_dataset = lambda repo, split: FakeDataset(records)
    monkeypatch.setitem(sys.modules, 'datasets', module)


def test_prepare_builds_pinned_dir_atomically(tmp_path, monkeypatch):
    stub_datasets(monkeypatch, [{'instance_id': DJANGO_INSTANCE_ID}, {'instance_id': 'other'}])
    result = prepare_django_dataset(tmp_path)
    assert result['status'] == 'created'
    assert result['revision'] == REVISION
    out = Path(result['path'])
    source = json.loads((out / 'source.json').read_text())
    assert source['instance_ids'] == [DJANGO_INSTANCE_ID]
    assert source['revision'] == REVISION
    assert (out / 'test-00000-of-00001.parquet').stat().st_size > 0
    assert validate_pinned_dir(out, DJANGO_INSTANCE_ID)[0] is True
    leftovers = list(out.parent.iterdir())
    assert leftovers == [out]


def test_prepare_reuses_valid_existing_dir(tmp_path, monkeypatch):
    stub_datasets(monkeypatch, [{'instance_id': DJANGO_INSTANCE_ID}])
    prepare_django_dataset(tmp_path)
    second = prepare_django_dataset(tmp_path)
    assert second['status'] == 'exists'


def test_prepare_replaces_invalid_existing_dir(tmp_path, monkeypatch):
    out = tmp_path / 'pinned' / 'swe_bench_verified_django1'
    out.mkdir(parents=True)
    (out / 'source.json').write_text('{broken')
    stub_datasets(monkeypatch, [{'instance_id': DJANGO_INSTANCE_ID}])
    result = prepare_django_dataset(tmp_path)
    assert result['status'] == 'created'
    assert validate_pinned_dir(out, DJANGO_INSTANCE_ID)[0] is True


def test_prepare_fails_loudly_when_instance_is_missing(tmp_path, monkeypatch):
    stub_datasets(monkeypatch, [{'instance_id': 'other'}])
    with pytest.raises(RuntimeError, match='exactly one'):
        prepare_django_dataset(tmp_path)
    parent = tmp_path / 'pinned'
    assert not parent.exists() or list(parent.iterdir()) == []


def test_validate_pinned_dir_rejection_reasons(tmp_path):
    ok, reason = validate_pinned_dir(tmp_path / 'nope')
    assert not ok and 'missing' in reason

    root = tmp_path / 'pinned'
    root.mkdir()
    ok, reason = validate_pinned_dir(root)
    assert not ok and 'source.json' in reason

    (root / 'source.json').write_text(json.dumps({'instance_ids': [DJANGO_INSTANCE_ID]}))
    ok, reason = validate_pinned_dir(root)
    assert not ok and 'parquet' in reason

    (root / 'test-00000-of-00001.parquet').write_bytes(b'')
    ok, reason = validate_pinned_dir(root)
    assert not ok and 'parquet' in reason

    (root / 'test-00000-of-00001.parquet').write_bytes(b'data')
    (root / 'source.json').write_text('{broken')
    ok, reason = validate_pinned_dir(root)
    assert not ok and 'corrupt' in reason

    (root / 'source.json').write_text(json.dumps({'instance_ids': ['a', 'b']}))
    ok, reason = validate_pinned_dir(root)
    assert not ok and 'exactly one' in reason

    (root / 'source.json').write_text(json.dumps({'instance_ids': ['a']}))
    ok, reason = validate_pinned_dir(root, DJANGO_INSTANCE_ID)
    assert not ok and 'expected' in reason

    (root / 'source.json').write_text(json.dumps(
        {'instance_ids': [DJANGO_INSTANCE_ID], 'revision': REVISION}))
    ok, reason = validate_pinned_dir(root, DJANGO_INSTANCE_ID)
    assert ok and reason == REVISION
