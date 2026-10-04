"""Build the pinned single-instance SWE-bench dataset, with source provenance."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .config import validate_pinned_content, validate_pinned_dir
from .util import file_digest, now_iso

DJANGO_DATASET_NAME = 'swe_bench_verified_django1'
DJANGO_INSTANCE_ID = 'django__django-10097'
SWE_BENCH_REPO = 'princeton-nlp/SWE-bench_Verified'

_REVISION_RE = re.compile(r'([0-9a-f]{40})')


def _cached_revision(dataset) -> str:
    for cache_file in getattr(dataset, 'cache_files', []) or []:
        match = _REVISION_RE.search(str(cache_file.get('filename', '')))
        if match:
            return match.group(1)
    return 'unknown'


def prepare_django_dataset(data_root: Path, *, force: bool = False) -> dict:
    import datasets

    out_dir = Path(data_root) / 'pinned' / DJANGO_DATASET_NAME
    if out_dir.exists() and not force:
        ok, reason = validate_pinned_dir(out_dir, DJANGO_INSTANCE_ID)
        content_ok, content_reason = validate_pinned_content(out_dir, DJANGO_INSTANCE_ID)
        if ok and content_ok:
            return {'status': 'exists', 'path': str(out_dir), 'reason': content_reason}
        # An incomplete, corrupt or checksum-mismatching pinned dir is rebuilt
        # instead of being trusted.

    dataset = datasets.load_dataset(SWE_BENCH_REPO, split='test')
    revision = _cached_revision(dataset)
    revision_source = 'hf-datasets cache snapshot path' if revision != 'unknown' else 'unknown'
    pinned = dataset.filter(lambda record: record['instance_id'] == DJANGO_INSTANCE_ID)
    if len(pinned) != 1:
        raise RuntimeError(f'expected exactly one {DJANGO_INSTANCE_ID}, found {len(pinned)}')

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(out_dir.parent), prefix=f'.{DJANGO_DATASET_NAME}.staging.'))
    try:
        parquet_path = staging / 'test-00000-of-00001.parquet'
        pinned.to_parquet(str(parquet_path))
        if parquet_path.stat().st_size == 0:
            raise RuntimeError('prepared parquet is empty')
        source = {
            'repo': SWE_BENCH_REPO,
            'revision': revision,
            'revision_source': revision_source,
            'instance_ids': [DJANGO_INSTANCE_ID],
            'parquet_sha256': file_digest(parquet_path),
            'created_at': now_iso(),
        }
        (staging / 'source.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
        ok, reason = validate_pinned_dir(staging)
        if not ok:
            raise RuntimeError(f'prepared dataset failed validation: {reason}')
        content_ok, content_reason = validate_pinned_content(staging, DJANGO_INSTANCE_ID)
        if not content_ok:
            raise RuntimeError(f'prepared dataset failed content validation: {content_reason}')

        backup = out_dir.with_name(out_dir.name + '.old')
        if backup.exists():
            shutil.rmtree(backup)
        if out_dir.exists():
            os.replace(out_dir, backup)
        os.replace(staging, out_dir)
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {'status': 'created', 'path': str(out_dir), 'revision': revision}
