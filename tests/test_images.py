import json
from pathlib import Path

from llmbench.images import cleanup_owned_images, collect_owned_candidates, collect_owned_images


def write_manifest(output_dir, created_at='2026-01-01T00:00:00+00:00'):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'run_manifest.json').write_text(
        json.dumps({'created_at': created_at}), encoding='utf-8')


def write_predictions(output_dir, entries, folder='predictions'):
    path = output_dir / folder / 'MiniMax' / 'dataset_default.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(entry) for entry in entries) + '\n', encoding='utf-8')


def test_collect_owned_images_only_from_swe_metadata(tmp_path):
    out = tmp_path / 'attempt'
    write_predictions(out, [
        {'index': 0, 'metadata': {'docker_image': 'swebench/sweb.eval.x86_64.django_1776_django-10097:latest'}},
        {'index': 1, 'metadata': {'docker_image': 'python:3.12'}},
        {'index': 2, 'metadata': {}},
    ])
    write_predictions(out, [
        {'index': 0, 'metadata': {'docker_image': 'sweb.eval.x86_64.other:latest'}},
    ], folder='reviews')
    (out / 'predictions' / 'MiniMax' / 'broken.jsonl').write_text('{oops\n', encoding='utf-8')

    assert collect_owned_images([out]) == {
        'swebench/sweb.eval.x86_64.django_1776_django-10097:latest',
        'sweb.eval.x86_64.other:latest',
    }


def test_collect_owned_candidates_records_attempt_start(tmp_path):
    out = tmp_path / 'attempt'
    write_manifest(out, created_at='2026-01-02T03:04:05+00:00')
    write_predictions(out, [{'index': 0, 'metadata': {'docker_image': 'swebench/a:latest'}}])
    assert collect_owned_candidates([out]) == {'swebench/a:latest': '2026-01-02T03:04:05+00:00'}


def test_preexisting_image_is_kept_even_if_reused(tmp_path):
    removed = []
    report = cleanup_owned_images(
        {'swebench/shared:latest'},
        created_after={'swebench/shared:latest': '2026-01-02T00:00:00+00:00'},
        image_created=lambda image: '2025-12-01T00:00:00+00:00',  # older than the run
        image_lister=lambda: ['swebench/shared:latest'],
        container_lister=lambda: [],
        remover=removed.extend,
    )
    assert report['removed'] == []
    assert removed == []
    assert 'swebench/shared:latest' in report['unproven']


def test_image_created_by_this_run_is_removed():
    removed = []
    report = cleanup_owned_images(
        {'swebench/built:latest'},
        created_after={'swebench/built:latest': '2026-01-02T00:00:00+00:00'},
        image_created=lambda image: '2026-01-02T00:10:00+00:00',
        image_lister=lambda: ['swebench/built:latest'],
        container_lister=lambda: [],
        remover=removed.extend,
    )
    assert report['removed'] == ['swebench/built:latest']
    assert removed == ['swebench/built:latest']


def test_in_use_image_is_kept():
    report = cleanup_owned_images(
        {'swebench/in-use:latest'},
        created_after={'swebench/in-use:latest': '2026-01-02T00:00:00+00:00'},
        image_created=lambda image: '2026-01-02T00:10:00+00:00',
        image_lister=lambda: ['swebench/in-use:latest'],
        container_lister=lambda: ['swebench/in-use:latest'],
        remover=lambda names: None,
    )
    assert report['removed'] == []
    assert report['in_use'] == ['swebench/in-use:latest']


def test_uninspectable_image_is_kept():
    def broken_inspect(image):
        raise RuntimeError('docker daemon unavailable')

    report = cleanup_owned_images(
        {'swebench/unknown:latest'},
        created_after={'swebench/unknown:latest': '2026-01-02T00:00:00+00:00'},
        image_created=broken_inspect,
        image_lister=lambda: ['swebench/unknown:latest'],
        container_lister=lambda: [],
        remover=lambda names: None,
    )
    assert report['removed'] == []
    assert 'swebench/unknown:latest' in report['unproven']


def test_removal_failure_is_reported_not_claimed():
    class Failure:
        returncode = 1
        stderr = 'conflict: image is being used'

    report = cleanup_owned_images(
        {'swebench/built:latest'},
        created_after={'swebench/built:latest': '2026-01-02T00:00:00+00:00'},
        image_created=lambda image: '2026-01-02T00:10:00+00:00',
        image_lister=lambda: ['swebench/built:latest'],
        container_lister=lambda: [],
        remover=lambda names: Failure(),
    )
    assert report['removed'] == []
    assert report['failed'] and 'conflict' in report['failed'][0]['error']


def test_unrelated_images_are_never_touched():
    removed = []
    report = cleanup_owned_images(
        {'swebench/owned:latest'},
        created_after={'swebench/owned:latest': '2026-01-02T00:00:00+00:00'},
        image_created=lambda image: '2026-01-02T00:10:00+00:00',
        image_lister=lambda: ['swebench/owned:latest', 'someone-elses:latest'],
        container_lister=lambda: [],
        remover=removed.extend,
    )
    assert report['removed'] == ['swebench/owned:latest']
    assert removed == ['swebench/owned:latest']


def test_candidate_mapping_is_accepted_directly():
    removed = []
    report = cleanup_owned_images(
        {'swebench/a:latest': '2026-01-02T00:00:00+00:00'},
        image_created=lambda image: '2026-01-02T00:10:00+00:00',
        image_lister=lambda: ['swebench/a:latest'],
        container_lister=lambda: [],
        remover=removed.extend,
    )
    assert report['removed'] == ['swebench/a:latest']
    assert removed == ['swebench/a:latest']
