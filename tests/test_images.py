import json

from llmbench.images import cleanup_owned_images, collect_owned_images


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


def test_cleanup_keeps_shared_and_in_use_images():
    removed = []

    def remover(names):
        removed.extend(names)

    report = cleanup_owned_images(
        {'swebench/a:latest', 'sweb.eval.x:1'},
        image_lister=lambda: ['swebench/a:latest', 'sweb.eval.x:1', 'unrelated:1'],
        container_lister=lambda: ['sweb.eval.x:1'],
        remover=remover,
    )
    assert report['removed'] == ['swebench/a:latest']
    assert report['in_use'] == ['sweb.eval.x:1']
    assert 'sweb.eval.x:1' in report['kept']
    assert removed == ['swebench/a:latest']


def test_cleanup_never_touches_images_outside_the_owned_set():
    removed = []
    report = cleanup_owned_images(
        {'swebench/owned:latest'},
        image_lister=lambda: ['swebench/owned:latest', 'someone-elses:latest'],
        container_lister=lambda: [],
        remover=removed.extend,
    )
    assert report['removed'] == ['swebench/owned:latest']  # only the owned one, never a global prune
    assert removed == ['swebench/owned:latest']
