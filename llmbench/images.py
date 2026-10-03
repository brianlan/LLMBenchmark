"""Docker image ownership: only images proven to belong to the current run are removable."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

SWE_IMAGE_PREFIXES = ('swebench/', 'sweb.eval', 'sweb.env', 'sweap', 'jefzda/sweap')


def _image_is_swe(name: str) -> bool:
    return name.startswith(SWE_IMAGE_PREFIXES)


def collect_owned_images(output_dirs) -> set:
    """Images referenced by this run's own prediction metadata."""
    images = set()
    for output_dir in output_dirs:
        for folder in ('predictions', 'reviews'):
            root = Path(output_dir) / folder
            if not root.exists():
                continue
            for path in root.rglob('*.jsonl'):
                try:
                    with open(path, encoding='utf-8') as handle:
                        for line in handle:
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            metadata = data.get('metadata') or {}
                            image = metadata.get('docker_image')
                            if image and _image_is_swe(image):
                                images.add(image)
                except OSError:
                    continue
    return images


def _docker_output(args) -> list:
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f'docker command failed: {args}')
    return result.stdout.splitlines()


def cleanup_owned_images(images, *, image_lister=None, container_lister=None, remover=None) -> dict:
    """Remove only the given, provably-owned images.

    Shared/pre-existing images are kept: an image is removed only when it appears in
    ``images`` and no container (running or stopped) still uses it.
    """
    image_lister = image_lister or (lambda: _docker_output(
        ['docker', 'image', 'ls', '--format', '{{.Repository}}:{{.Tag}}']))
    container_lister = container_lister or (lambda: _docker_output(
        ['docker', 'ps', '-a', '--format', '{{.Image}}']))
    remover = remover or (lambda names: subprocess.run(['docker', 'rmi', *names], check=False))

    available = set(image_lister())
    in_use = set(container_lister())
    removable = sorted(image for image in images if image in available and image not in in_use)
    report = {
        'requested': sorted(images),
        'available': sorted(images & available),
        'in_use': sorted(images & in_use),
        'removed': removable,
        'kept': sorted(images - set(removable)),
    }
    if removable:
        remover(removable)
    return report
