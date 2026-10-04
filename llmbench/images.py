"""Docker image ownership: only images proven to belong to the current run are removable.

Referencing an image in a prediction is not proof of ownership: a shared cache
image can be reused by this run without being created by it.  An image is only
removable when it exists, no container uses it, and its Docker creation time is
not older than the attempt that referenced it.  Anything unprovable is kept.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SWE_IMAGE_PREFIXES = ('swebench/', 'sweb.eval', 'sweb.env', 'sweap', 'jefzda/sweap')


def _image_is_swe(name: str) -> bool:
    return name.startswith(SWE_IMAGE_PREFIXES)


def _referenced_images(output_dir) -> set:
    images = set()
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


def collect_owned_images(output_dirs) -> set:
    """Images referenced by this run's own prediction metadata (candidates)."""
    images = set()
    for output_dir in output_dirs:
        images |= _referenced_images(output_dir)
    return images


def _attempt_started_at(output_dir) -> str | None:
    manifest = Path(output_dir) / 'run_manifest.json'
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('created_at'):
            return str(data['created_at'])
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    try:
        return datetime.fromtimestamp(Path(output_dir).stat().st_mtime,
                                      tz=timezone.utc).astimezone().isoformat()
    except OSError:
        return None


def collect_owned_candidates(output_dirs) -> dict:
    """Map each referenced image to the attempt that referenced it.

    Each value is ``{'attempt_start': iso, 'baseline': [images] | None}`` where
    ``baseline`` is the pre-run image inventory recorded in the attempt
    manifest (None means the inventory could not be taken).
    """
    candidates = {}
    for output_dir in output_dirs:
        manifest = Path(output_dir) / 'run_manifest.json'
        started_at = _attempt_started_at(output_dir)
        baseline = None
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            baseline = data.get('baseline_swe_images')
        except (OSError, json.JSONDecodeError, ValueError):
            baseline = None
        entry = {'attempt_start': started_at, 'baseline': baseline}
        for image in _referenced_images(output_dir):
            current = candidates.get(image)
            if current is None:
                candidates[image] = entry
            elif current.get('baseline') is None and baseline is not None:
                candidates[image] = entry
            elif baseline is not None and started_at and (
                    current.get('attempt_start') or '~') > started_at:
                candidates[image] = entry
    return candidates


def _docker_output(args) -> list:
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f'docker command failed: {args}')
    return result.stdout.splitlines()


def _parse_timestamp(value):
    if not value:
        return None
    text = str(value).strip().replace('Z', '+00:00')
    # Docker returns nanosecond precision; datetime handles at most microseconds.
    if '.' in text:
        head, rest = text.split('.', 1)
        fraction = rest.split('+', 1)[0][:6]
        suffix = ('+' + rest.split('+', 1)[1]) if '+' in rest else ''
        text = f'{head}.{fraction}{suffix}'
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _docker_image_created(image: str) -> str | None:
    lines = _docker_output(['docker', 'image', 'inspect', '--format', '{{.Created}}', image])
    return lines[0] if lines else None


def cleanup_owned_images(images, *, created_after=None, image_created=None,
                         image_lister=None, container_lister=None, remover=None) -> dict:
    """Remove only images proven created by this run and unused by any container."""
    if created_after is None and isinstance(images, dict):
        # ``collect_owned_candidates()`` already carries image -> attempt start.
        created_after = dict(images)
    images = set(images)
    image_lister = image_lister or (lambda: _docker_output(
        ['docker', 'image', 'ls', '--format', '{{.Repository}}:{{.Tag}}']))
    container_lister = container_lister or (lambda: _docker_output(
        ['docker', 'ps', '-a', '--format', '{{.Image}}']))
    image_created = image_created or _docker_image_created
    remover = remover or (lambda names: subprocess.run(['docker', 'rmi', *names], check=False))

    available = set(image_lister())
    in_use = set(container_lister())
    owned, unproven = [], {}
    for image in sorted(images):
        if image not in available:
            continue
        if image in in_use:
            continue
        spec = created_after.get(image) if isinstance(created_after, dict) else created_after
        if isinstance(spec, dict):
            started_at = spec.get('attempt_start')
            baseline = spec.get('baseline')
        else:
            started_at = spec
            baseline = None
        created = None
        try:
            created = image_created(image)
        except Exception as exc:  # noqa: BLE001 - cannot prove ownership, keep the image
            unproven[image] = f'inspect failed: {exc.__class__.__name__}: {exc}'
            continue
        if isinstance(spec, dict) and baseline is not None and image in set(baseline):
            unproven[image] = 'present in the pre-run image inventory'
            continue
        if isinstance(spec, dict) and baseline is None:
            unproven[image] = 'no pre-run image inventory was recorded'
            continue
        if _created_since(created, started_at):
            owned.append(image)
        else:
            unproven[image] = f'created={created or "unknown"} attempt_start={started_at or "unknown"}'

    removed, failed = [], []
    if owned:
        try:
            result = remover(owned)
            returncode = getattr(result, 'returncode', 0)
            if returncode == 0:
                removed = list(owned)
            else:
                detail = (getattr(result, 'stderr', '') or '').strip() or f'rmi exit code {returncode}'
                failed = [{'images': list(owned), 'error': detail}]
        except Exception as exc:  # noqa: BLE001
            failed = [{'images': list(owned), 'error': f'{exc.__class__.__name__}: {exc}'}]

    removed_set = set(removed)
    return {
        'requested': sorted(images),
        'available': sorted(set(images) & available),
        'in_use': sorted(set(images) & in_use),
        'owned': owned,
        'removed': removed,
        'failed': failed,
        'unproven': unproven,
        'kept': sorted(set(images) - removed_set),
    }


def _created_since(created, started_at) -> bool:
    created_stamp = _parse_timestamp(created)
    start_stamp = _parse_timestamp(started_at)
    if created_stamp is None or start_stamp is None:
        return False
    return created_stamp >= start_stamp
