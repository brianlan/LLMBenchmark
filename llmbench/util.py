"""Shared helpers: canonical hashing, secret redaction, atomic writes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import __version__ as TOOL_VERSION  # re-exported for convenience

SECRET_KEY_NAMES = {
    'token', 'access_token', 'refresh_token', 'id_token',
    'auth', 'authorization', 'auth_header',
    'secret', 'client_secret', 'api_secret', 'private_key',
    'password', 'passwd',
    'api_key', 'apikey', 'credential', 'credentials',
}
SECRET_KEY_SUFFIXES = ('_api_key', '_secret', '_password', '_token')

_URL_CREDENTIALS = re.compile(r'(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@')


def is_secret_key(key: str) -> bool:
    """Exact key-name match, so counters like ``input_tokens``/``max_tokens`` are kept."""
    normalized = str(key).strip().lower().replace('-', '_').replace(' ', '_')
    return normalized in SECRET_KEY_NAMES or normalized.endswith(SECRET_KEY_SUFFIXES)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def to_jsonable(value):
    """Convert arbitrary values into a canonical, JSON-serializable form."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'non-finite float cannot be canonicalized: {value!r}')
        return value
    if isinstance(value, bytes):
        return {'__bytes_sha256__': hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    dump = getattr(value, 'model_dump', None)
    if callable(dump):
        return to_jsonable(dump(mode='json'))
    if hasattr(value, '__dict__') and not callable(value):
        return to_jsonable({k: v for k, v in vars(value).items() if not k.startswith('_')})
    return str(value)


def canonical_json(value) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def short_digest(value, length: int = 12) -> str:
    return digest(value)[:length]


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_secret_key(key: str) -> bool:
    """Exact key-name match, so counters like ``input_tokens``/``max_tokens`` are kept."""
    normalized = str(key).strip().lower().replace('-', '_').replace(' ', '_')
    return normalized in SECRET_KEY_NAMES or normalized.endswith(SECRET_KEY_SUFFIXES)


def redact(value, secrets=()):
    """Recursively replace secret-looking keys and known secret values."""
    secrets = [s for s in secrets if s]
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            redacted[key] = '***' if is_secret_key(key) else redact(item, secrets)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, Path):
        return redact_text(str(value), secrets)
    return value


def redact_text(text: str, secrets=()) -> str:
    text = _URL_CREDENTIALS.sub(r'\g<scheme>***@', text or '')
    for secret in secrets:
        if secret:
            text = text.replace(secret, '***')
    return text


def collect_secrets(*configs) -> list:
    """Collect literal secret values from environment variables referenced by configs."""
    secrets = []
    for config in configs:
        if not isinstance(config, dict):
            continue
        for key in ('api_key', 'token', 'secret', 'password'):
            value = config.get(key)
            if isinstance(value, str) and value and value != 'EMPTY':
                secrets.append(value)
        env_name = config.get('api_key_env')
        if env_name and os.environ.get(env_name):
            secrets.append(os.environ[env_name])
    return secrets


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path: Path, on_error=None):
    """Yield parsed JSONL objects; report malformed lines through ``on_error``."""
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if on_error is not None:
                    on_error(line_number, exc)


def safe_slug(name: str, max_length: int = 80) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '-', str(name)).strip('-')
    return slug[:max_length] or 'unnamed'
