"""Configuration loading, run planning, identity hashing and preflight checks."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .util import deep_merge as _deep_merge, digest, file_digest, redact_text

SUITE_NAMES = ('knowledge', 'swe', 'vision')
PROFILE_NAMES = ('smoke', 'lite', 'full')


class ConfigError(RuntimeError):
    """Bad or inconsistent user configuration.  Raised before any paid work."""


@dataclass
class PlanEntry:
    model_alias: str
    model_cfg: dict
    suite: str
    dataset: str
    spec: dict
    profile: str
    limit: int | None
    batch_size: int
    pinned: bool
    local_path: Path | None = None
    requires: tuple = ()
    status: str = 'configured'
    pinned_instance: str | None = None
    output_subdir: Path | None = None

    @property
    def per_subset_limit(self) -> int | None:
        return self.limit


@dataclass
class Plan:
    data_root: Path
    profile: str
    entries: list[PlanEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def load_models(config_dir: Path) -> dict:
    try:
        models = (load_yaml(Path(config_dir) / 'models.yaml').get('models') or {})
    except FileNotFoundError as exc:
        raise ConfigError(f'models config not found: {exc}') from exc
    if not models:
        raise ConfigError('models.yaml contains no models')
    return models


def load_suites(config_dir: Path) -> dict:
    try:
        return load_yaml(Path(config_dir) / 'suites.yaml')
    except FileNotFoundError as exc:
        raise ConfigError(f'suites config not found: {exc}') from exc


def _suite_of(dataset: str, suites_cfg: dict):
    for suite_name, suite_cfg in suites_cfg.items():
        if dataset in (suite_cfg.get('datasets') or {}):
            return suite_name
    return None


def build_plan(*, models_cfg: dict, suites_doc: dict, data_root: Path, model_names, suite_names,
               profile: str, datasets=None, limit=None, batch_size=None,
               include_extensions=False) -> Plan:
    data_root = Path(data_root).resolve()
    suites_cfg = suites_doc.get('suites') or {}
    defaults = suites_doc.get('defaults') or {}
    profiles = suites_doc.get('profiles') or {}

    if profile not in PROFILE_NAMES:
        raise ConfigError(f'unknown profile {profile!r}; expected one of {PROFILE_NAMES}')
    if limit is not None and limit <= 0:
        raise ConfigError(f'--limit must be positive, got {limit}')
    if batch_size is not None and batch_size <= 0:
        raise ConfigError(f'--batch-size must be positive, got {batch_size}')

    # R7: duplicate model selections are de-duplicated, order preserved.
    ordered_models = []
    for name in (list(models_cfg) if list(model_names) == ['all'] else model_names):
        if name not in models_cfg:
            raise ConfigError(f'unknown model {name!r}; known: {", ".join(models_cfg)}')
        if name not in ordered_models:
            ordered_models.append(name)
    if not ordered_models:
        raise ConfigError('no model selected')

    selected_suites = list(SUITE_NAMES) if list(suite_names) == ['all'] else list(suite_names)
    for name in selected_suites:
        if name not in suites_cfg:
            raise ConfigError(f'unknown suite {name!r}')

    # R8: explicit --datasets selects exactly those datasets (overrides the core filter)
    # but a dataset must belong to one of the selected suites.
    explicit = list(datasets or [])
    for dataset in explicit:
        owner = _suite_of(dataset, suites_cfg)
        if owner is None:
            raise ConfigError(f'unknown dataset {dataset!r}')
        if owner not in selected_suites:
            raise ConfigError(
                f'dataset {dataset!r} belongs to suite {owner!r}, '
                f'not to the selected suite(s) {selected_suites}'
            )

    plan = Plan(data_root=data_root, profile=profile)
    for model_alias in ordered_models:
        model_cfg = models_cfg[model_alias]
        for suite_name in selected_suites:
            suite_cfg = suites_cfg[suite_name]
            for dataset, raw_spec in (suite_cfg.get('datasets') or {}).items():
                # suite-level execution policy (sandbox, agent limits) is inherited by
                # every dataset in the suite; the dataset spec may override it.
                spec = {key: suite_cfg[key] for key in ('sandbox', 'agent_config') if key in suite_cfg}
                spec.update(raw_spec or {})
                if explicit:
                    if dataset not in explicit:
                        continue
                else:
                    if not spec.get('core', False) and not include_extensions:
                        continue
                entry = _build_entry(
                    model_alias=model_alias, model_cfg=model_cfg, suite_cfg=suite_cfg,
                    suite_name=suite_name, dataset=dataset, spec=spec, defaults=defaults,
                    profiles=profiles, profile=profile, data_root=data_root,
                    cli_limit=limit, cli_batch=batch_size,
                )
                plan.entries.append(entry)

    if not plan.entries:
        raise ConfigError('dataset selection is empty; use --include-extensions or name datasets explicitly')
    return plan


def _build_entry(*, model_alias, model_cfg, suite_cfg, suite_name, dataset, spec, defaults,
                 profiles, profile, data_root, cli_limit, cli_batch) -> PlanEntry:
    if cli_limit is not None:
        resolved_limit = cli_limit
    elif spec.get('pinned'):
        resolved_limit = 1
    elif profile == 'full':
        resolved_limit = spec.get('full_per_subset_limit')
    else:
        resolved_limit = (profiles.get(profile) or {}).get('limit')

    batch = None
    for candidate in (cli_batch, spec.get('eval_batch_size'), suite_cfg.get('eval_batch_size'),
                      defaults.get('eval_batch_size')):
        if candidate is not None:
            batch = int(candidate)
            break
    if batch is None:
        batch = 4
    if batch <= 0:
        raise ConfigError(f'{dataset}: eval_batch_size must be positive, got {batch}')

    local_path = None
    if spec.get('local_path'):
        local_path = Path(str(spec['local_path']).replace('${DATA_ROOT}', str(data_root))).resolve()
    pinned = bool(spec.get('pinned'))

    return PlanEntry(
        model_alias=model_alias, model_cfg=model_cfg, suite=suite_name, dataset=dataset,
        spec=spec, profile=profile, limit=resolved_limit, batch_size=batch, pinned=pinned,
        local_path=local_path, requires=tuple(spec.get('requires') or ()),
        status=str(spec.get('status') or 'configured'),
        pinned_instance=spec.get('pinned_instance_id'),
    )


# ---------------------------------------------------------------------------
# identity hashes -- model axis and protocol axis are deliberately separate
# ---------------------------------------------------------------------------

def endpoint_identity(api_url: str) -> str:
    """Endpoint identifier without credentials or query string."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(redact_text(api_url or ''))
    netloc = parts.hostname or ''
    if parts.port:
        netloc = f'{netloc}:{parts.port}'
    return urlunsplit((parts.scheme or 'https', netloc, parts.path, '', '')).rstrip('/')


def model_config_identity(model_alias: str, model_cfg: dict) -> str:
    return digest({
        'alias': model_alias,
        'model_id': model_cfg.get('model_id', ''),
        'endpoint': endpoint_identity(model_cfg.get('api_url', '')),
        'deployment_version': model_cfg.get('deployment_version', 'unknown'),
    })


PROTOCOL_EXCLUDED_KEYS = {
    'model', 'model_id', 'api_url', 'api_key', 'work_dir', 'no_timestamp', 'use_cache',
    'collect_perf', 'ignore_errors', 'limit', 'eval_batch_size', 'output_dir',
    # pure storage locations; dataset content identity is covered by the manifest
    'dataset_dir',
}


def protocol_identity(resolved_task_config: dict) -> str:
    """Hash of the effective evaluation protocol (not the model identity, not sample count)."""
    payload = {}
    for key, value in resolved_task_config.items():
        if key in PROTOCOL_EXCLUDED_KEYS:
            continue
        payload[key] = value
    dataset_args = payload.get('dataset_args')
    if isinstance(dataset_args, dict):
        # local_path describes where data lives, not what protocol was run; the sample
        # manifest carries content evidence instead.
        payload['dataset_args'] = {
            name: {k: v for k, v in (args or {}).items() if k != 'local_path'}
            for name, args in dataset_args.items()
        }
    return digest(payload)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def docker_available() -> bool:
    if shutil.which('docker') is None:
        return False
    try:
        result = subprocess.run(
            ['docker', 'info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def validate_pinned_dir(path: Path, expected_instance: str | None = None):
    """Return (ok, reason) for a prepared pinned dataset directory."""
    source_file = Path(path) / 'source.json'
    parquet = Path(path) / 'test-00000-of-00001.parquet'
    if not Path(path).is_dir():
        return False, f'pinned dataset missing: {path}'
    if not source_file.exists():
        return False, f'pinned dataset has no source.json: {path}'
    if not parquet.exists() or parquet.stat().st_size == 0:
        return False, f'pinned dataset has no non-empty test parquet: {path}'
    try:
        source = json.loads(source_file.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        return False, f'pinned source.json is corrupt: {exc}'
    instance_ids = source.get('instance_ids') or []
    if len(instance_ids) != 1:
        return False, f'pinned dataset must contain exactly one instance, got {instance_ids}'
    if expected_instance and instance_ids != [expected_instance]:
        return False, f'pinned dataset is {instance_ids}, expected [{expected_instance}]'
    recorded_hash = source.get('parquet_sha256')
    if recorded_hash:
        actual_hash = file_digest(parquet)
        if actual_hash != recorded_hash:
            return False, (
                f'pinned parquet checksum mismatch: recorded {recorded_hash[:12]}, '
                f'actual {actual_hash[:12]}'
            )
    return True, source.get('revision', 'unknown')


def validate_pinned_content(path: Path, expected_instance: str | None = None):
    """Verify the pinned parquet is readable and actually contains the instance."""
    parquet = Path(path) / 'test-00000-of-00001.parquet'
    if not parquet.exists():
        return False, f'pinned dataset has no test parquet: {path}'
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001 - pyarrow is an extras dependency
        return False, f'pyarrow is required to verify pinned parquet ({exc.__class__.__name__})'
    try:
        table = pq.read_table(parquet)
    except Exception as exc:  # noqa: BLE001
        return False, f'pinned parquet is not readable: {exc.__class__.__name__}: {exc}'
    if table.num_rows != 1:
        return False, f'pinned dataset must contain exactly one sample, found {table.num_rows}'
    if 'instance_id' not in table.column_names:
        return False, 'pinned parquet has no instance_id column'
    values = table.column('instance_id').to_pylist()
    if not expected_instance:
        return False, 'pinned content check requires an expected instance id'
    if values != [expected_instance]:
        return False, f'pinned instance_id is {values}, expected [{expected_instance}]'
    return True, 'content verified'


def check_entry_requirements(entry: PlanEntry, *, check_keys: bool, check_docker: bool,
                             check_pinned: bool) -> list[str]:
    errors = []
    api_key_env = entry.model_cfg.get('api_key_env')
    if check_keys and api_key_env and not os.environ.get(api_key_env):
        errors.append(
            f'{entry.model_alias}: environment variable {api_key_env!r} is not set'
        )
    for module in entry.requires:
        if not module_available(module):
            errors.append(
                f'{entry.dataset}: required module {module!r} is not installed '
                f'(status={entry.status})'
            )
    needs_sandbox = bool(entry.spec.get('requires_sandbox'))
    if needs_sandbox:
        sandbox = entry.spec.get('sandbox')
        if not sandbox or not sandbox.get('enabled'):
            errors.append(
                f'{entry.dataset}: code-execution benchmark requires sandbox.enabled=true; '
                'refusing to fall back to host execution'
            )
        if check_docker and not docker_available():
            errors.append(f'{entry.dataset}: docker is required for the sandbox but is unavailable')
    if check_pinned and entry.pinned and entry.local_path is not None:
        if not entry.pinned_instance:
            errors.append(
                f'{entry.dataset}: pinned dataset has no pinned_instance_id in the config'
            )
        else:
            ok, reason = validate_pinned_dir(entry.local_path, entry.pinned_instance)
            if not ok:
                errors.append(f'{entry.dataset}: {reason}; run `bench.py prepare` first')
            else:
                content_ok, content_reason = validate_pinned_content(
                    entry.local_path, entry.pinned_instance)
                if not content_ok:
                    errors.append(f'{entry.dataset}: {content_reason}; run `bench.py prepare` first')
    return errors


def preflight(plan: Plan, *, check_keys: bool = True, check_docker: bool = True,
              check_pinned: bool = True) -> list[str]:
    errors = []
    for entry in plan.entries:
        errors.extend(check_entry_requirements(
            entry, check_keys=check_keys, check_docker=check_docker, check_pinned=check_pinned
        ))
    return errors
