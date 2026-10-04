"""Run orchestration: attempt lifecycle, error boundaries, persistence and re-import."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import __version__ as TOOL_VERSION
from .compat import ocr_compat_patch, report_patch_status
from .config import ConfigError, model_config_identity, protocol_identity
from .evidence import (
    MetricParseError,
    ReportError,
    apply_coverage,
    build_sample_manifest,
    capture_loaded_dataset,
    collect_output_evidence,
    locate_report,
    metric_rows,
    summarize_diagnostics,
    write_run_manifest,
)
from .images import SWE_IMAGE_PREFIXES
from .store import DuplicateAttemptError, Store
from .util import (
    atomic_write_json,
    collect_secrets,
    deep_merge,
    digest,
    now_iso,
    read_json,
    redact,
    redact_text,
    safe_slug,
)
from .validation import (
    COMPARABILITY_UNKNOWN,
    VALID,
    assess_comparability,
    assess_run,
    has_formal_eligibility,
)


class BatchInterrupted(RuntimeError):
    pass


@dataclass
class AttemptResult:
    run_id: str
    output_dir: Path
    execution_status: str
    validity_status: str
    status_reason: str
    persisted: bool
    error: str | None = None

    @property
    def acceptable(self) -> bool:
        return self.execution_status == 'completed' and self.validity_status == VALID


def batch_exit_code(results, *, allow_partial: bool = False) -> int:
    """Exit code 0 only when every selected attempt met the agreed success contract."""
    accepted = {'complete', 'partial'} if allow_partial else {'complete'}
    if not results:
        return 1
    clean = all(
        result.persisted and result.execution_status == 'completed'
        and result.validity_status in accepted
        for result in results
    )
    return 0 if clean else 1


def new_attempt_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo_dir), 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ''


def evalscope_version() -> str:
    try:
        import evalscope
        return getattr(evalscope, '__version__', 'unknown')
    except Exception:
        return 'unknown'


def resolve_task_config(raw_cfg: dict) -> dict:
    """Resolve EvalScope defaults once; used for protocol identity and the manifest."""
    from evalscope.config import TaskConfig
    return TaskConfig.from_dict(raw_cfg).model_dump(mode='json')


def build_raw_task_config(entry, *, data_root: Path, output_dir: Path, api_key: str) -> dict:
    model_cfg = entry.model_cfg
    generation_config = deep_merge(
        model_cfg.get('generation_config') or {},
        entry.spec.get('generation_config') or {},
    )
    task_cfg = {
        'model': model_cfg['model_id'],
        'model_id': model_cfg.get('report_id') or model_cfg['model_id'],
        'api_url': model_cfg['api_url'],
        'api_key': api_key,
        'eval_type': model_cfg.get('eval_type', 'openai_api'),
        'datasets': [entry.dataset],
        'dataset_dir': str(data_root / 'datasets'),
        'generation_config': generation_config,
        'limit': entry.limit,
        'eval_batch_size': entry.batch_size,
        'work_dir': str(output_dir),
        'no_timestamp': True,
        'seed': 42,
        'collect_perf': True,
        'ignore_errors': True,
    }
    dataset_args = dict(entry.spec.get('dataset_args') or {})
    if entry.local_path is not None:
        dataset_args['local_path'] = str(entry.local_path)
    if entry.spec.get('extra_params'):
        dataset_args['extra_params'] = deep_merge(
            dataset_args.get('extra_params') or {}, entry.spec['extra_params']
        )
    if dataset_args:
        task_cfg['dataset_args'] = {entry.dataset: dataset_args}
    if entry.spec.get('agent_config'):
        task_cfg['agent_config'] = dict(entry.spec['agent_config'])
    sandbox = entry.spec.get('sandbox') or {}
    if sandbox:
        task_cfg['sandbox'] = dict(sandbox)
    return task_cfg


def build_run_manifest(entry, *, attempt_id: str, run_group: str, output_dir: Path,
                       raw_cfg: dict, resolved_cfg: dict, repo_dir: Path,
                       data_root: Path) -> dict:
    secrets = collect_secrets(raw_cfg)
    return {
        'tool': 'llmbench',
        'tool_version': TOOL_VERSION,
        'created_at': now_iso(),
        'attempt_id': attempt_id,
        'run_group': run_group,
        'suite': entry.suite,
        'profile': entry.profile,
        'dataset': entry.dataset,
        'model_alias': entry.model_alias,
        'model_id': entry.model_cfg['model_id'],
        'report_id': raw_cfg.get('model_id'),
        'api_url': redact(entry.model_cfg.get('api_url', ''), secrets),
        'model_config_identity': model_config_identity(entry.model_alias, entry.model_cfg),
        'protocol_identity': protocol_identity(resolved_cfg),
        'dataset_source': dataset_source(entry, data_root),
        'limits': {
            'profile': entry.profile,
            'per_subset': entry.limit,
            'cli_override': None,
        },
        'generation_config': redact(resolved_cfg.get('generation_config') or {}, secrets),
        'agent_config': redact(resolved_cfg.get('agent_config') or {}, secrets),
        'sandbox': redact(resolved_cfg.get('sandbox') or {}, secrets),
        'task_config': redact(raw_cfg, secrets),
        'dataset_status': entry.status,
        'git_commit': git_commit(repo_dir),
        'evalscope_version': evalscope_version(),
        'output_dir': str(output_dir),
        'sample_manifest': [],
        'sample_manifest_detail': {},
        'sample_manifest_identity': None,
    }


def dataset_source(entry, data_root: Path) -> dict:
    source = {
        'dataset_id': entry.spec.get('dataset_args', {}).get('inference_dataset_id'),
        'local_path': str(entry.local_path) if entry.local_path else None,
        'pinned': entry.pinned,
        'revision': 'unknown',
        'revision_source': 'unknown',
    }
    if entry.local_path is not None:
        source_file = entry.local_path / 'source.json'
        if source_file.exists():
            try:
                pinned_source = read_json(source_file)
            except Exception:
                pinned_source = {}
            source['revision'] = pinned_source.get('revision', 'unknown')
            source['revision_source'] = pinned_source.get('revision_source', 'pinned source.json')
            source['pinned_source'] = pinned_source
    if not source['dataset_id']:
        try:
            from evalscope.api.registry import BENCHMARK_REGISTRY
            meta = BENCHMARK_REGISTRY.get(entry.dataset)
            source['dataset_id'] = getattr(meta, 'dataset_id', None)
        except Exception:
            source['dataset_id'] = None
    return source


def _attempt_record(entry, *, attempt_id, run_group, output_dir, raw_cfg, manifest, repo_dir) -> dict:
    secrets = collect_secrets(raw_cfg)
    return {
        'run_id': attempt_id,
        'run_group': run_group,
        'model_alias': entry.model_alias,
        'model_id': entry.model_cfg['model_id'],
        'api_url': redact(entry.model_cfg.get('api_url', ''), secrets),
        'model_config_identity': manifest['model_config_identity'],
        'suite': entry.suite,
        'profile': entry.profile,
        'dataset': entry.dataset,
        'execution_status': 'running',
        'validity_status': None,
        'status_reason': None,
        'phase': 'preflight',
        'comparability': COMPARABILITY_UNKNOWN,
        'started_at': now_iso(),
        'evalscope_version': evalscope_version(),
        'tool_version': TOOL_VERSION,
        'git_commit': git_commit(repo_dir),
        'output_dir': str(output_dir),
        'config_json': json.dumps(redact(raw_cfg, secrets), ensure_ascii=False, sort_keys=True),
        'protocol_identity': manifest['protocol_identity'],
        'sample_manifest_identity': None,
    }


def _manifest_rows_from_details(detail: dict) -> list[dict]:
    rows = []
    for subset, samples in (detail or {}).items():
        rows.append({
            'subset': subset,
            'selected': len(samples),
            'sample_ids': [item['id'] for item in samples],
            'question_digest': digest([item['question_digest'] for item in samples]),
            'media_digest': digest([item['media_digest'] for item in samples]),
            'input_digest': digest([item['input_digest'] for item in samples]),
            'media_evidence': all(item.get('media_evidence', True) for item in samples),
        })
    return rows


def _plain_failure(attempt_id: str, output_dir: Path, message: str, secrets) -> AttemptResult:
    reason = redact_text(message, secrets)
    return AttemptResult(attempt_id, output_dir, 'failed', 'invalid', reason, False, reason)


def _swe_image_snapshot():
    """Pre-run SWE image inventory; None means ownership cannot be proven."""
    try:
        output = subprocess.check_output(
            ['docker', 'image', 'ls', '--format', '{{.Repository}}:{{.Tag}}'],
            text=True, stderr=subprocess.DEVNULL, timeout=60,
        )
        return [line for line in output.splitlines() if line.startswith(SWE_IMAGE_PREFIXES)]
    except Exception:
        return None


def execute_attempt(entry, *, store: Store, attempt_id: str, run_group: str, output_dir: Path,
                    raw_cfg: dict, resolved_cfg: dict, repo_dir: Path, data_root: Path) -> AttemptResult:
    """Run one attempt inside a complete error boundary.

    Every exit path terminates the attempt in the database: task errors, report
    errors, evidence I/O errors and unexpected exceptions all become an explicit
    failed/invalid result instead of a bare exception or a permanent `running`.
    """
    secrets = collect_secrets(raw_cfg)
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return _plain_failure(attempt_id, output_dir,
                              f'evidence_setup_error: {exc.__class__.__name__}: {exc}', secrets)

    manifest = build_run_manifest(
        entry, attempt_id=attempt_id, run_group=run_group, output_dir=output_dir,
        raw_cfg=raw_cfg, resolved_cfg=resolved_cfg, repo_dir=repo_dir, data_root=data_root,
    )
    if entry.spec.get('requires_sandbox'):
        # Snapshot before the run so an image can be proven to be new later.
        manifest['baseline_swe_images'] = _swe_image_snapshot()
    try:
        store.start_attempt(_attempt_record(entry, attempt_id=attempt_id, run_group=run_group,
                                            output_dir=output_dir, raw_cfg=raw_cfg,
                                            manifest=manifest, repo_dir=repo_dir))
    except DuplicateAttemptError:
        return AttemptResult(attempt_id, output_dir, 'failed', 'invalid',
                             'duplicate attempt id', persisted=False)
    except Exception as exc:  # noqa: BLE001
        return _plain_failure(attempt_id, output_dir,
                              f'database_unavailable: {exc.__class__.__name__}: {exc}', secrets)

    phase = 'inference'
    interrupted = False
    unexpected_error = None
    compat_status = {'applied': False, 'reason': 'not attempted', 'target': 'nltk.edit_distance'}
    report = None
    report_error = None
    metrics = []
    output_evidence = {'subsets': {}, 'malformed_lines': []}

    def on_samples(raw_dataset, processed_dataset):
        rows, details = build_sample_manifest(raw_dataset, processed_dataset)
        manifest['sample_manifest'] = rows
        manifest['sample_manifest_detail'] = details
        manifest['sample_manifest_identity'] = digest(rows) if rows else None
        write_run_manifest(output_dir, redact(manifest, secrets))

    try:
        write_run_manifest(output_dir, redact(manifest, secrets))
        with ocr_compat_patch() as status:
            compat_status = status
            with capture_loaded_dataset(on_samples):
                from evalscope import run_task
                run_task(raw_cfg)
        phase = 'report'
        try:
            report_path, report = locate_report(
                output_dir, entry.dataset, raw_cfg.get('model_id'),
                expected_pretty_name=_pretty_name(entry.dataset),
            )
            manifest['report_path'] = str(report_path)
            metrics = metric_rows(report)
        except ReportError as exc:
            report_error = exc
        except MetricParseError as exc:
            report_error = ReportError('metric_parse_error', str(exc))
        except Exception as exc:  # noqa: BLE001 - keep the batch alive, record phase
            report_error = ReportError('parse_error', f'{exc.__class__.__name__}: {exc}')
        phase = 'diagnostics'
        output_evidence = collect_output_evidence(
            output_dir, entry.dataset, report_id=raw_cfg.get('model_id')
        )
    except KeyboardInterrupt:
        interrupted = True
    except Exception:
        unexpected_error = traceback.format_exc()

    manifest_rows = manifest.get('sample_manifest') or []
    coverage_rows = []
    comparability, comparability_reasons = COMPARABILITY_UNKNOWN, ['assessment did not run']
    diagnostics = {}
    assessment_phase = 'assessment'
    try:
        coverage_rows = apply_coverage(manifest_rows, output_evidence)
        if interrupted:
            outcome = assess_run(interrupted=True)
        elif unexpected_error:
            outcome = assess_run(task_error=f'[{phase}] {unexpected_error}')
        else:
            outcome = assess_run(
                report_error=report_error, metrics=metrics,
                execution_summary=(report or {}).get('execution_summary'),
                manifest_rows=coverage_rows,
            )
        assessment_phase = 'comparability'
        comparability, comparability_reasons = assess_comparability(coverage_rows, output_evidence)
        if comparability_reasons and outcome['validity_status'] == VALID and comparability != 'verified':
            outcome['status_reason'] = (outcome['status_reason'] + '; ' if outcome['status_reason'] else '') \
                + 'comparability: ' + '; '.join(comparability_reasons)
        assessment_phase = 'diagnostics'
        diagnostics = summarize_diagnostics(output_evidence)
        diagnostics['ocr_compat'] = compat_status
        diagnostics['comparability_reasons'] = comparability_reasons
        diagnostics['raw_output_dir'] = str(output_dir)
        diagnostics['repeat_policy'] = 'review keys are matched against prediction keys'
    except KeyboardInterrupt:
        # User interrupt during assessment must terminate this attempt too.
        interrupted = True
        outcome = assess_run(interrupted=True)
        diagnostics = {
            'attempt_error_phase': assessment_phase,
            'interrupted_during': assessment_phase,
            'ocr_compat': compat_status,
            'comparability_reasons': [f'interrupted during {assessment_phase}'],
            'raw_output_dir': str(output_dir),
        }
        comparability, comparability_reasons = COMPARABILITY_UNKNOWN, [
            f'interrupted during {assessment_phase}']
    except Exception:
        # B4: validation/diagnostics are part of the attempt lifecycle too.
        unexpected_error = traceback.format_exc()
        outcome = assess_run(task_error=f'[{assessment_phase}] {unexpected_error}')
        diagnostics = {
            'attempt_error_phase': assessment_phase,
            'assessment_error': redact_text(unexpected_error, secrets),
            'ocr_compat': compat_status,
            'comparability_reasons': [f'assessment failed in phase {assessment_phase}'],
            'raw_output_dir': str(output_dir),
        }
        comparability, comparability_reasons = COMPARABILITY_UNKNOWN, [
            f'assessment failed in phase {assessment_phase}']
    if unexpected_error:
        diagnostics.setdefault('attempt_error_phase', phase)
        if phase == 'diagnostics':
            diagnostics['diagnostics_error'] = redact_text(unexpected_error, secrets)

    if report is not None:
        diagnostics['primary_metric_identity'] = report.get('primary_metric_identity')
        execution = report.get('execution_summary') or {}
    else:
        execution = {}

    outcome.update({
        'phase': 'persist',
        'finished_at': now_iso(),
        'comparability': comparability,
        'num_requested': execution.get('requested'),
        'num_succeeded': execution.get('succeeded'),
        'num_errored': execution.get('errored'),
        'incomplete': bool(execution.get('incomplete')),
        'report_path': manifest.get('report_path'),
        'metrics': metrics,
        'sample_manifest': coverage_rows,
        'diagnostics': diagnostics,
        'perf_metrics': (report or {}).get('perf_metrics'),
        'primary_metric_identity': (report or {}).get('primary_metric_identity'),
        'protocol_identity': manifest.get('protocol_identity'),
        'sample_manifest_identity': manifest.get('sample_manifest_identity'),
        'error': redact_text(unexpected_error or '', secrets) or None,
    })

    # B1: every external surface (DB, outcome file, terminal, later summary)
    # consumes the same fully redacted result object.
    report_patch_status(compat_status)
    safe_outcome = redact(outcome, secrets)
    outcome_file = output_dir / 'run_outcome.json'
    try:
        atomic_write_json(outcome_file, safe_outcome)
    except Exception as exc:  # noqa: BLE001 - file evidence failed, DB must still terminate
        evidence_error = f'{exc.__class__.__name__}: {exc}'
        safe_outcome = redact({
            **safe_outcome,
            'execution_status': 'failed',
            'validity_status': 'invalid',
            'phase': 'evidence',
            'status_reason': f'evidence_write_error: {evidence_error}',
            'error': safe_outcome.get('error') or evidence_error,
        }, secrets)
        try:
            atomic_write_json(outcome_file, safe_outcome)
        except Exception:  # noqa: BLE001 - nothing more we can write
            pass

    persisted = True
    persist_error = None
    try:
        store.finish_attempt(attempt_id, safe_outcome)
    except Exception as exc:  # noqa: BLE001
        persisted = False
        persist_error = redact_text(f'{exc.__class__.__name__}: {exc}', secrets)
        print(f'!!! failed to persist {attempt_id}: {persist_error}\n'
              f'    evidence kept at {output_dir}; re-import with: bench.py import --output-dir {output_dir}',
              file=sys.stderr)

    if interrupted:
        raise BatchInterrupted()
    result = AttemptResult(
        run_id=attempt_id, output_dir=output_dir,
        execution_status=safe_outcome['execution_status'],
        validity_status=safe_outcome['validity_status'],
        status_reason=safe_outcome['status_reason'], persisted=persisted,
        error=safe_outcome.get('error') or persist_error,
    )
    print(f"--- {entry.dataset} [{attempt_id}] {result.execution_status}/{result.validity_status}"
          f" ({result.status_reason})")
    if persisted and safe_outcome.get('metrics'):
        rows = safe_outcome['metrics']
        top = [row for row in rows if row['is_primary']] or [row for row in rows if not row['category']]
        for row in top[:3]:
            score = 'N/A' if row['score'] is None else f"{row['score']:.4f}"
            print(f"    {row['metric_name']} = {score} (n={row['num']})")
    return result


def _pretty_name(dataset: str):
    try:
        from evalscope.api.registry import BENCHMARK_REGISTRY
        meta = BENCHMARK_REGISTRY.get(dataset)
        return getattr(meta, 'pretty_name', None)
    except Exception:
        return None


def _identity_conflicts(manifest: dict, existing) -> list:
    """Compare immutable run identity between the manifest and the DB row.

    Moving an output directory is fine; rebinding a stored result to another
    dataset/model/protocol/manifest is not.
    """
    fields = ('dataset', 'suite', 'model_id', 'model_config_identity',
              'protocol_identity', 'sample_manifest_identity')
    conflicts = []
    existing_keys = set(existing.keys()) if hasattr(existing, 'keys') else set()
    for field in fields:
        new_value = manifest.get(field)
        old_value = existing[field] if field in existing_keys else None
        if new_value and old_value and str(new_value) != str(old_value):
            conflicts.append(f'{field}: db={old_value} manifest={new_value}')
    return conflicts


def _write_audit(output_dir: Path, name: str, payload: dict, secrets) -> None:
    try:
        atomic_write_json(output_dir / name, redact(payload, secrets))
    except Exception:  # noqa: BLE001 - audit is best effort
        pass


def import_output(output_dir: Path, store: Store, *, repo_dir: Path, data_root: Path) -> dict:
    """Re-parse an existing output directory without calling any model.

    Parse and assessment run behind this boundary: failures are returned as a
    redacted structured result (using secrets collected from the manifest)
    instead of escaping as raw exception text.  An import that would rebind
    results to another run identity, or downgrade an accepted run, is rejected
    and audited; it never rewrites the original evaluation completion time.
    """
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / 'run_manifest.json') if (output_dir / 'run_manifest.json').exists() else None
    if manifest is None:
        raise ConfigError(f'no run_manifest.json under {output_dir}')
    entry = _ManifestEntry(manifest)
    run_id = manifest['attempt_id']
    secrets = collect_secrets(manifest.get('task_config') or {})

    existing = None
    existing_metrics = None
    try:
        report_path, report = locate_report(
            output_dir, manifest['dataset'], manifest['report_id'], expected_pretty_name=None
        )
        metrics = metric_rows(report)
        output_evidence = collect_output_evidence(
            output_dir, manifest['dataset'], report_id=manifest.get('report_id')
        )
        raw_rows = manifest.get('sample_manifest') or _manifest_rows_from_details(
            manifest.get('sample_manifest_detail') or {}
        )
        coverage_rows = apply_coverage(raw_rows, output_evidence)
        manifest_identity = manifest.get('sample_manifest_identity') or (digest(raw_rows) if raw_rows else None)
        outcome = assess_run(
            report_error=None, metrics=metrics,
            execution_summary=report.get('execution_summary'), manifest_rows=coverage_rows,
        )
        comparability, comparability_reasons = assess_comparability(coverage_rows, output_evidence)
        diagnostics = summarize_diagnostics(output_evidence)
        diagnostics['comparability_reasons'] = comparability_reasons
        diagnostics['primary_metric_identity'] = report.get('primary_metric_identity')
        diagnostics['imported'] = True

        existing = store.get_attempt(run_id)
        if existing is not None:
            existing_metrics = store.conn.execute(
                'SELECT COUNT(*) AS n FROM metrics WHERE run_id=?', (run_id,)
            ).fetchone()['n']
        # Re-import must not pretend an old evaluation just finished.
        finished_at = None
        if existing is not None and existing['finished_at']:
            finished_at = existing['finished_at']
        elif manifest.get('created_at'):
            finished_at = manifest['created_at']

        outcome.update({
            'phase': 'imported', 'finished_at': finished_at, 'imported_at': now_iso(),
            'comparability': comparability,
            'num_requested': (report.get('execution_summary') or {}).get('requested'),
            'num_succeeded': (report.get('execution_summary') or {}).get('succeeded'),
            'num_errored': (report.get('execution_summary') or {}).get('errored'),
            'incomplete': bool((report.get('execution_summary') or {}).get('incomplete')),
            'report_path': str(report_path), 'metrics': metrics, 'sample_manifest': coverage_rows,
            'diagnostics': diagnostics, 'perf_metrics': report.get('perf_metrics'),
            'primary_metric_identity': report.get('primary_metric_identity'),
            'protocol_identity': manifest.get('protocol_identity'),
            'sample_manifest_identity': manifest_identity,
        })
        safe_outcome = redact(outcome, secrets)
    except Exception as exc:  # noqa: BLE001 - parse failures are redacted at this boundary
        return {
            'run_id': run_id, 'status': 'failed', 'committed': False,
            'evidence_file_updated': False,
            'reason': redact_text(f'{exc.__class__.__name__}: {exc}', secrets),
        }

    conflicts = _identity_conflicts(manifest, existing) if existing is not None else []
    if conflicts:
        audit = {
            'run_id': run_id, 'kind': 'identity_conflict', 'rejected_at': now_iso(),
            'conflicts': conflicts,
            'kept_validity_status': existing['validity_status'],
            'kept_comparability': existing['comparability'],
            'kept_metrics': existing_metrics,
            'report_path': str(report_path),
        }
        _write_audit(output_dir, 'run_import_rejected.json', audit, secrets)
        return {
            'run_id': run_id, 'status': 'rejected', 'committed': False,
            'reason': 'identity_conflict: ' + '; '.join(conflicts),
            'conflicts': conflicts,
            'validity_status': existing['validity_status'],
            'comparability': existing['comparability'],
            'metrics_kept': existing_metrics,
        }

    accepted_existing = {'complete', 'partial'}
    existing_eligible = existing is not None and has_formal_eligibility(
        existing['validity_status'], existing['comparability'])
    new_eligible = has_formal_eligibility(
        safe_outcome['validity_status'], safe_outcome['comparability'])
    downgraded = existing is not None and (
        (existing['validity_status'] in accepted_existing
         and safe_outcome['validity_status'] != 'complete')
        or (existing_eligible and not new_eligible)
    )
    if downgraded:
        audit = {
            'run_id': run_id,
            'rejected_at': now_iso(),
            'reason': safe_outcome['status_reason'],
            'new_validity_status': safe_outcome['validity_status'],
            'new_comparability': safe_outcome['comparability'],
            'kept_validity_status': existing['validity_status'],
            'kept_comparability': existing['comparability'],
            'kept_metrics': existing_metrics,
            'report_path': str(report_path),
        }
        _write_audit(output_dir, 'run_import_rejected.json', audit, secrets)
        return {
            'run_id': run_id, 'status': 'rejected', 'committed': False,
            'validity_status': existing['validity_status'],
            'comparability': existing['comparability'],
            'new_validity_status': safe_outcome['validity_status'],
            'new_comparability': safe_outcome['comparability'],
            'reason': safe_outcome['status_reason'],
            'metrics_kept': existing_metrics,
        }

    # Write the evidence file before committing: if it fails, the database still
    # holds the previous state and the CLI can say so truthfully.
    try:
        atomic_write_json(output_dir / 'run_outcome.json', safe_outcome)
    except Exception as exc:  # noqa: BLE001
        return {
            'run_id': run_id, 'status': 'failed', 'committed': False,
            'evidence_file_updated': False,
            'reason': redact_text(
                f'evidence_write_error: {exc.__class__.__name__}: {exc}', secrets),
        }
    try:
        if existing is None:
            # One transaction: a first import can never leave a `running` row.
            store.create_attempt_with_outcome(
                _manifest_attempt_record(entry, manifest, output_dir, repo_dir), safe_outcome
            )
        else:
            store.finish_attempt(run_id, safe_outcome)
    except Exception as exc:  # noqa: BLE001
        return {
            'run_id': run_id, 'status': 'failed', 'committed': False,
            'evidence_file_updated': True,
            'reason': redact_text(
                f'database_error: {exc.__class__.__name__}: {exc}', secrets),
        }
    return {
        'run_id': run_id, 'status': 'committed', 'committed': True,
        'validity_status': safe_outcome['validity_status'],
        'metrics': len(metrics), 'comparability': comparability,
    }


class _ManifestEntry:
    """Minimal entry view for re-import and image ownership checks."""

    def __init__(self, manifest: dict):
        self.model_alias = manifest.get('model_alias')
        self.model_cfg = {
            'model_id': manifest.get('model_id'),
            'api_url': manifest.get('api_url'),
            'report_id': manifest.get('report_id'),
        }
        self.suite = manifest.get('suite')
        self.dataset = manifest.get('dataset')
        self.profile = manifest.get('profile')
        self.spec = {}
        self.limit = (manifest.get('limits') or {}).get('per_subset')
        self.batch_size = 1
        self.pinned = False
        self.local_path = None
        self.requires = ()
        self.status = 'imported'


def _manifest_attempt_record(entry, manifest: dict, output_dir: Path, repo_dir: Path) -> dict:
    return {
        'run_id': manifest['attempt_id'], 'run_group': manifest.get('run_group'),
        'model_alias': entry.model_alias, 'model_id': entry.model_cfg['model_id'],
        'api_url': entry.model_cfg.get('api_url'),
        'model_config_identity': manifest.get('model_config_identity'),
        'suite': entry.suite, 'profile': entry.profile, 'dataset': entry.dataset,
        'execution_status': 'running', 'validity_status': None, 'status_reason': 'imported',
        'phase': 'imported', 'comparability': COMPARABILITY_UNKNOWN,
        'started_at': manifest.get('created_at') or now_iso(),
        'evalscope_version': manifest.get('evalscope_version'), 'tool_version': TOOL_VERSION,
        'git_commit': git_commit(repo_dir), 'output_dir': str(output_dir),
        'config_json': json.dumps(redact(manifest.get('task_config') or {}), ensure_ascii=False,
                                  sort_keys=True),
        'protocol_identity': manifest.get('protocol_identity'),
        'sample_manifest_identity': manifest.get('sample_manifest_identity'),
        'imported_at': now_iso(),
    }


def run_plan(plan, *, store: Store, repo_dir: Path, dry_run: bool = False) -> list[AttemptResult]:
    results = []
    run_group_base = time.strftime('%Y%m%d_%H%M%S')
    try:
        for entry in plan.entries:
            api_key = _api_key(entry, dry_run=dry_run)
            attempt_id = new_attempt_id()
            run_group = f'{run_group_base}_{entry.model_alias}_{entry.suite}_{entry.profile}'
            output_dir = plan.data_root / 'outputs' / run_group / safe_slug(entry.dataset) / attempt_id
            raw_cfg = build_raw_task_config(entry, data_root=plan.data_root,
                                            output_dir=output_dir, api_key=api_key)
            if dry_run:
                resolved = {}
                try:
                    resolved = resolve_task_config(raw_cfg)
                except Exception:
                    pass
                print(json.dumps({
                    'attempt_id': attempt_id, 'suite': entry.suite, 'dataset': entry.dataset,
                    'model_alias': entry.model_alias, 'model_id': entry.model_cfg['model_id'],
                    'profile': entry.profile, 'per_subset_limit': entry.limit,
                    'batch_size': entry.batch_size, 'status': entry.status,
                    'output_dir': str(output_dir),
                    'protocol_identity': protocol_identity(resolved) if resolved else None,
                    'requires': list(entry.requires), 'pinned': entry.pinned,
                }, ensure_ascii=False, indent=2))
                continue
            resolved_cfg = resolve_task_config(raw_cfg)
            try:
                result = execute_attempt(
                    entry, store=store, attempt_id=attempt_id, run_group=run_group,
                    output_dir=output_dir, raw_cfg=raw_cfg, resolved_cfg=resolved_cfg,
                    repo_dir=repo_dir, data_root=plan.data_root,
                )
            except BatchInterrupted:
                results.append(AttemptResult(attempt_id, output_dir, 'interrupted', 'invalid',
                                             'interrupted by user', persisted=True))
                raise
            except Exception:  # noqa: BLE001 - never let one attempt kill the batch
                detail = redact_text(traceback.format_exc(), collect_secrets(entry.model_cfg))
                try:
                    # The attempt record may already be running; close it explicitly.
                    store.finish_attempt(attempt_id, {
                        'execution_status': 'failed', 'validity_status': 'invalid',
                        'status_reason': f'runner_error: {detail}', 'phase': 'runner',
                        'finished_at': now_iso(), 'comparability': COMPARABILITY_UNKNOWN,
                    })
                except Exception:  # noqa: BLE001 - DB may be unavailable; report the state we know
                    pass
                results.append(AttemptResult(attempt_id, output_dir, 'failed', 'invalid',
                                             f'runner_error: {detail}', persisted=False,
                                             error=detail))
                continue
            results.append(result)
    except KeyboardInterrupt:
        raise BatchInterrupted()
    return results


def _api_key(entry, *, dry_run: bool) -> str:
    env_name = entry.model_cfg.get('api_key_env')
    if env_name:
        value = os.environ.get(env_name, '')
        if value:
            return value
    literal = entry.model_cfg.get('api_key')
    if literal:
        return literal
    if dry_run:
        return 'EMPTY'
    raise ConfigError(
        f'{entry.model_alias}: environment variable {env_name!r} is not set'
    )
