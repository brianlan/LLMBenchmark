"""Run lifecycle statuses: execution status, formal validity and comparability verdicts."""

from __future__ import annotations

# execution lifecycle
RUNNING = 'running'
COMPLETED = 'completed'
FAILED = 'failed'
INTERRUPTED = 'interrupted'

# formal validity (eligibility for comparison)
VALID = 'complete'
PARTIAL = 'partial'
INVALID = 'invalid'
UNVERIFIED = 'unverified'
LEGACY = 'legacy'

FORMAL_STATUSES = (VALID,)
FORMAL_STATUSES_WITH_PARTIAL = (VALID, PARTIAL)

COMPARABILITY_VERIFIED = 'verified'
COMPARABILITY_UNKNOWN = 'unknown'
COMPARABILITY_LEGACY = 'legacy'


def _reason(reasons, key, detail=None):
    reasons.append(f'{key}: {detail}' if detail else key)


def assess_run(*, interrupted=False, task_error=None, report_error=None, metrics=None,
               execution_summary=None, manifest_rows=None) -> dict:
    """Return execution_status, validity_status and status_reason.

    A model that answers and scores zero is a valid complete run; infrastructure
    or scoring failures are not, and missing completion evidence is `unverified`.
    """
    reasons = []
    if interrupted:
        return _outcome(INTERRUPTED, INVALID, ['interrupted by user'])
    if task_error:
        return _outcome(FAILED, INVALID, [f'task_error: {task_error}'])

    if report_error is not None:
        kind = getattr(report_error, 'kind', 'error')
        _reason(reasons, f'report_{kind}', str(report_error))
        return _outcome(COMPLETED, INVALID, reasons)

    metrics = metrics or []
    if not metrics:
        _reason(reasons, 'no_metrics')
        return _outcome(COMPLETED, INVALID, reasons)

    quality = [row for row in metrics if row.get('semantics_kind') == 'quality']
    if not quality:
        if any(row.get('semantics_kind') for row in metrics):
            _reason(reasons, 'no_quality_metric', 'only diagnostic metrics were reported')
        else:
            _reason(reasons, 'metric_semantics_missing')
        return _outcome(COMPLETED, INVALID, reasons)

    if not execution_summary:
        _reason(reasons, 'missing_execution_summary')
        return _outcome(COMPLETED, UNVERIFIED, reasons)

    requested = execution_summary.get('requested')
    succeeded = execution_summary.get('succeeded')
    errored = execution_summary.get('errored') or 0
    if requested is None or requested <= 0:
        _reason(reasons, 'no_requested_samples')
        return _outcome(COMPLETED, UNVERIFIED, reasons)

    selected = sum((row.get('selected') or 0) for row in (manifest_rows or []))
    if manifest_rows and requested > selected:
        _reason(reasons, 'requested_exceeds_manifest', f'requested={requested} selected={selected}')
        return _outcome(COMPLETED, UNVERIFIED, reasons)
    if succeeded is None:
        _reason(reasons, 'missing_succeeded_count')
        return _outcome(COMPLETED, UNVERIFIED, reasons)

    if succeeded <= 0:
        _reason(reasons, 'all_samples_failed', f'requested={requested} errored={errored}')
        return _outcome(COMPLETED, INVALID, reasons)

    if errored or succeeded < requested:
        _reason(reasons, 'partial_completion', f'succeeded={succeeded}/{requested} errored={errored}')
        return _outcome(COMPLETED, PARTIAL, reasons)

    if execution_summary.get('incomplete'):
        _reason(reasons, 'upstream_marked_incomplete')
        return _outcome(COMPLETED, PARTIAL, reasons)

    _reason(reasons, 'complete', f'succeeded={succeeded}/{requested}')
    return _outcome(COMPLETED, VALID, reasons)


def assess_comparability(manifest_rows, output_evidence) -> tuple[str, list[str]]:
    """Content evidence must be complete for a run to claim verified comparability."""
    reasons = []
    rows = manifest_rows or []
    if not rows:
        return COMPARABILITY_UNKNOWN, ['no_sample_manifest']
    for row in rows:
        subset = row.get('subset')
        if not row.get('question_digest') or not row.get('input_digest'):
            reasons.append(f'{subset}: missing content digest')
        if not row.get('media_evidence', True):
            reasons.append(f'{subset}: media content not verifiable')
        selected = row.get('selected') or 0
        predicted = row.get('predicted')
        reviewed = row.get('reviewed')
        if predicted is None or reviewed is None:
            reasons.append(f'{subset}: coverage unknown')
        elif predicted < selected or reviewed < selected:
            reasons.append(f'{subset}: coverage gap predicted={predicted} reviewed={reviewed}/{selected}')
    if reasons:
        return COMPARABILITY_UNKNOWN, reasons
    return COMPARABILITY_VERIFIED, ['content evidence complete']


def _outcome(execution_status: str, validity_status: str, reasons: list) -> dict:
    return {
        'execution_status': execution_status,
        'validity_status': validity_status,
        'status_reason': '; '.join(reasons) if reasons else '',
    }
