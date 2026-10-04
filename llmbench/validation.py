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
    or scoring failures are not.  ``complete`` additionally requires the
    benchmark's primary quality metric to be identifiable and to carry a usable
    numeric score -- a stored NULL is not a usable result.  Missing completion
    evidence keeps the run ``unverified``.
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

    primary = next((row for row in metrics if row.get('is_primary')), None)
    if primary is None:
        _reason(reasons, 'missing_primary_metric', 'no primary metric identity in the report')
        return _outcome(COMPLETED, UNVERIFIED, reasons)
    if primary.get('semantics_kind') != 'quality':
        _reason(reasons, 'primary_metric_not_quality', str(primary.get('metric_name')))
        return _outcome(COMPLETED, INVALID, reasons)
    if not primary.get('metric_name') or not primary.get('metric_key'):
        _reason(reasons, 'primary_metric_invalid_identity')
        return _outcome(COMPLETED, INVALID, reasons)
    if primary.get('score') is None:
        _reason(reasons, 'primary_metric_missing_score', str(primary.get('metric_name')))
        return _outcome(COMPLETED, UNVERIFIED, reasons)
    metric_num = primary.get('num')
    if metric_num is None or metric_num <= 0:
        _reason(reasons, 'primary_metric_no_samples', str(metric_num))
        return _outcome(COMPLETED, UNVERIFIED, reasons)

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

    # Count consistency is checked per benchmark semantics: an aggregate metric
    # may count fewer rows than succeeded, but never more.
    if metric_num > succeeded:
        _reason(reasons, 'primary_metric_count_exceeds_succeeded',
                f'metric_num={metric_num} succeeded={succeeded}')
        return _outcome(COMPLETED, UNVERIFIED, reasons)

    _reason(reasons, 'complete', f'succeeded={succeeded}/{requested}')
    return _outcome(COMPLETED, VALID, reasons)


def assess_comparability(manifest_rows, output_evidence) -> tuple[str, list[str]]:
    """Content evidence must be complete and ID-matched for a run to claim verified."""
    reasons = []
    rows = manifest_rows or []
    evidence = output_evidence or {}
    if not rows:
        return COMPARABILITY_UNKNOWN, ['no_sample_manifest']

    manifest_subsets = {row.get('subset') for row in rows}
    evidence_subsets = set((evidence.get('subsets') or {}).keys())
    unexpected_subsets = sorted(evidence_subsets - manifest_subsets)
    if unexpected_subsets:
        reasons.append(f'unexpected evidence subsets: {unexpected_subsets[:5]}')
    malformed = evidence.get('malformed_lines') or []
    if malformed:
        reasons.append(f'{len(malformed)} malformed evidence line(s)')

    for row in rows:
        subset = row.get('subset')
        if not row.get('question_digest') or not row.get('input_digest'):
            reasons.append(f'{subset}: missing content digest')
        if not row.get('media_evidence', True):
            reasons.append(f'{subset}: media content not verifiable')
        selected = row.get('selected') or 0
        if row.get('predicted') is None or row.get('reviewed') is None:
            reasons.append(f'{subset}: coverage unknown')
            continue
        if row.get('predicted') < selected or row.get('reviewed') < selected:
            reasons.append(
                f'{subset}: coverage gap predicted={row.get("predicted")} '
                f'reviewed={row.get("reviewed")}/{selected}'
            )
        for kind in ('predicted', 'reviewed'):
            missing = row.get(f'{kind}_missing') or []
            extra = row.get(f'{kind}_extra') or []
            duplicates = row.get(f'{kind}_duplicates') or 0
            if missing:
                reasons.append(f'{subset}: {kind} missing ids {missing[:5]}')
            if extra:
                reasons.append(f'{subset}: {kind} unexpected ids {extra[:5]}')
            if duplicates:
                reasons.append(f'{subset}: {kind} duplicate ids x{duplicates}')

    if reasons:
        return COMPARABILITY_UNKNOWN, reasons
    return COMPARABILITY_VERIFIED, ['content evidence complete']


def _outcome(execution_status: str, validity_status: str, reasons: list) -> dict:
    return {
        'execution_status': execution_status,
        'validity_status': validity_status,
        'status_reason': '; '.join(reasons) if reasons else '',
    }
