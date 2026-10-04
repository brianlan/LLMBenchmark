import pytest

from llmbench.evidence import ReportError
from llmbench.validation import (
    COMPARABILITY_UNKNOWN,
    COMPARABILITY_VERIFIED,
    VALID,
    INVALID,
    PARTIAL,
    UNVERIFIED,
    assess_comparability,
    assess_run,
)


def quality_metric(score=0.5, is_primary=True, metric_name='accuracy', num=5,
                   semantics='quality'):
    return {'metric_key': f'mk-{metric_name}', 'metric_name': metric_name,
            'semantics_kind': semantics, 'score': score, 'num': num,
            'is_primary': is_primary}


def diagnostic_metric():
    return {'metric_key': 'mk-diag', 'metric_name': 'latency', 'semantics_kind': 'diagnostic',
            'score': 1.0, 'num': 5, 'is_primary': False}


def manifest_row(selected=5, predicted=5, reviewed=5, media=True, digests=True):
    return {
        'subset': 'default', 'selected': selected, 'predicted': predicted, 'reviewed': reviewed,
        'question_digest': 'q' if digests else None, 'media_digest': 'm' if digests else None,
        'input_digest': 'i' if digests else None, 'media_evidence': media,
    }


def test_no_report_is_invalid_but_task_completed():
    result = assess_run(report_error=ReportError('missing', 'no report'))
    assert result['execution_status'] == 'completed'
    assert result['validity_status'] == INVALID
    assert 'report_missing' in result['status_reason']


def test_no_metrics_is_invalid():
    result = assess_run(metrics=[], execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == INVALID
    assert 'no_metrics' in result['status_reason']


def test_diagnostic_only_is_invalid():
    result = assess_run(metrics=[diagnostic_metric()],
                        execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == INVALID
    assert 'no_quality_metric' in result['status_reason']


def test_missing_semantics_is_unverified_not_silently_valid():
    metric = {'semantics_kind': None, 'score': 0.5}
    result = assess_run(metrics=[metric], execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == INVALID
    assert 'metric_semantics_missing' in result['status_reason']


def test_partial_completion():
    result = assess_run(metrics=[quality_metric()],
                        execution_summary={'requested': 5, 'succeeded': 1, 'errored': 4},
                        manifest_rows=[manifest_row()])
    assert result['validity_status'] == PARTIAL
    assert 'partial_completion' in result['status_reason']


def test_all_failed_is_invalid():
    result = assess_run(metrics=[quality_metric(0.0)],
                        execution_summary={'requested': 5, 'succeeded': 0, 'errored': 5},
                        manifest_rows=[manifest_row()])
    assert result['validity_status'] == INVALID
    assert 'all_samples_failed' in result['status_reason']


def test_missing_execution_summary_is_unverified():
    result = assess_run(metrics=[quality_metric()], execution_summary=None,
                        manifest_rows=[manifest_row()])
    assert result['validity_status'] == UNVERIFIED
    assert 'missing_execution_summary' in result['status_reason']


def test_requested_exceeding_manifest_is_unverified():
    result = assess_run(metrics=[quality_metric()],
                        execution_summary={'requested': 20, 'succeeded': 20, 'errored': 0},
                        manifest_rows=[manifest_row(selected=5)])
    assert result['validity_status'] == UNVERIFIED
    assert 'requested_exceeds_manifest' in result['status_reason']


def test_complete_run_with_real_zero_stays_valid():
    result = assess_run(metrics=[quality_metric(0.0)],
                        execution_summary={'requested': 5, 'succeeded': 5, 'errored': 0},
                        manifest_rows=[manifest_row()])
    assert result['validity_status'] == VALID
    assert result['execution_status'] == 'completed'


def test_interrupted_and_task_error_statuses():
    result = assess_run(interrupted=True)
    assert result['execution_status'] == 'interrupted' and result['validity_status'] == INVALID
    result = assess_run(task_error='Traceback ...')
    assert result['execution_status'] == 'failed' and result['validity_status'] == INVALID


def test_missing_primary_metric_is_unverified():
    result = assess_run(metrics=[quality_metric(is_primary=False)],
                        execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == UNVERIFIED
    assert 'missing_primary_metric' in result['status_reason']


def test_null_primary_score_is_unverified_not_complete():
    result = assess_run(metrics=[quality_metric(score=None)],
                        execution_summary={'requested': 5, 'succeeded': 5},
                        manifest_rows=[manifest_row()])
    assert result['validity_status'] == UNVERIFIED
    assert 'primary_metric_missing_score' in result['status_reason']


def test_primary_metric_without_samples_is_unverified():
    result = assess_run(metrics=[quality_metric(num=0)],
                        execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == UNVERIFIED
    assert 'primary_metric_no_samples' in result['status_reason']


def test_primary_metric_count_above_succeeded_is_unverified():
    result = assess_run(metrics=[quality_metric(num=9)],
                        execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == UNVERIFIED
    assert 'primary_metric_count_exceeds_succeeded' in result['status_reason']


def test_diagnostic_primary_metric_is_invalid():
    metrics = [
        quality_metric(is_primary=False),
        quality_metric(metric_name='latency', semantics='diagnostic', is_primary=True),
    ]
    result = assess_run(metrics=metrics, execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == INVALID
    assert 'primary_metric_not_quality' in result['status_reason']


def test_invalid_primary_identity_is_invalid():
    metric = quality_metric(metric_name='')
    result = assess_run(metrics=[metric], execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == INVALID
    assert 'primary_metric_invalid_identity' in result['status_reason']


def test_fallback_metric_name_does_not_count_as_identity():
    metric = quality_metric()
    metric['identity_valid'] = False
    result = assess_run(metrics=[metric], execution_summary={'requested': 5, 'succeeded': 5})
    assert result['validity_status'] == INVALID
    assert 'primary_metric_invalid_identity' in result['status_reason']


def test_non_integer_execution_counts_are_unverified_not_crash():
    result = assess_run(metrics=[quality_metric()],
                        execution_summary={'requested': '5', 'succeeded': 5, 'errored': 0})
    assert result['validity_status'] == UNVERIFIED
    assert 'invalid_execution_summary' in result['status_reason']

    result = assess_run(metrics=[quality_metric()],
                        execution_summary={'requested': 5, 'succeeded': '5', 'errored': 0})
    assert result['validity_status'] == UNVERIFIED
    assert 'invalid_execution_summary' in result['status_reason']

    result = assess_run(metrics=[quality_metric()], execution_summary=['not', 'a', 'dict'])
    assert result['validity_status'] == UNVERIFIED
    assert 'invalid_execution_summary' in result['status_reason']

    result = assess_run(metrics=[quality_metric()],
                        execution_summary={'requested': 5, 'succeeded': 5,
                                           'incomplete': 'false'})
    assert result['validity_status'] == UNVERIFIED
    assert 'invalid_execution_summary' in result['status_reason']


def test_comparability_verified_and_unknown():
    verified_row = manifest_row()
    verified_row.update({'predicted_missing': [], 'predicted_extra': [], 'predicted_duplicates': 0,
                         'reviewed_missing': [], 'reviewed_extra': [], 'reviewed_duplicates': 0})
    evidence = {'subsets': {'default': {}}, 'malformed_lines': []}
    status, _ = assess_comparability([verified_row], evidence)
    assert status == COMPARABILITY_VERIFIED

    status, reasons = assess_comparability([{**verified_row, 'reviewed_missing': ['3']}], evidence)
    assert status == COMPARABILITY_UNKNOWN
    assert any('missing evidence keys' in reason for reason in reasons)

    status, reasons = assess_comparability([verified_row], {'subsets': {}, 'malformed_lines': [{}]})
    assert status == COMPARABILITY_UNKNOWN
    assert any('malformed' in reason for reason in reasons)

    status, reasons = assess_comparability([manifest_row(media=False)], evidence)
    assert status == COMPARABILITY_UNKNOWN
    assert any('media' in reason for reason in reasons)

    status, reasons = assess_comparability([], evidence)
    assert status == COMPARABILITY_UNKNOWN
    assert 'no_sample_manifest' in reasons
