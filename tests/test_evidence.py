import base64
import json
from pathlib import Path

import pytest

from llmbench.evidence import (
    MetricParseError,
    ReportError,
    apply_coverage,
    build_sample_manifest,
    collect_output_evidence,
    locate_report,
    metric_rows,
    summarize_diagnostics,
)
from llmbench.store import Store

FIXTURES = Path(__file__).parent / 'fixtures'

MODEL = 'MiniMax-M3.1-Flash-Preview'


class FakeSample:
    def __init__(self, sample_id, input_value):
        self.id = sample_id
        self.input = input_value


def quality_semantics():
    return {'kind': 'quality', 'display_kind': 'percent', 'direction': 'higher_is_better', 'display_unit': '%'}


def make_metric(name, aggregation='mean', dimensions=None, score=0.5, num=5, primitive=False):
    if dimensions is None:
        dimensions = {'k': 5} if aggregation == 'pass_at_k' else {}
    metric = {
        'identity': {'name': name, 'aggregation': aggregation, 'dimensions': dimensions},
        'num': num, 'score': score, 'macro_score': score,
        'categories': [{
            'name': ['default'], 'num': num, 'score': score, 'macro_score': score,
            'subsets': [{'name': 'default', 'score': score, 'num': num, 'is_aggregate': False}],
        }],
        'semantics': quality_semantics(),
    }
    return metric


def make_report(metrics=None, primary=None, execution=None, dataset='gpqa_diamond', model=MODEL):
    return {
        'schema_version': 2, 'dataset_name': dataset, 'model_name': model,
        'metrics': metrics or [], 'primary_metric_identity': primary,
        'execution_summary': execution or {'requested': 5, 'succeeded': 5, 'errored': 0, 'incomplete': False},
    }


def write_report(root: Path, dataset, model, report):
    path = Path(root) / 'reports' / model / f'{dataset}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# R2: report ownership
# ---------------------------------------------------------------------------

def test_locate_report_accepts_exactly_one_match(tmp_path):
    write_report(tmp_path, 'gpqa_diamond', MODEL, make_report())
    write_report(tmp_path, 'docvqa', MODEL, make_report(dataset='docvqa'))
    path, report = locate_report(tmp_path, 'gpqa_diamond', MODEL)
    assert path.name == 'gpqa_diamond.json'
    assert report['dataset_name'] == 'gpqa_diamond'


def test_locate_report_rejects_wrong_dataset(tmp_path):
    write_report(tmp_path, 'docvqa', MODEL, make_report(dataset='docvqa'))
    with pytest.raises(ReportError) as excinfo:
        locate_report(tmp_path, 'gpqa_diamond', MODEL)
    assert excinfo.value.kind == 'mismatch'


def test_locate_report_rejects_wrong_model(tmp_path):
    write_report(tmp_path, 'gpqa_diamond', 'other-model', make_report(model='other-model'))
    with pytest.raises(ReportError) as excinfo:
        locate_report(tmp_path, 'gpqa_diamond', MODEL)
    assert excinfo.value.kind == 'mismatch'


def test_locate_report_rejects_ambiguous(tmp_path):
    write_report(tmp_path, 'gpqa_diamond', MODEL, make_report())
    write_report(tmp_path, 'gpqa_diamond', MODEL + '-copy', make_report(model=MODEL + '-copy'))
    _, report = locate_report(tmp_path, 'gpqa_diamond', MODEL)
    assert report['model_name'] == MODEL
    (tmp_path / 'reports' / 'alias').mkdir(parents=True)
    (tmp_path / 'reports' / 'alias' / 'gpqa_diamond.json').write_text(json.dumps(make_report()), encoding='utf-8')
    with pytest.raises(ReportError) as excinfo:
        locate_report(tmp_path, 'gpqa_diamond', MODEL)
    assert excinfo.value.kind == 'ambiguous'


def test_locate_report_reports_corrupt_json(tmp_path):
    path = tmp_path / 'reports' / MODEL / 'gpqa_diamond.json'
    path.parent.mkdir(parents=True)
    path.write_text('{not json', encoding='utf-8')
    with pytest.raises(ReportError) as excinfo:
        locate_report(tmp_path, 'gpqa_diamond', MODEL)
    assert excinfo.value.kind == 'parse_error'


def test_locate_report_missing(tmp_path):
    (tmp_path / 'reports').mkdir()
    with pytest.raises(ReportError) as excinfo:
        locate_report(tmp_path, 'gpqa_diamond', MODEL)
    assert excinfo.value.kind == 'missing'


# ---------------------------------------------------------------------------
# R3: metric identity, NULL vs zero
# ---------------------------------------------------------------------------

def test_real_eval_scope_fixture_parses():
    report = json.loads((FIXTURES / 'report_gpqa_diamond.json').read_text(encoding='utf-8'))
    rows = metric_rows(report)
    top = [row for row in rows if not row['category'] and not row['subset']]
    assert top and top[0]['is_primary'] is True
    assert top[0]['semantics_kind'] == 'quality'
    assert top[0]['score'] is not None


def test_same_name_different_aggregation_and_dimensions_are_distinct(tmp_path):
    metrics = [
        make_metric('accuracy', aggregation='mean', score=0.4),
        make_metric('accuracy', aggregation='pass_at_k', score=0.7),
        make_metric('accuracy', aggregation='pass_at_k', dimensions={'k': 3}, score=0.6),
    ]
    report = make_report(metrics=metrics, primary={'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}})
    rows = metric_rows(report)
    top = [row for row in rows if not row['category'] and not row['subset']]
    assert len({row['metric_key'] for row in top}) == 3
    assert len([row for row in top if row['is_primary']]) == 1

    # store must keep all three (no INSERT OR REPLACE overwrite)
    store = Store(tmp_path / 'results.db')
    store.start_attempt({
        'run_id': 'r', 'run_group': 'g', 'model_alias': 'minimax', 'model_id': 'm',
        'api_url': 'u', 'model_config_identity': 'mid', 'suite': 'knowledge', 'profile': 'lite',
        'dataset': 'gpqa_diamond', 'execution_status': 'running', 'started_at': 't',
        'evalscope_version': '1.12.0', 'tool_version': '0.2.0', 'git_commit': 'g',
        'output_dir': 'o', 'config_json': '{}',
    })
    store.finish_attempt('r', {
        'execution_status': 'completed', 'validity_status': 'complete', 'status_reason': 'ok',
        'phase': 'persist', 'comparability': 'verified', 'finished_at': 't2',
        'num_requested': 5, 'num_succeeded': 5, 'num_errored': 0, 'incomplete': False,
        'report_path': 'p', 'metrics': rows, 'sample_manifest': [], 'diagnostics': None,
        'perf_metrics': None, 'primary_metric_identity': {'name': 'accuracy'},
        'protocol_identity': 'pid', 'sample_manifest_identity': 'smid',
    })
    assert store.conn.execute('SELECT COUNT(*) AS n FROM metrics WHERE run_id=?', ('r',)).fetchone()['n'] >= 3
    store.close()


def test_score_none_zero_and_invalid_values():
    report = make_report(metrics=[make_metric('accuracy', score=None)], primary=None)
    assert metric_rows(report)[0]['score'] is None

    report = make_report(metrics=[make_metric('accuracy', score=0.0)])
    assert metric_rows(report)[0]['score'] == 0.0

    report = make_report(metrics=[make_metric('accuracy', score='0.5')])
    assert metric_rows(report)[0]['score'] == 0.5

    for bad in ('not-a-number', float('nan'), float('inf'), True):
        report = make_report(metrics=[make_metric('accuracy', score=bad)])
        with pytest.raises(MetricParseError):
            metric_rows(report)


# ---------------------------------------------------------------------------
# R6: sample manifest and content evidence
# ---------------------------------------------------------------------------

def test_question_digest_ignores_answers_and_input_digest_tracks_template():
    raw = {'default': [FakeSample(0, 'What is 2+2?'), FakeSample(1, 'What is 3+3?')]}
    processed = {'default': [
        FakeSample(0, 'Answer the question.\nWhat is 2+2?'),
        FakeSample(1, 'Answer the question.\nWhat is 3+3?'),
    ]}
    rows, details = build_sample_manifest(raw, processed)
    assert rows[0]['selected'] == 2
    assert set(details['default'][0]) == {'id', 'question_digest', 'media_digest', 'input_digest', 'media_evidence'}

    changed_template = {'default': [
        FakeSample(0, 'Reply briefly.\nWhat is 2+2?'),
        FakeSample(1, 'Reply briefly.\nWhat is 3+3?'),
    ]}
    other_rows, _ = build_sample_manifest(raw, changed_template)
    assert other_rows[0]['question_digest'] == rows[0]['question_digest']
    assert other_rows[0]['input_digest'] != rows[0]['input_digest']


def test_build_sample_manifest_handles_evalscope_datasetdict():
    class FakeDataset:
        def __init__(self, samples):
            self.samples = samples

        def __iter__(self):
            return iter(self.samples)

    class FakeDatasetDict:
        def __init__(self, mapping):
            self.datasets = mapping

        def __getitem__(self, key):
            return self.datasets[key]

    raw = FakeDatasetDict({'default': FakeDataset([FakeSample(0, 'Q')])})
    processed = FakeDatasetDict({'default': FakeDataset([FakeSample(0, 'templated Q')])})
    rows, details = build_sample_manifest(raw, processed)
    assert rows[0]['subset'] == 'default'
    assert rows[0]['selected'] == 1
    assert details['default'][0]['id'] == '0'


def test_media_change_changes_media_digest_and_url_is_unverified():
    image_a = 'data:image/png;base64,' + base64.b64encode(b'image-a').decode()
    image_b = 'data:image/png;base64,' + base64.b64encode(b'image-b').decode()

    def processed(image):
        part = type('Part', (), {'type': 'image', 'image': image})()
        message = type('Msg', (), {'content': [part], 'role': 'user'})()
        return {'default': [FakeSample(0, [message])]}

    rows_a, _ = build_sample_manifest(None, processed(image_a))
    rows_b, _ = build_sample_manifest(None, processed(image_b))
    assert rows_a[0]['media_digest'] != rows_b[0]['media_digest']
    assert rows_a[0]['media_evidence'] is True

    rows_url, _ = build_sample_manifest(None, processed('https://example.com/a.png'))
    assert rows_url[0]['media_evidence'] is False


def test_coverage_detects_missing_review(tmp_path):
    out = tmp_path / 'attempt'
    (out / 'predictions' / MODEL).mkdir(parents=True)
    (out / 'reviews' / MODEL).mkdir(parents=True)
    line = json.dumps({'index': 0, 'model_output': {'choices': [{'stop_reason': 'stop'}], 'usage': {'input_tokens': 10, 'output_tokens': 5}}})
    (out / 'predictions' / MODEL / 'gpqa_diamond_default.jsonl').write_text(line + '\n' + line.replace('"index": 0', '"index": 1') + '\n')
    (out / 'reviews' / MODEL / 'gpqa_diamond_default.jsonl').write_text(line + '\n')

    evidence = collect_output_evidence(out, 'gpqa_diamond')
    manifest = [{'subset': 'default', 'selected': 2, 'sample_ids': ['0', '1'],
                 'question_digest': 'q', 'media_digest': 'm', 'input_digest': 'i', 'media_evidence': True}]
    merged = apply_coverage(manifest, evidence)
    assert merged[0]['predicted'] == 2
    assert merged[0]['reviewed'] == 1


def test_diagnostics_count_truncation_and_ignore_tool_calls(tmp_path):
    out = tmp_path / 'attempt'
    (out / 'predictions' / MODEL).mkdir(parents=True)
    lines = []
    for index in range(10):
        stop = 'max_tokens' if index == 0 else 'stop'
        lines.append(json.dumps({
            'index': index,
            'model_output': {'choices': [{'stop_reason': stop}], 'usage': {'input_tokens': 1, 'output_tokens': 1}},
        }))
    # one agentic-style record with intermediate tool_calls must not count as truncation
    lines.append(json.dumps({
        'index': 10,
        'model_output': {'choices': [{'stop_reason': 'tool_calls'}], 'usage': {}},
        'agent_trace': {'events': [
            {'type': 'model_generate', 'payload': {'stop_reason': 'tool_calls'}},
            {'type': 'tool_result'},
            {'type': 'model_generate', 'payload': {'stop_reason': 'stop'}},
        ]},
    }))
    (out / 'predictions' / MODEL / 'gpqa_diamond_default.jsonl').write_text('\n'.join(lines) + '\n')

    evidence = collect_output_evidence(out, 'gpqa_diamond')
    diagnostics = summarize_diagnostics(evidence)
    assert diagnostics['truncated_calls'] == 1
    assert diagnostics['content_filter_calls'] == 0
    assert diagnostics['agent_steps']['max'] == 2
    assert diagnostics['generation_calls'] == 12


def _manifest_for_evidence(sample_ids=('0', '1')):
    return [{
        'subset': 'default', 'selected': len(sample_ids), 'sample_ids': list(sample_ids),
        'question_digest': 'q', 'media_digest': 'm', 'input_digest': 'i', 'media_evidence': True,
    }]


def _write_evidence_files(out, report_id, predicted, reviewed, dataset='gpqa_diamond'):
    for folder, ids in (('predictions', predicted), ('reviews', reviewed)):
        root = out / folder / report_id
        root.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({'index': sample_id, 'model_output': {
            'choices': [{'stop_reason': 'stop'}], 'usage': {'input_tokens': 1, 'output_tokens': 1}}})
            for sample_id in ids]
        (root / f'{dataset}_default.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_comparability_requires_matching_sample_ids(tmp_path):
    out = tmp_path / 'attempt'
    _write_evidence_files(out, MODEL, [0, 1], [0, 1])
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(), evidence)
    assert (rows[0]['predicted'], rows[0]['reviewed']) == (2, 2)
    assert rows[0]['predicted_missing'] == [] and rows[0]['reviewed_missing'] == []



def test_disjoint_ids_are_not_verified(tmp_path):
    out = tmp_path / 'attempt'
    _write_evidence_files(out, MODEL, [100, 101], [100, 101])
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(), evidence)
    assert rows[0]['predicted'] == 0
    assert rows[0]['predicted_missing'] == ['0', '1']
    assert rows[0]['predicted_extra'] == ['100', '101']
    from llmbench.validation import assess_comparability, COMPARABILITY_UNKNOWN
    status, reasons = assess_comparability(rows, evidence)
    assert status == COMPARABILITY_UNKNOWN
    assert any('missing evidence keys' in reason for reason in reasons)


def test_other_model_directory_is_not_this_attempts_evidence(tmp_path):
    out = tmp_path / 'attempt'
    _write_evidence_files(out, 'other-model', [0, 1], [0, 1])
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(), evidence)
    assert rows[0]['predicted'] == 0
    assert rows[0]['predicted_missing'] == ['0', '1']


def test_duplicates_and_malformed_lines_block_verification(tmp_path):
    out = tmp_path / 'attempt'
    _write_evidence_files(out, MODEL, [0, 0], [0, 1])
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(), evidence)
    assert rows[0]['predicted_duplicates'] == 1

    malformed_out = tmp_path / 'attempt2'
    _write_evidence_files(malformed_out, MODEL, [0], [0, 1])
    path = malformed_out / 'predictions' / MODEL / 'gpqa_diamond_default.jsonl'
    path.write_text(path.read_text(encoding='utf-8') + '{broken\n', encoding='utf-8')
    evidence = collect_output_evidence(malformed_out, 'gpqa_diamond', report_id=MODEL)
    from llmbench.validation import assess_comparability, COMPARABILITY_UNKNOWN
    status, reasons = assess_comparability(apply_coverage(_manifest_for_evidence(), evidence), evidence)
    assert status == COMPARABILITY_UNKNOWN
    assert any('malformed' in reason for reason in reasons)


def test_repeat_ids_are_not_duplicates(tmp_path):
    out = tmp_path / 'attempt'
    root = out / 'predictions' / MODEL
    root.mkdir(parents=True)
    lines = [json.dumps({'index': 0, 'repeat_id': repeat,
                         'model_output': {'choices': [{'stop_reason': 'stop'}], 'usage': {}}})
             for repeat in (0, 1)]
    (root / 'gpqa_diamond_default.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(sample_ids=('0',)), evidence)
    assert rows[0]['predicted'] == 1
    assert rows[0]['predicted_duplicates'] == 0


def _write_repeat_evidence(out, predicted_repeats, reviewed_repeats):
    for folder, repeats in (('predictions', predicted_repeats), ('reviews', reviewed_repeats)):
        root = out / folder / MODEL
        root.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({'index': 0, 'repeat_id': repeat, 'model_output': {
            'choices': [{'stop_reason': 'stop'}], 'usage': {}}}) for repeat in repeats]
        (root / 'gpqa_diamond_default.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_review_of_another_repeat_is_not_verified(tmp_path):
    # prediction repeat 0, review repeat 1: same base sample, different execution
    out = tmp_path / 'attempt'
    _write_repeat_evidence(out, [0], [1])
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(sample_ids=('0',)), evidence)
    assert rows[0]['reviewed_missing'] == ['0#r0']
    assert rows[0]['reviewed_extra'] == ['0#r1']
    from llmbench.validation import assess_comparability, COMPARABILITY_UNKNOWN
    status, _ = assess_comparability(rows, evidence)
    assert status == COMPARABILITY_UNKNOWN


def test_predicted_repeat_without_review_is_not_verified(tmp_path):
    out = tmp_path / 'attempt'
    _write_repeat_evidence(out, [0, 1], [0])
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(sample_ids=('0',)), evidence)
    assert rows[0]['reviewed_missing'] == ['0#r1']
    from llmbench.validation import assess_comparability, COMPARABILITY_UNKNOWN
    status, _ = assess_comparability(rows, evidence)
    assert status == COMPARABILITY_UNKNOWN


def test_matching_repeats_are_verified(tmp_path):
    out = tmp_path / 'attempt'
    _write_repeat_evidence(out, [0, 1], [0, 1])
    evidence = collect_output_evidence(out, 'gpqa_diamond', report_id=MODEL)
    rows = apply_coverage(_manifest_for_evidence(sample_ids=('0',)), evidence)
    assert rows[0]['reviewed_missing'] == [] and rows[0]['reviewed_extra'] == []
    assert rows[0]['reviewed'] == 2
    from llmbench.validation import assess_comparability, COMPARABILITY_VERIFIED
    status, _ = assess_comparability(rows, evidence)
    assert status == COMPARABILITY_VERIFIED


def test_report_identity_without_name_is_marked_invalid():
    report = make_report(metrics=[make_metric('accuracy')], primary=
                         {'aggregation': 'mean'})
    report['metrics'][0]['identity'] = {'aggregation': 'mean', 'dimensions': {}}
    rows = [row for row in metric_rows(report) if not row['category']]
    assert rows[0]['is_primary'] is True  # hash matched the declared primary
    assert rows[0]['identity_valid'] is False
    assert rows[0]['metric_name'] == 'metric'


def test_malformed_jsonl_lines_are_reported_not_hidden(tmp_path):
    out = tmp_path / 'attempt'
    (out / 'predictions' / MODEL).mkdir(parents=True)
    (out / 'predictions' / MODEL / 'gpqa_diamond_default.jsonl').write_text(
        json.dumps({'index': 0, 'model_output': {'choices': [{'stop_reason': 'stop'}], 'usage': {}}}) + '\n'
        + '{broken\n'
    )
    evidence = collect_output_evidence(out, 'gpqa_diamond')
    assert evidence['malformed_lines']
    assert evidence['subsets']['default']['predicted'] == ['0#r0']


def test_real_prediction_fixture_integration(tmp_path):
    out = tmp_path / 'attempt'
    reviews = out / 'reviews' / MODEL
    predictions = out / 'predictions' / MODEL
    reviews.mkdir(parents=True)
    predictions.mkdir(parents=True)
    (reviews / 'gpqa_diamond_default.jsonl').write_text(
        (FIXTURES / 'reviews_gpqa_diamond_default.jsonl').read_text(encoding='utf-8'))
    (predictions / 'gpqa_diamond_default.jsonl').write_text(
        (FIXTURES / 'predictions_gpqa_diamond_default.jsonl').read_text(encoding='utf-8'))
    evidence = collect_output_evidence(out, 'gpqa_diamond')
    assert len(evidence['subsets']['default']['predicted']) == 5
    assert len(evidence['subsets']['default']['reviewed']) == 5
    diagnostics = summarize_diagnostics(evidence)
    assert diagnostics['generation_calls'] == 5


def test_duplicate_metric_identity_is_rejected():
    metric = make_metric('accuracy')
    report = make_report(metrics=[metric, dict(metric)], primary=None)
    with pytest.raises(MetricParseError, match='duplicate metric identity'):
        metric_rows(report)
