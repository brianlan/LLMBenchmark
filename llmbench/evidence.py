"""Evidence collection: sample manifests, report location/parsing, run diagnostics.

This module is the only place that reads EvalScope outputs.  Everything it emits is
either raw upstream data or a digest/coverage fact derived from it -- never guessed.
"""

from __future__ import annotations

import base64
import json
import math
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from .util import (
    atomic_write_json,
    digest,
    file_digest,
    read_jsonl,
    to_jsonable,
)


class EvidenceError(RuntimeError):
    """Base class for evidence collection failures."""


class ReportError(EvidenceError):
    def __init__(self, kind: str, message: str, details=None):
        super().__init__(message)
        self.kind = kind  # missing | mismatch | ambiguous | parse_error
        self.details = details or {}


class MetricParseError(EvidenceError):
    pass


# ---------------------------------------------------------------------------
# dataset capture
# ---------------------------------------------------------------------------

@contextmanager
def capture_loaded_dataset(callback):
    """Capture the exact samples EvalScope loads, before inference.

    Patches the adapter base class for the duration of one run so no dataset is
    loaded twice and no prompt post-processing is replayed.
    """
    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter

    state = {'raw': None, 'processed': None}
    original_load = DefaultDataAdapter.load
    original_load_dataset = DefaultDataAdapter.load_dataset

    def load_wrapper(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        state['raw'] = result[0] if isinstance(result, tuple) else result
        return result

    def load_dataset_wrapper(self, *args, **kwargs):
        result = original_load_dataset(self, *args, **kwargs)
        state['processed'] = result
        callback(state['raw'], state['processed'])
        return result

    DefaultDataAdapter.load = load_wrapper
    DefaultDataAdapter.load_dataset = load_dataset_wrapper
    try:
        yield state
    finally:
        DefaultDataAdapter.load = original_load
        DefaultDataAdapter.load_dataset = original_load_dataset


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------

def _iter_parts(value, role=None):
    """Yield (kind, role, payload) for a str, a ChatMessage list or a media part.

    ``kind`` is ``'text'`` or ``'media'``; ``role`` is the chat role when known.
    """
    if isinstance(value, str):
        yield 'text', role, value
        return
    if isinstance(value, dict):
        if value.get('type') == 'text':
            yield 'text', role, str(value.get('text', ''))
        else:
            yield 'media', role, value
        return
    if not isinstance(value, (list, tuple)):
        return
    for item in value:
        content = getattr(item, 'content', item)
        item_role = getattr(item, 'role', role)
        if isinstance(content, str):
            yield 'text', item_role, content
        elif isinstance(content, (list, tuple)):
            for part in content:
                part_type = getattr(part, 'type', None)
                if part_type == 'text' or (isinstance(part, dict) and part.get('type') == 'text'):
                    text = part.get('text') if isinstance(part, dict) else getattr(part, 'text', '')
                    yield 'text', item_role, text or ''
                else:
                    yield 'media', item_role, part
        else:
            yield 'media', item_role, content


def _media_fingerprint(part):
    """Return (fingerprint, content_evidence) for one media part."""
    media = part.get('image') if isinstance(part, dict) else getattr(part, 'image', None)
    if media is None and isinstance(part, dict):
        media = part.get('url') or part.get('video') or part.get('audio')
    if media is None:
        media = getattr(part, 'url', None)
    if isinstance(media, (bytes, bytearray)):
        return digest({'media_sha256': bytes(media).hex()}), True
    if isinstance(media, str):
        if media.startswith('data:'):
            _, _, payload = media.partition(',')
            try:
                raw = base64.b64decode(payload)
            except ValueError:
                return digest({'media': media[:64]}), False
            return digest({'media_sha256': raw.hex()}), True
        if media.startswith(('http://', 'https://')):
            return digest({'media_url': media}), False
        path = Path(media)
        if path.exists() and path.is_file():
            return file_digest(path), True
        return digest({'media_ref': media}), False
    tobytes = getattr(media, 'tobytes', None)
    if callable(tobytes):
        try:
            return digest({'media_sha256': tobytes().hex()}), True
        except Exception:
            return None, False
    if media is None:
        return None, False
    return digest({'media_repr': to_jsonable(media)}), False


def question_digest(value) -> str:
    return digest({'question_text': [payload for kind, _, payload in _iter_parts(value) if kind == 'text']})


def media_digest(value):
    fingerprints = []
    evidence = True
    for kind, _, part in _iter_parts(value):
        if kind != 'media':
            continue
        fingerprint, ok = _media_fingerprint(part)
        fingerprints.append(fingerprint)
        evidence = evidence and ok
    return digest({'media': fingerprints}), evidence


def input_digest(value) -> str:
    messages = []
    for kind, role, payload in _iter_parts(value):
        if kind == 'text':
            messages.append({'role': role, 'kind': 'text', 'text': payload})
        else:
            fingerprint, _ = _media_fingerprint(payload)
            messages.append({'role': role, 'kind': 'media', 'fingerprint': fingerprint})
    return digest({'input_messages': messages})


def _dataset_mapping(dataset):
    """Normalize plain dicts and EvalScope ``DatasetDict``/``Dataset`` objects."""
    if dataset is None:
        return {}
    underlying = getattr(dataset, 'datasets', None)
    if isinstance(underlying, dict):
        return underlying
    if isinstance(dataset, dict):
        return dataset
    try:
        return dict(dataset.items())
    except (AttributeError, TypeError, ValueError):
        return {}


def build_sample_manifest(raw_dataset, processed_dataset):
    """Return (per-subset rows, per-sample detail map)."""
    rows = []
    details = {}
    raw_map = _dataset_mapping(raw_dataset)
    for subset, samples in _dataset_mapping(processed_dataset).items():
        raw_by_id = {getattr(sample, 'id', None): sample for sample in raw_map.get(subset) or []}
        per_sample = []
        media_evidence = True
        for sample in samples:
            raw = raw_by_id.get(getattr(sample, 'id', None))
            question_source = raw.input if raw is not None else sample.input
            media, media_ok = media_digest(sample.input)
            per_sample.append({
                'id': str(getattr(sample, 'id', None)),
                'question_digest': question_digest(question_source),
                'media_digest': media,
                'input_digest': input_digest(sample.input),
                'media_evidence': media_ok,
            })
            media_evidence = media_evidence and media_ok
        rows.append({
            'subset': subset,
            'selected': len(per_sample),
            'sample_ids': [item['id'] for item in per_sample],
            'question_digest': digest([item['question_digest'] for item in per_sample]),
            'media_digest': digest([item['media_digest'] for item in per_sample]),
            'input_digest': digest([item['input_digest'] for item in per_sample]),
            'media_evidence': media_evidence,
        })
        details[subset] = per_sample
    return rows, details


def write_run_manifest(output_dir: Path, manifest: dict) -> Path:
    path = Path(output_dir) / 'run_manifest.json'
    atomic_write_json(path, manifest)
    return path


def read_run_manifest(output_dir: Path):
    path = Path(output_dir) / 'run_manifest.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# report location and parsing
# ---------------------------------------------------------------------------

def _looks_like_report(data) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get('metrics'), list)
        and 'dataset_name' in data
    )


def locate_report(output_dir: Path, expected_dataset: str, expected_model_id: str,
                  expected_pretty_name: str | None = None):
    """Return (path, report) for exactly one report belonging to this attempt.

    Raises ReportError with kind missing/mismatch/ambiguous/parse_error.
    """
    reports_root = Path(output_dir) / 'reports'
    parsed = []
    parse_errors = []
    candidate_paths = sorted(reports_root.rglob('*.json')) if reports_root.exists() else []
    for path in candidate_paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            parse_errors.append((str(path), str(exc)))
            continue
        if _looks_like_report(data):
            parsed.append((path, data))

    accepted_names = {expected_dataset}
    if expected_pretty_name:
        accepted_names.add(expected_pretty_name)
    matched = [
        (path, data) for path, data in parsed
        if data.get('dataset_name') in accepted_names and data.get('model_name') == expected_model_id
    ]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise ReportError(
            'ambiguous', f'multiple reports match {expected_dataset!r}/{expected_model_id!r}',
            {'paths': [str(path) for path, _ in matched]},
        )
    if parsed:
        raise ReportError(
            'mismatch',
            f'no report matches dataset={expected_dataset!r} model={expected_model_id!r}',
            {'found': [{'path': str(path), 'dataset_name': data.get('dataset_name'),
                        'model_name': data.get('model_name')} for path, data in parsed]},
        )
    if parse_errors:
        raise ReportError('parse_error', 'report JSON could not be parsed',
                          {'errors': [{'path': p, 'error': e} for p, e in parse_errors]})
    raise ReportError('missing', 'no report file was produced for this attempt')


def _numeric(value, where: str):
    if value is None:
        return None
    if isinstance(value, bool):
        raise MetricParseError(f'{where}: boolean is not a score')
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            raise MetricParseError(f'{where}: non-finite score {value!r}')
        return result
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError as exc:
            raise MetricParseError(f'{where}: non-numeric score {value!r}') from exc
        if not math.isfinite(result):
            raise MetricParseError(f'{where}: non-finite score {value!r}')
        return result
    raise MetricParseError(f'{where}: unsupported score type {type(value).__name__}')


def metric_identity(identity: dict):
    identity = identity or {}
    payload = {
        'name': identity.get('name'),
        'aggregation': identity.get('aggregation'),
        'dimensions': identity.get('dimensions') or {},
    }
    return digest(payload), payload


def primary_metric_key(report: dict) -> str | None:
    identity = report.get('primary_metric_identity')
    if not identity:
        return None
    return metric_identity(identity)[0]


def metric_rows(report: dict) -> list[dict]:
    """Convert a report into full-identity metric rows.  Missing scores stay None."""
    primary_key = primary_metric_key(report)
    rows = []
    for index, metric in enumerate(report.get('metrics') or []):
        where = f'metrics[{index}]'
        key, identity = metric_identity(metric.get('identity') or {})
        semantics = metric.get('semantics') or {}
        base = {
            'metric_key': key,
            'metric_name': identity['name'] or metric.get('legacy_name') or 'metric',
            'aggregation': identity['aggregation'],
            'dimensions': identity['dimensions'],
            'semantics_kind': semantics.get('kind'),
            'display_kind': semantics.get('display_kind'),
            'direction': semantics.get('direction'),
            'unit': semantics.get('display_unit') or semantics.get('raw_unit'),
            'is_primary': key == primary_key,
            'score': _numeric(metric.get('score'), f'{where}.score'),
            'macro_score': _numeric(metric.get('macro_score'), f'{where}.macro_score'),
            'num': metric.get('num'),
        }
        rows.append({**base, 'category': [], 'category_key': digest([]), 'subset': ''})
        for cat_index, category in enumerate(metric.get('categories') or []):
            category_path = list(category.get('name') or [])
            category_key = digest(category_path)
            cat_where = f'{where}.categories[{cat_index}]'
            rows.append({
                **base,
                'category': category_path,
                'category_key': category_key,
                'subset': '',
                'score': _numeric(category.get('score'), f'{cat_where}.score'),
                'macro_score': _numeric(category.get('macro_score'), f'{cat_where}.macro_score'),
                'num': category.get('num'),
            })
            for subset in category.get('subsets') or []:
                rows.append({
                    **base,
                    'category': category_path,
                    'category_key': category_key,
                    'subset': subset.get('name') or '',
                    'score': _numeric(subset.get('score'), f'{cat_where}.subsets.score'),
                    'macro_score': None,
                    'num': subset.get('num'),
                })
    return rows


# ---------------------------------------------------------------------------
# prediction/review coverage and diagnostics
# ---------------------------------------------------------------------------

def _subset_from_stem(stem: str, dataset: str) -> str:
    prefix = f'{dataset}_'
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _sample_id(line: dict):
    sample_id = line.get('sample_id', line.get('index'))
    if sample_id is None:
        score = line.get('sample_score')
        if isinstance(score, dict):
            sample_id = score.get('sample_id')
    return None if sample_id is None else str(sample_id)


def _repeat_id(line: dict):
    repeat = line.get('repeat_id', line.get('repeat'))
    try:
        return int(repeat)
    except (TypeError, ValueError):
        return 0


def _evidence_key(line: dict):
    sample_id = _sample_id(line)
    if sample_id is None:
        return None
    return f'{sample_id}#r{_repeat_id(line)}'


def _generation_diagnostics(line: dict):
    """Return (stop_reasons, usage, agent_steps) for one prediction line."""
    reasons = []
    usage = None
    steps = 0
    output = line.get('model_output')
    if isinstance(output, dict):
        for choice in output.get('choices') or []:
            reasons.append(choice.get('stop_reason') or 'unknown')
        usage = output.get('usage') or None
    trace = line.get('agent_trace')
    if isinstance(trace, dict):
        events = [event for event in (trace.get('events') or [])
                  if event.get('type') == 'model_generate']
        if events:
            steps = len(events)
            reasons = [
                ((event.get('payload') or {}).get('stop_reason')) or 'unknown'
                for event in events
            ]
            if trace.get('total_usage'):
                usage = trace['total_usage']
    if not reasons:
        reasons = ['unknown']
    return reasons, usage, steps


def collect_output_evidence(output_dir: Path, dataset: str, *, report_id: str | None = None) -> dict:
    """Collect per-subset coverage and diagnostics from this attempt's own files.

    When ``report_id`` is given, only ``predictions/<report_id>`` and
    ``reviews/<report_id>`` are read: files of another model are not this run's
    evidence.  Sample IDs keep their repeat id, so legitimate repeats are not
    mistaken for duplicates.
    """
    per_subset = defaultdict(lambda: {
        'predicted_ids': [], 'reviewed_ids': [],
        'predicted_sources': [], 'reviewed_sources': [],
        'stop_reasons': Counter(), 'usage': defaultdict(float), 'usage_samples': 0,
        'agent_steps': [], 'missing_stop_reason': 0, 'missing_usage': 0,
    })
    malformed = []
    for folder, key in (('predictions', 'predicted'), ('reviews', 'reviewed')):
        root = Path(output_dir) / folder
        if report_id is not None:
            root = root / report_id
        if not root.exists():
            continue
        paths = sorted(root.glob('*.jsonl')) if report_id is not None else sorted(root.rglob('*.jsonl'))
        for path in paths:
            subset = _subset_from_stem(path.stem, dataset)
            entry = per_subset[subset]
            entry[f'{key}_sources'].append(str(path))

            def on_error(line_number, exc, path=path):
                malformed.append({'path': str(path), 'line': line_number, 'error': str(exc)})

            for line in read_jsonl(path, on_error=on_error):
                sample_id = _evidence_key(line)
                if sample_id is None:
                    malformed.append({'path': str(path), 'line': None, 'error': 'no sample id'})
                    continue
                entry[f'{key}_ids'].append(sample_id)
                if key == 'predicted':
                    reasons, usage, steps = _generation_diagnostics(line)
                    has_stop = any(reason != 'unknown' for reason in reasons)
                    if not has_stop:
                        entry['missing_stop_reason'] += 1
                    for reason in reasons:
                        entry['stop_reasons'][reason] += 1
                    if steps:
                        entry['agent_steps'].append(steps)
                    if isinstance(usage, dict) and usage:
                        for name in ('input_tokens', 'output_tokens', 'total_tokens', 'input', 'output', 'total'):
                            value = usage.get(name)
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                canonical = {
                                    'input': 'input_tokens', 'output': 'output_tokens',
                                    'total': 'total_tokens',
                                }.get(name, name)
                                entry['usage'][canonical] += value
                        entry['usage_samples'] += 1
                    else:
                        entry['missing_usage'] += 1

    result = {'subsets': {}, 'malformed_lines': malformed, 'report_id': report_id}
    for subset, entry in sorted(per_subset.items()):
        result['subsets'][subset] = {
            'predicted': entry['predicted_ids'],
            'reviewed': entry['reviewed_ids'],
            'predicted_sources': entry['predicted_sources'],
            'reviewed_sources': entry['reviewed_sources'],
            'stop_reasons': dict(entry['stop_reasons']),
            'usage': dict(entry['usage']),
            'usage_samples': entry['usage_samples'],
            'agent_steps': entry['agent_steps'],
            'missing_stop_reason': entry['missing_stop_reason'],
            'missing_usage': entry['missing_usage'],
        }
    return result


def _coverage_diff(expected: list, observed: list) -> dict:
    """Compare expected sample IDs with observed evidence keys.

    Observed keys carry a repeat id (``<sample>#r<n>``); multiple legitimate
    repeats count as coverage, while the same (sample, repeat) key twice counts
    as a duplicate.
    """
    expected_set = set(expected)
    observed_bases = {key.rsplit('#r', 1)[0] for key in observed}
    if expected_set:
        missing = sorted(expected_set - observed_bases)
        extra = sorted(observed_bases - expected_set)
        covered = len(expected_set & observed_bases)
    else:
        missing, extra, covered = [], sorted(observed_bases), len(observed_bases)
    occurrences = Counter(observed)
    return {
        'covered': covered,
        'total': len(observed),
        'missing': missing,
        'extra': extra,
        'duplicates': sum(count - 1 for count in occurrences.values() if count > 1),
    }


def apply_coverage(manifest_rows: list[dict], output_evidence: dict) -> list[dict]:
    """Merge coverage into sample_manifest rows, keeping the ID-set differences."""
    subsets = (output_evidence or {}).get('subsets') or {}
    merged = []
    for row in manifest_rows:
        subset = row['subset']
        coverage = subsets.get(subset, {})
        expected = [str(item) for item in (row.get('sample_ids') or [])]
        predicted = _coverage_diff(expected, [str(item) for item in coverage.get('predicted') or []])
        reviewed = _coverage_diff(expected, [str(item) for item in coverage.get('reviewed') or []])
        merged.append({
            **row,
            'predicted': predicted['covered'],
            'predicted_total': predicted['total'],
            'predicted_missing': predicted['missing'],
            'predicted_extra': predicted['extra'],
            'predicted_duplicates': predicted['duplicates'],
            'reviewed': reviewed['covered'],
            'reviewed_total': reviewed['total'],
            'reviewed_missing': reviewed['missing'],
            'reviewed_extra': reviewed['extra'],
            'reviewed_duplicates': reviewed['duplicates'],
        })
    return merged


def summarize_diagnostics(output_evidence: dict) -> dict:
    stop_reasons = Counter()
    total_generations = 0
    samples_with_truncation = 0
    missing_stop_reason = 0
    missing_usage = 0
    usage_totals = defaultdict(float)
    agent_steps = []
    subsets = output_evidence.get('subsets') or {}
    for coverage in subsets.values():
        reasons = coverage.get('stop_reasons') or {}
        stop_reasons.update(reasons)
        generations = sum(reasons.values())
        total_generations += generations
        truncating = reasons.get('max_tokens', 0) + reasons.get('model_length', 0)
        if truncating:
            samples_with_truncation += 1
        missing_stop_reason += coverage.get('missing_stop_reason') or 0
        missing_usage += coverage.get('missing_usage') or 0
        for name, value in (coverage.get('usage') or {}).items():
            usage_totals[name] += value
        agent_steps.extend(coverage.get('agent_steps') or [])
    diagnostics = {
        'generation_calls': total_generations,
        'stop_reason_counts': dict(stop_reasons),
        'truncated_calls': stop_reasons.get('max_tokens', 0) + stop_reasons.get('model_length', 0),
        'content_filter_calls': stop_reasons.get('content_filter', 0),
        'unknown_stop_reason_calls': stop_reasons.get('unknown', 0),
        'samples_without_stop_reason': missing_stop_reason,
        'samples_without_usage': missing_usage,
        'usage_totals': {k: v for k, v in usage_totals.items()},
        'malformed_lines': output_evidence.get('malformed_lines') or [],
    }
    if agent_steps:
        diagnostics['agent_steps'] = {
            'count': len(agent_steps),
            'total': sum(agent_steps),
            'max': max(agent_steps),
            'mean': round(sum(agent_steps) / len(agent_steps), 2),
        }
    return diagnostics
