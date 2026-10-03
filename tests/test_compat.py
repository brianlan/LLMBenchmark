import pytest

nltk = pytest.importorskip('nltk')
Levenshtein = pytest.importorskip('Levenshtein')

from llmbench.compat import apply_ocr_compat, ocr_compat_available  # noqa: E402


def reference_distance(a: str, b: str) -> int:
    """Plain Levenshtein distance: substitution cost 1, no transpositions."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def test_apply_patch_targets_only_nltk_edit_distance():
    import importlib

    metrics_distance_module = importlib.import_module('nltk.metrics.distance')
    original = metrics_distance_module.edit_distance
    status = apply_ocr_compat()
    assert status['applied'] is True
    assert status['target'] == 'nltk.edit_distance'
    assert nltk.edit_distance is Levenshtein.distance
    # the underlying nltk implementation is untouched, the patch is call-site scoped
    assert metrics_distance_module.edit_distance is original


def test_patched_function_matches_reference_semantics():
    apply_ocr_compat()
    cases = [
        ('kitten', 'sitting'),
        ('', 'abc'),
        ('abc', ''),
        ('你好世界', '你好世'),
        ('café', 'cafe'),
        ('a' * 2500, 'a' * 2499 + 'b'),  # nltk >= 3.9 refuses >2000 chars
    ]
    for left, right in cases:
        assert nltk.edit_distance(left, right) == reference_distance(left, right)


def test_availability_probe_reports_importable_dependencies():
    available, reason = ocr_compat_available()
    assert available is True
    assert 'importable' in reason
