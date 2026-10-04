import importlib

import pytest

nltk = pytest.importorskip('nltk')
Levenshtein = pytest.importorskip('Levenshtein')

from llmbench.compat import ocr_compat_available, ocr_compat_patch  # noqa: E402


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


def test_patch_is_scoped_and_restored():
    metrics_distance_module = importlib.import_module('nltk.metrics.distance')
    original_top_level = nltk.edit_distance
    original_impl = metrics_distance_module.edit_distance

    with ocr_compat_patch() as status:
        assert status['applied'] is True
        assert status['target'] == 'nltk.edit_distance'
        assert nltk.edit_distance is Levenshtein.distance
        # the underlying nltk implementation is untouched, the patch is call-site scoped
        assert metrics_distance_module.edit_distance is original_impl

    # the global patch does not leak past the context
    assert nltk.edit_distance is original_top_level
    assert metrics_distance_module.edit_distance is original_impl


def test_patched_function_matches_reference_semantics():
    cases = [
        ('kitten', 'sitting'),
        ('', 'abc'),
        ('abc', ''),
        ('你好世界', '你好世'),
        ('café', 'cafe'),
        ('a' * 2500, 'a' * 2499 + 'b'),  # nltk >= 3.9 refuses >2000 chars
    ]
    with ocr_compat_patch():
        for left, right in cases:
            assert nltk.edit_distance(left, right) == reference_distance(left, right)


def test_availability_probe_reports_importable_dependencies():
    available, reason = ocr_compat_available()
    assert available is True
    assert 'importable' in reason
