"""Scoped compatibility patches, with an explicit status record.

Only the OCRBench-v2 scorer's ``nltk.edit_distance`` call is redirected: nltk >= 3.9
refuses inputs longer than 2000 chars and its pure-Python implementation cannot score
full-page OCR.  ``Levenshtein.distance`` computes the same plain edit distance with
``substitution_cost=1`` and no transpositions, in C.

The patch is applied inside a context manager and restored on exit, so it never
leaks into unrelated benchmarks or post-run tooling.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager


@contextmanager
def ocr_compat_patch():
    """Yield a status dict; ``nltk.edit_distance`` is patched for the block only."""
    status = {'applied': False, 'reason': '', 'target': 'nltk.edit_distance'}
    original = None
    try:
        import Levenshtein
        import nltk
    except Exception as exc:
        status['reason'] = f'missing dependency ({exc.__class__.__name__}: {exc})'
        yield status
        return
    try:
        original = nltk.edit_distance
        nltk.edit_distance = Levenshtein.distance
        status['applied'] = True
        status['reason'] = 'nltk.edit_distance -> Levenshtein.distance (C implementation)'
    except Exception as exc:  # pragma: no cover - only if nltk internals change
        status['reason'] = f'patch failed: {exc}'
    try:
        yield status
    finally:
        if original is not None:
            nltk.edit_distance = original


def ocr_compat_available() -> tuple[bool, str]:
    try:
        import Levenshtein  # noqa: F401
        import nltk  # noqa: F401
    except Exception as exc:
        return False, f'missing dependency: {exc}'
    return True, 'nltk and python-Levenshtein importable'


def report_patch_status(status: dict) -> None:
    if not status.get('applied'):
        print(f'ocr compat patch not applied: {status.get("reason")}', file=sys.stderr)
