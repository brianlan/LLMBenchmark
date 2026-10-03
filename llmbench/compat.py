"""Scoped compatibility patches, with an explicit status record.

Only the OCRBench-v2 scorer's ``nltk.edit_distance`` call is redirected: nltk >= 3.9
refuses inputs longer than 2000 chars and its pure-Python implementation cannot score
full-page OCR.  ``Levenshtein.distance`` computes the same plain edit distance with
``substitution_cost=1`` and no transpositions, in C.
"""

from __future__ import annotations

import sys


def apply_ocr_compat() -> dict:
    status = {'applied': False, 'reason': '', 'target': 'nltk.edit_distance'}
    try:
        import Levenshtein
        import nltk
    except Exception as exc:
        status['reason'] = f'missing dependency ({exc.__class__.__name__}: {exc})'
        return status
    try:
        nltk.edit_distance = Levenshtein.distance
        status['applied'] = True
        status['reason'] = 'nltk.edit_distance -> Levenshtein.distance (C implementation)'
    except Exception as exc:  # pragma: no cover - only if nltk internals change
        status['reason'] = f'patch failed: {exc}'
    return status


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
