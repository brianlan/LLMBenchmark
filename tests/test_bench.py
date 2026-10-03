"""The bench.py entry point is a thin shim over llmbench.cli."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bench  # noqa: E402
import llmbench.cli as cli  # noqa: E402


def test_bench_delegates_to_cli():
    assert bench.main is cli.main
    assert callable(bench.main)
