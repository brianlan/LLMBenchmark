#!/usr/bin/env python3
"""LLMBenchmark: run EvalScope benchmarks through OpenAI-compatible APIs.

Usage:
    python bench.py run --model minimax --suite knowledge --profile smoke
    python bench.py run --model minimax --suite all --profile lite [--dry-run]
    python bench.py prepare                     # pinned SWE-bench Django dataset
    python bench.py report                      # results/summary.md from the DB
    python bench.py import --output-dir <dir>   # re-parse an attempt, no model calls

Every dataset is one attempt with a unique output directory.  Raw outputs stay
under ``$LLMBENCH_DATA_ROOT/outputs``; validated records go to
``$LLMBENCH_DATA_ROOT/results.db`` and ``results/summary.md``.
"""

import sys

from llmbench.cli import main

if __name__ == '__main__':
    sys.exit(main())
