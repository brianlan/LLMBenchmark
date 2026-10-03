# LLMBenchmark

Configurable benchmarking of models behind any **OpenAI-compatible API** through
[EvalScope](https://evalscope.readthedocs.io).  Three suites are covered:

- **knowledge** — knowledge / reasoning: GPQA-Diamond, MATH-500, IFEval, …
- **swe** — software engineering: SWE-bench Verified (oracle + agentic), LiveCodeBench, …
- **vision** — vision understanding / OCR: OCRBench-v2, DocVQA, MathVista, …

Every dataset is one isolated EvalScope run.  Raw outputs (predictions, reviews,
logs, reports) are kept, and every run is recorded in a SQLite database that
`results/summary.md` is generated from.

## Layout

```
bench.py                  CLI (run / prepare / report)
configs/models.yaml       model registry (API URL + api_key env var name)
configs/suites.yaml       suites, datasets, profiles, per-dataset caps
results/summary.md        generated from the DB (committed)
tests/test_bench.py       self-checks (python tests/test_bench.py)
$DATA_ROOT/               default /ssd4/LLMBenchmark
├── datasets/             EvalScope dataset cache
├── pinned/               pinned local datasets (SWE-bench Django)
├── evalscope-cache/      EvalScope cache
├── hf-cache/             HuggingFace cache
├── outputs/<group>/<dataset>/   raw EvalScope output per run
└── results.db            SQLite (runs / metrics / run_samples)
```

Set `LLMBENCH_DATA_ROOT` (or `--data-root`) to move the data root.

## Setup

Use the existing environment:

```bash
/ssd4/envs/aco_py312/bin/pip install -r requirements.txt
```

Network is usually reachable directly; add
`http_proxy=http://127.0.0.1:18080 https_proxy=http://127.0.0.1:18080`
if a download stalls.  Docker pulls already use the daemon proxy configured in
`/etc/docker/daemon.json`.

## Usage

```bash
# connectivity / quick slice
python bench.py run --model minimax --suite knowledge --profile smoke
python bench.py run --model minimax --suite all      --profile smoke

# comparable slice across models (deterministic first N per dataset)
python bench.py run --model minimax --suite all --profile lite

# one-time: build the pinned single Django instance for agentic SWE
python bench.py prepare

# regenerate results/summary.md from the DB
python bench.py report

# see the effective EvalScope config without calling the model
python bench.py run --model minimax --suite vision --profile smoke --dry-run
```

Useful flags: `--datasets <names…>`, `--limit N`, `--batch-size N`,
`--model all`, `--include-extensions` (run non-core datasets),
`--cleanup-images` (remove only `swebench/`, `sweb.eval*`, `sweap*` images —
never a global prune).

## Profiles and comparability

| Profile | Samples per dataset | Purpose |
|---|---|---|
| `smoke` | 5 per subset | API/env connectivity |
| `lite` (default) | 20 per subset | cheap cross-model comparison |
| `full` | whole dataset, capped by `full_max_samples` | formal runs |

EvalScope applies `limit` per **subset**, so a dataset with several subtasks
(MATH-500 has 5) runs `limit × subsets` questions.

Sampling is EvalScope's deterministic first N (no shuffle), and the sample IDs
actually used are stored in `run_samples`, so two runs can be checked for
identical question sets.  The effective config of every run is stored in
`runs.config_json`.  SWE agentic is pinned to one Django instance
(`django__django-10097`) built by `python bench.py prepare` — run that once
before the first agentic run.

## Recording

- `runs` — one row per dataset run: status, counts, EvalScope version, git
  commit, output dir, effective config, perf metrics.
- `metrics` — per metric/category/subset: `num`, `score`, `macro_score`.
- `run_samples` — per-subset sample IDs used (JSON: `{subset: [ids]}`).

Per-sample predictions and scores stay in `outputs/.../predictions|reviews`.
`results/summary.md` shows the latest successful score per model × dataset ×
metric plus any non-successful runs.

## Adding a model or dataset

- **Model**: append an entry to `configs/models.yaml`.  Never put the key in
  the file; name the environment variable in `api_key_env`.
- **Dataset**: append it under the right suite in `configs/suites.yaml`.  Set
  `core: true` for defaults, `full_max_samples` for expensive full runs, and
  use pass-through `dataset_args` / `extra_params` / `generation_config` /
  `agent_config` for EvalScope-specific options.
