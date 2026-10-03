# LLMBenchmark

Configurable benchmarking of models behind any **OpenAI-compatible API** through
[EvalScope](https://evalscope.readthedocs.io).  Three suites are covered:

- **knowledge** — knowledge / reasoning: GPQA-Diamond, MATH-500, IFEval, …
- **swe** — software engineering: SWE-bench Verified (oracle + agentic), LiveCodeBench, …
- **vision** — vision understanding / OCR: OCRBench-v2, DocVQA, MathVista, …

Every dataset is one isolated EvalScope run.  Raw outputs are kept, and every
attempt is recorded in a SQLite database from which `results/summary.md` is
generated.  A run is only considered a result when its report can be attributed
to the exact dataset/model attempt and the content evidence is complete — see
[Success contract](#success-contract).

## Layout

```
bench.py                    thin CLI entry point (run / prepare / report / import)
llmbench/                   implementation package
configs/models.yaml         model registry (API URL + api_key env var name)
configs/suites.yaml         suites, datasets, profiles, per-dataset caps, status
results/summary.md          generated from the DB (committed)
tests/                      offline test suite (pytest)
$DATA_ROOT/                 default /ssd4/LLMBenchmark
├── datasets/               EvalScope dataset cache
├── pinned/                 pinned local datasets (SWE-bench Django)
├── evalscope-cache/        EvalScope cache
├── hf-cache/               HuggingFace cache
├── outputs/<group>/<dataset>/<attempt>/   raw evidence per attempt
└── results.db              SQLite schema v2 (runs / metrics / sample_manifest / meta)
```

Set `LLMBENCH_DATA_ROOT` (or `--data-root`) to move the data root.

## Setup

Python **3.12** is what this repository is tested with (EvalScope 1.12 requires
Python ≥ 3.10).  Dependencies are layered so the offline CLI/tests do not pull
in the whole evaluation stack:

| File | Contents |
|---|---|
| `requirements.txt` | core CLI: PyYAML |
| `requirements-extras.txt` | EvalScope 1.12 + suite extras + `python-Levenshtein` |
| `requirements-test.txt` | core + pytest (CI runs exactly this) |
| `requirements-lock.txt` | exact `pip freeze` snapshot of the environment used for recorded runs |

```bash
python -m pip install -r requirements-extras.txt   # to run benchmarks
python -m pip install -r requirements-test.txt     # to run the offline tests
```

Network is usually reachable directly; add
`http_proxy=http://127.0.0.1:18080 https_proxy=http://127.0.0.1:18080`
if a download stalls.  Docker pulls already use the daemon proxy configured in
`/etc/docker/daemon.json`.

## Usage

```bash
# see the effective plan (per-subset limits, pinned data, required modules) without
# touching keys, Docker, downloads or the database
python bench.py run --dry-run --model minimax --suite all --profile smoke

# connectivity / quick slice
python bench.py run --model minimax --suite knowledge --profile smoke

# comparable slice across models (deterministic first N per dataset)
python bench.py run --model minimax --suite all --profile lite

# formal run: whole dataset, capped per dataset by full_per_subset_limit
python bench.py run --model minimax --suite all --profile full

# one-time: build the pinned single Django instance for agentic SWE
python bench.py prepare

# regenerate results/summary.md from the DB
python bench.py report

# re-parse an attempt directory without calling the model (repairs/backfills)
python bench.py import --output-dir $DATA_ROOT/outputs/<group>/<dataset>/<attempt>
```

Useful flags: `--datasets <names…>` (explicit selection **overrides** the
`core` filter), `--limit N` (overrides the profile), `--batch-size N`,
`--model all`, `--include-extensions` (adds non-core datasets),
`--allow-partial`, `--cleanup-images` (removes only images this run provably
owns and that are not in use — never a global prune), `--dry-run`.

`import`, `report --allow-partial` etc. never call a model.

## Profiles and limits

| Profile | Samples per dataset | Purpose |
|---|---|---|
| `smoke` | 5 per subset | API/env connectivity; diagnostics only |
| `lite` (default) | 20 per subset | cheap cross-model comparison |
| `full` | whole dataset, capped by `full_per_subset_limit` | formal runs |

EvalScope applies `limit` **per subset** (and per `full_per_subset_limit` when
set per dataset), so a dataset with several subtasks runs `limit × subsets`
questions.  The effective number is printed by `--dry-run` and stored in the
sample manifest.  Sampling is EvalScope's deterministic first N (no shuffle);
pinned datasets use `limit=1`.

## Success contract

"EvalScope did not raise" is **not** success.  Two independent axes are recorded
per attempt:

- `execution_status` — did the attempt itself run: `completed`, `failed`
  (task exception), `interrupted` (Ctrl+C), `skipped`.
- `validity_status` — is the result usable:
  - `complete` — report found, owned by this dataset + model report id,
    schema parsed, quality metric present, completion verified.
  - `partial` — some samples errored; shown in the formal table only with
    `--allow-partial`, and then explicitly marked.
  - `unverified` — evidence insufficient to certify completion (e.g. missing
    execution summary, requested > manifest selected).
  - `invalid` — missing/ambiguous/foreign/corrupt report, no quality metric,
    all samples failed, interrupted or failed run.
  - `legacy` — migrated v1 rows with no identity evidence; diagnostics only.

A real score of `0` from a complete run is valid and is recorded as `0`, not as
"missing" (the DB distinguishes `NULL` from `0`).  The CLI exit code is `0`
only when every selected attempt is `complete` (or `partial` with
`--allow-partial`), `1` when some attempt did not meet that contract, `2` for
configuration/preflight errors, `130` when interrupted by Ctrl+C.

## Comparability and evidence

Formal results are grouped by two identity hashes, so different evaluation
conditions are never silently mixed:

- **protocol identity** — effective EvalScope task config (generation config,
  agent config, dataset behavior knobs) with model, endpoint, sample count and
  local data path removed.
- **sample manifest identity** — per subset: selected/predicted/reviewed counts,
  sample IDs, question digest, media digest, request-input digest.
- **model config identity** — model id + normalized endpoint + deployment
  version (recorded as `unknown` when the provider exposes none).

Before inference, the actual loaded samples are fingerprinted: question content
(digest), media content (digest of base64 image bytes), and the fully rendered
request input (digest).  URL-only media is marked as *not verifiable* in the
manifest instead of being treated as content evidence.  Smoke runs, partial
runs, unverified runs and legacy rows appear in the diagnostics section of
`results/summary.md`, never in the formal table.

Per-attempt diagnostics also extract, from the predictions themselves:
`stop_reason` distribution (truncation, content filter, unknown/missing stop
reason), missing usage counters and agent step counts.  Intermediate agent
`tool_calls` are normal and are not counted as truncation.

## Recording and re-import

- `runs` — one row per attempt: status, phase, counts, identities, EvalScope
  version, tool version, git commit, output dir, redacted effective config,
  report path, error, primary metric identity.
- `metrics` — full metric identity (`name` + `aggregation` + `dimensions`,
  category path and subset) plus semantics (`kind`, `display_kind`, `direction`,
  `unit`) and `is_primary`; `NULL` score is stored as `NULL`.
- `sample_manifest` — per subset selected/predicted/reviewed, sample ids and
  content digests.

Attempt IDs are `<timestamp>_<uuid8>`; each attempt gets its own output
directory, and `INSERT` never replaces an existing attempt.  Writes are short
transactions (`WAL` + busy timeout); `finish_attempt` replaces metrics and the
manifest atomically.  `import` is idempotent and, if re-parsing fails, leaves
the previous scores untouched.  Schema v2 migrates v1 databases in place:
legacy rows are preserved and marked `legacy`/`unverified`.

Each attempt directory keeps the evidence needed to audit or re-import it:
`run_manifest.json` (redacted plan + config + identities),
`run_outcome.json` (validated outcome + diagnostics), and the raw EvalScope
`reports/`, `predictions/`, `reviews/`, `logs/`.

## Safety

- Secrets are only read from the environment variable named in
  `configs/models.yaml`; `api_key`/token-like keys are recursively redacted in
  manifests, the DB, summaries and exception text (including URL credentials).
- Sandboxed benchmarks refuse to run when Docker is unavailable; there is **no
  fallback to host execution**.
- `--cleanup-images` deletes only images recorded in this run's own predictions
  (`swebench/`, `sweb.eval*`, `sweap*` prefixes) that no container is using.
- `--dry-run` has zero side effects: no key access, no database, no Docker, no
  downloads.
- `prepare` writes to a staging directory and atomically replaces the pinned
  dataset; the pinned revision (40-hex) and parquet hash are recorded.

## Testing

```bash
python -m pytest tests/ -q          # offline: no API, Docker or EvalScope needed
```

CI (`.github/workflows/tests.yml`) runs exactly this suite after installing
`requirements-test.txt`.  Tests that need nltk/Levenshtein skip when those
extras are absent.  The suite covers: metric identity and `NULL` vs `0`,
report ownership/ambiguity/parse errors, sample manifests and media evidence,
completion/partial/unverified rules, summary grouping and smoke handling,
attempt lifecycle and error boundaries (mocked EvalScope), idempotent import,
v1→v2 migration, image ownership, the scoped OCR patch semantics, and
`prepare` atomic replacement.

## Dataset status

`configs/suites.yaml` records a `status` per dataset: `tested` means it has
been executed end-to-end against a real endpoint at least once; `configured`
means the config exists but the dataset has not been exercised here;
`requires-extra-deps` lists Python modules that are not installed by
`requirements-extras.txt`.  `--dry-run` prints `requires`/`requires_missing`
per entry, and preflight fails before spending money when requirements are not
met.

## Adding a model or dataset

- **Model**: append an entry to `configs/models.yaml`.  Never put the key in
  the file; name the environment variable in `api_key_env`.  Add
  `deployment_version` when the provider exposes a stable one.
- **Dataset**: append it under the right suite in `configs/suites.yaml`.  Set
  `core: true` for defaults, `full_per_subset_limit` for expensive full runs,
  `requires`/`requires_sandbox` for preflight checks, and use pass-through
  `dataset_args` / `extra_params` / `generation_config` / `agent_config` for
  EvalScope-specific options.
