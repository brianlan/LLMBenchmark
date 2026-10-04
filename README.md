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
  (task exception, evidence-write error, runner error), `interrupted`
  (Ctrl+C), `skipped`.
- `validity_status` — is the result usable:
  - `complete` — report found, owned by this dataset + model report id,
    schema parsed, the benchmark's **primary quality metric** is identifiable
    and carries a usable numeric score, completion verified and sample
    evidence ID-matched.
  - `partial` — some samples errored; shown in the formal table only with
    `--allow-partial`, and then explicitly marked.
  - `unverified` — evidence insufficient to certify completion or the primary
    score (missing execution summary, non-integer count fields, requested >
    manifest selected, missing primary metric identity, NULL primary score,
    aggregate count above succeeded).
  - `invalid` — missing/ambiguous/foreign/corrupt report, no quality metric,
    non-quality or malformed primary identity (including an identity that only
    has a display fallback name), all samples failed, interrupted or failed run.
  - `legacy` — migrated v1 rows with no identity evidence; diagnostics only.

A real score of `0` from a complete run is valid and is recorded as `0`, not as
"missing" (the DB distinguishes `NULL` from `0`); a `NULL` primary score is
`unverified`, never a formal result.  The CLI exit code is `0` only when every
selected attempt is `complete` (or `partial` with `--allow-partial`), `1` when
some attempt did not meet that contract (including a failed summary write),
`2` for configuration/preflight errors, `130` when interrupted by Ctrl+C.

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
manifest instead of being treated as content evidence.  After inference,
predictions/reviews are read **only from this attempt's own model directory**
(`predictions/<report_id>`, `reviews/<report_id>`).  Prediction keys are
matched to the manifest's selected sample IDs, and review keys are matched to
the **prediction keys** including `repeat_id`, so the score for repeat 1 can
never certify repeat 0.  Missing, unexpected, duplicate or unparsable evidence
blocks `verified` and is recorded per subset in the DB.  Smoke runs, partial
runs, unverified runs and legacy rows appear in the diagnostics section of
`results/summary.md`, never in the formal table.

Every external surface (DB, `run_outcome.json`, terminal, summary) is rendered
from the same fully redacted result object, so exception text, report parse
errors and diagnostics errors cannot leak credentials through one path only.
Provider key patterns (`sk-…`, JWTs) are masked even when the secret value is
not known to the current process.

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
manifest atomically.  Every attempt has a complete failure boundary: manifest
writes, inference, report parsing, diagnostics, evidence writes and DB
persistence all terminate the attempt (`failed`/`invalid`, with the failing
phase recorded).  A failed `run_outcome.json` write does not abort the batch or
leave a `running` row; a failed summary write is reported and turns the batch
exit code into `1`.

`import` is idempotent, never rewrites the original evaluation `finished_at`
(only `imported_at` is updated) and **rejects** a re-parse that would downgrade
an already accepted run.  "Downgrade" is evaluated with the same shared rule
as the summary: an existing `complete`+`verified` result stays in place if the
new parse is not `complete`+`verified` (including `verified → unknown`), the
previous metrics are untouched, and the rejection is audited in
`run_import_rejected.json`.  The evidence file is written before the database
commit; if either step fails the CLI reports exactly which one was updated.
Schema v3 adds empty migration columns for the extra coverage evidence.

Each attempt directory keeps the evidence needed to audit or re-import it:
`run_manifest.json` (redacted plan + config + identities),
`run_outcome.json` (validated outcome + diagnostics), and the raw EvalScope
`reports/`, `predictions/`, `reviews/`, `logs/`.

## Safety

- Secrets are only read from the environment variable named in
  `configs/models.yaml`; `api_key`/token-like keys are recursively redacted in
  manifests, the DB, summaries, terminal output and exception text (including
  URL credentials and provider key patterns).
- Sandboxed benchmarks refuse to run when Docker is unavailable; there is **no
  fallback to host execution**.
- `--cleanup-images` deletes only images that this run provably created: they
  must be referenced by this run's own predictions, unused by any container,
  absent from the pre-run image inventory recorded in the attempt manifest
  (when the inventory could not be taken, nothing is removed), and their
  Docker creation time must not be older than the attempt that referenced
  them (`docker image inspect`).  The CLI passes each attempt's start time and
  inventory to the cleanup; anything unprovable is kept and failed removals
  are reported.
- `--dry-run` has zero side effects: no key access, no database, no Docker, no
  downloads.
- `prepare` writes to a staging directory and atomically replaces the pinned
  dataset; the pinned revision (40-hex, with its source recorded) and parquet
  hash are stored.  `pinned_instance_id` in `suites.yaml` is required: preflight
  and `prepare` verify the hash and read the parquet back to confirm one
  readable sample with exactly that instance id.

## Testing

```bash
python -m pytest tests/ -q          # offline: no API, Docker or EvalScope needed
```

CI (`.github/workflows/tests.yml`) runs exactly this suite after installing
`requirements-test.txt`.  Tests that need nltk/Levenshtein or pyarrow skip when
those extras are absent.  The suite covers: metric identity and `NULL` vs `0`,
report ownership/ambiguity/parse errors, primary-metric validity, sample
manifests and media evidence, ID-set coverage (missing/extra/duplicate/
malformed, other-model files), completion/partial/unverified rules, summary
grouping and smoke handling, attempt lifecycle and error boundaries (mocked
EvalScope, including manifest/outcome write failures), secret-sentinel
redaction through DB/stdout/outcome/summary, idempotent and rejecting import
with stable evaluation times, v1→v2→v3 migration, recursive config merging,
protocol storage invariance, image ownership proof, the scoped OCR patch
semantics, and `prepare` hash/content verification.

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
  EvalScope-specific options.  Nested dicts are merged recursively (model
  defaults are preserved when a dataset overrides one nested field); lists and
  scalars replace the base value.
