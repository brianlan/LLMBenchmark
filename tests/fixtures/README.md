# Test fixtures provenance

These files are verbatim copies of artifacts from a **real** EvalScope run of the
MiniMax endpoint, kept so the parsers can be tested against the real report
schema without spending API credit.

- Source run: `20261003_145714_minimax_knowledge_smoke` (recorded in the v1 DB,
  later migrated to `legacy`).
- Command: `python bench.py run --model minimax --suite knowledge --profile smoke --datasets gpqa_diamond`
- Model: `MiniMax-M3.1-Flash-Preview` (report id `MiniMax-M3.1-Flash-Preview`)
- Dataset: `gpqa_diamond`, 5 samples (one subset), recorded score 0.800
- Git commit of that run: `383c683` (pre-review baseline)
- Data root at capture time: `/ssd4/LLMBenchmark`

Files:

| File | Source |
|---|---|
| `report_gpqa_diamond.json` | `outputs/<run>/gpqa_diamond/reports/MiniMax-M3.1-Flash-Preview/gpqa_diamond.json` |
| `reviews_gpqa_diamond_default.jsonl` | first 3 lines of `reviews/.../gpqa_diamond_default.jsonl` |
| `predictions_gpqa_diamond_default.jsonl` | first 3 lines of `predictions/.../gpqa_diamond_default.jsonl` |

GPQA-Diamond is a public benchmark; the files contain questions, gold answers
and the model's outputs/reasoning for those 3 samples. No credentials or
private data are present.
