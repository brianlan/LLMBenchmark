# LLM Benchmark Summary

Generated: 2026-10-03T18:00:03+08:00  
Data root: `/ssd4/LLMBenchmark`  
EvalScope: 1.12.0

## minimax

| Dataset | Metric | Score | Num | Done | Profile | Finished | Run |
|---|---|---:|---:|---|---|---|---|
| docvqa | anls | 0.9500 | 5 | 5/5 | smoke | 2026-10-03T15:02:53+08:00 | `20261003_150137_minimax_vision_smoke__docvqa` |
| gpqa_diamond | accuracy | 0.8000 | 5 | 5/5 | smoke | 2026-10-03T14:57:47+08:00 | `20261003_145714_minimax_knowledge_smoke__gpqa_diamond` |
| ifeval | inst_level_loose | 1.0000 | 8 | 5/5 | smoke | 2026-10-03T15:01:28+08:00 | `20261003_145714_minimax_knowledge_smoke__ifeval` |
| ifeval | inst_level_strict | 1.0000 | 8 | 5/5 | smoke | 2026-10-03T15:01:28+08:00 | `20261003_145714_minimax_knowledge_smoke__ifeval` |
| ifeval | prompt_level_loose | 1.0000 | 5 | 5/5 | smoke | 2026-10-03T15:01:28+08:00 | `20261003_145714_minimax_knowledge_smoke__ifeval` |
| ifeval | prompt_level_strict | 1.0000 | 5 | 5/5 | smoke | 2026-10-03T15:01:28+08:00 | `20261003_145714_minimax_knowledge_smoke__ifeval` |
| live_code_bench | accuracy | 0.8500 | 140 | 140/140 | smoke | 2026-10-03T17:35:43+08:00 | `20261003_154821_minimax_swe_smoke__live_code_bench` |
| math_500 | accuracy | 1.0000 | 25 | 25/25 | smoke | 2026-10-03T15:00:01+08:00 | `20261003_145714_minimax_knowledge_smoke__math_500` |
| math_vista | accuracy | 0.8000 | 5 | 5/5 | smoke | 2026-10-03T15:12:27+08:00 | `20261003_150256_minimax_vision_smoke__math_vista` |
| ocr_bench_v2 | accuracy | 0.6529 | 150 | 150/150 | smoke | 2026-10-03T17:54:36+08:00 | `20261003_173543_minimax_vision_smoke__ocr_bench_v2` |
| swe_bench_verified | accuracy | 0.3333 | 3 | 3/3 | smoke | 2026-10-03T15:48:18+08:00 | `20261003_153901_minimax_swe_smoke__swe_bench_verified` |
| swe_bench_verified_agentic | accuracy | 1.0000 | 1 | 1/1 | smoke | 2026-10-03T16:14:02+08:00 | `20261003_160523_minimax_swe_smoke__swe_bench_verified_agentic` |

## In-progress / failed runs

| Run | Model | Dataset | Status | Error |
|---|---|---|---|---|
| `20261003_155026_minimax_swe_smoke__swe_bench_verified_agentic` | minimax | swe_bench_verified_agentic | failed | Traceback (most recent call last):   File "/home/rlan/projects/tmp/LLMBenchmark/bench.py", line 526, in run_one     run_task(task_cfg)   File "/ssd4/envs/aco_py |
| `20261003_154819_minimax_swe_smoke__swe_bench_verified_agentic` | minimax | swe_bench_verified_agentic | failed | Traceback (most recent call last):   File "/home/rlan/projects/tmp/LLMBenchmark/bench.py", line 523, in run_one     run_task(task_cfg)   File "/ssd4/envs/aco_py |
| `20261003_150256_minimax_vision_smoke__ocr_bench_v2` | minimax | ocr_bench_v2 | failed | Traceback (most recent call last):   File "/home/rlan/projects/tmp/LLMBenchmark/bench.py", line 499, in run_one     run_task(task_cfg)   File "/ssd4/envs/aco_py |
