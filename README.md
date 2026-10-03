# LLMBenchmark

用 [EvalScope](https://github.com/modelscope/evalscope) 跑 OpenAI 兼容端点（MiniMax 等）在三类
benchmark 上的分数，并把结果记进 SQLite，用于跨模型、跨时间对比。

三类：基础知识/推理、软件工程、视觉理解与 OCR。清单不是 EvalScope 全部 200+ 个 benchmark，
而是按「调研模型评测接口」文档的建议收敛过的（见 `benchmarks.yaml` 里的注释）。

## 准备

```bash
PY=/ssd4/envs/aco_py312/bin/python
$PY -m pip install "evalscope[ifeval,ocr_bench,sandbox]==1.12.0" swebench==4.1.0
export MINIMAX_API_KEY=...                             # key 只从环境变量读，不落盘
export HTTPS_PROXY=http://127.0.0.1:18080              # 本机直连 GitHub/DockerHub 不通，走代理

# /ssd4 只剩几十 G，而 live_code_bench 一个数据集就 31G（evalscope 还会再存一份自己的 cache），
# 缓存必须放到大盘上，否则跑到一半会 ENOSPC。
export MODELSCOPE_CACHE=/home/rlan/.cache/modelscope
export MS_CACHE_HOME=$MODELSCOPE_CACHE
export EVALSCOPE_DATASET_DIR=$MODELSCOPE_CACHE/evalscope
```

数据集走 ModelScope（HF 直连不通，EvalScope 默认就是 ModelScope）。首次跑某个 benchmark 会下载数据。

## 用法

```bash
$PY run.py list                                        # 看可选清单
$PY run.py run --suite reasoning --model minimax       # 跑一类
$PY run.py run --suite all --model minimax --dry-run   # 只打印命令，不花钱
$PY run.py run --suite vision --only ocr_bench,docvqa --model minimax
$PY run.py show                                        # 跨模型对比表
$PY run.py show --save results/leaderboard.md          # 存成可提交的 markdown
$PY run.py reharvest                                   # 改了收割逻辑后，从 outputs/ 重新入库（不调 API）
$PY run.py selftest                                    # 收割逻辑自检
```

`--suite` = `reasoning` / `coding` / `vision` / `all`；`--only` 只能从 `benchmarks.yaml`
的清单里挑，不会跑到清单外的 benchmark。要加新模型就在 `configs/models.yaml` 加一条。
断了想接着跑：`--use-cache outputs/minimax/<之前那次目录>` 复用已有的 prediction。

## 跨模型可比性怎么保证

- **题目固定**：EvalScope 默认不打乱数据集（`shuffle=False`），`limit` 取到的就是原序前 N 题。
  换模型再跑一次，题目、顺序、提示词都一样。
- **题目指纹**：每道题的输入会算 SHA1 存进 `samples` 表，换模型后可以直接比对两边
  `input_hash` 是否一致。
- **参数入库**：`runs` 表记录了模型、端点、limit、max_tokens、完整 generation_config、
  原始产出目录，成绩和参数永远绑在一起。

## 分数怎么看

- 每个 benchmark **单独一行**，不合并成一个「总分」——分项差异本身就是信息。
- `truncation_risk` ⚠ 表示平均输出 token 到了上限的 90% 以上。因为 EvalScope 不暴露
  `finish_reason`，只能这样近似判断。**M3.1 是推理模型，thinking token 和正文共用同一个
  额度**（实测 `max_tokens=32768` 时可能全部烧在思考上、正文为空、直接判 0 分），所以
  看到 ⚠ 就该把 `benchmarks.yaml` 里的 `max_tokens` 调高重跑。
- 想看各 subset 明细加 `--subsets`。

## 文件

| 路径 | 作用 |
|---|---|
| `run.py` | 全部逻辑：拼命令 → 调 evalscope → 收割 report → 入库 → 出表 |
| `benchmarks.yaml` | 三类 benchmark 清单、每类每题的 limit、max_tokens |
| `configs/models.yaml` | 被测端点（api_key 只写环境变量名） |
| `compat/sitecustomize.py` | 给 evalscope 子进程打的补丁（见下） |
| `data/swe_bench_django-11532/` | 锁死的 SWE-bench 单题数据 |
| `results/bench.db` | SQLite：`runs` / `scores` / `samples` |
| `outputs/` | EvalScope 原始产出（逐题 prediction、review、HTML 报告），已 gitignore |

`results/bench.db` 本身是二进制，**没有进 git**（每次跑都变，进仓库只会无限膨胀）。
要留档就 `run.py show --save results/leaderboard.md` 提交那个 markdown；原始逐题结果留在
`outputs/`。

## 选型说明

- **SWE-bench 只跑 1 题**：`django__django-11532`。依据是 SWE-bench 官方人工难度标注里的
  medium 档（15 分钟~1 小时）——榜单最佳模型在这一档只有 55~62% 正确率（easy ~95%、
  hard ~26%），区分度刚好；patch 跨 5 个文件；django 的执行环境在 swebench 镜像里最轻。
  数据固化在 `data/`，通过 `dataset_args.local_path` 精确锁死，不会因为上游数据集变动而跑偏。
  Agent 模式（自己 `ls`/`grep`/`pytest` 探索仓库），`max_steps: 30`。
- **token 额度给到 32768**：M3.1 的 thinking 和正文共用额度，给不够就是「因为输出被截断
  而答错」，不是能力问题。
- **没用 LLM judge**：CharXiv 这类需要裁判模型的 benchmark 被排除，避免裁判版本影响分数。
- **没做 DeepSWE / Terminal-Bench**：依赖过重（`datacurve-pier` / `harbor`），性价比不划算。
- **live_code_bench 默认不跑**：数据集 31G，而且 evalscope 每次加载都按不同 cache key 再存一份，
  实测能把盘直接撑爆。要跑就把 `EVALSCOPE_DATASET_DIR` 指到大盘，再把 `benchmarks.yaml` 里
  那行注释打开。
- **用 plain `humaneval`，不用 `humaneval_plus` / `bigcodebench`**：后两者判分要执行模型生成的
  代码，evalscope 强制开沙箱，而沙箱镜像是在 `docker build` 阶段 `pip install` 依赖——容器默认
  bridge 网络在这台机器上出不去（只有 `docker build --network=host` 能通，evalscope 没暴露这个
  开关）。不开沙箱时它们会把每条都判成 false（`Sandbox is not initialized`），是个会骗人的分数。
  plain `humaneval` 直接在本机子进程里跑测试，干净。
- **OCRBench-v2 需要 `compat/sitecustomize.py`**：nltk 从 CVE-2026-12926 起给 `edit_distance`
  加了 2000 字符的 DoS 上限，而「整页 OCR」的答案动辄几千字符，不放开会直接打挂整轮评测。
  `run.py` 会自动把这个目录挂到子进程的 `PYTHONPATH`。

## 实测样本量（`limit` 是每个 subset 各取 N）

| benchmark | subset 数 | 实际题数 |
|---|---|---|
| `mmmu_pro` | 30 | 90 |
| `ocr_bench_v2` | 30 | 90 |
| `math_500` | 5（Level 1-5） | 50 |
| `ifeval` | 1 | 29 |
| `ocr_bench` | 10 | 50 |
| 其余 | 1 | 30 / 20 / 1 |

注意 `limit` 是**每个 subset 各取 N**，多 subset 的 benchmark 实际题数会翻倍；上表是实测值。

## 已知限制

- 机器只有 1 核，耗时主要在 API 往返和 docker 镜像上，不在本地计算。
- 数据集很大：`live_code_bench` 31G、`docvqa`/`mmmu_pro`/`math_vista` 各几 G，缓存盘要留够空间。
- 容器里没有外网（只有 docker build --network host 才行），所以任何需要 `pip install` 的沙箱
  镜像都建不起来。
- HF 直连不通；所有数据集必须能从 ModelScope 拿到。
- swe-bench agentic 会先构建执行环境（ms_enclave），第一次跑比较慢；镜像建好后重复跑也会
  因为 `django==3.0` 在镜像源里找不到而失败（已有结果在库里，重跑前先确认镜像还在）。
