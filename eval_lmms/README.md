# Vision-OPD lmms-eval 评估

这个目录提供一套与原 `eval/` 完全独立的新评估入口。旧评估代码没有被删除或修改。

`lmms-eval` 的完整源码已固定在 `third_party/lmms-eval`，对应上游提交
`84770436f0c3b099efbb56f75c6c0db79bb91739`。它不是 Git submodule，也不需要
额外 clone 或单独安装 Vision-OPD 的适配包。

## 为什么不用 `use_cache`

`use_cache` 适合减少同一种评估运行的重复请求，但不适合作为严格的两阶段边界。
`--predict_only` 会把任务 metric 临时改成 `bypass`，这会改变 response cache 的任务
指纹；第二阶段恢复真实 metric 后可能 cache miss，并重新调用生成模型。

本实现使用严格日志回放：

```text
generate
  └─ openai backend 连接外部生成 vLLM
     └─ --predict_only 写每个 task 的 JSONL

evaluate
  └─ from_log backend 只读取上述 JSONL
     └─ task process_results / aggregation / LLM Judge
```

评估阶段不会创建生成 API 客户端。缺少任务、样本、重复输出，或者最新答案为空、包含
API 错误标记时会立即失败，不会静默重新生成。因此生成服务可以在第一阶段结束后彻底
关闭，再把 GPU 交给 Judge 服务。

## 环境

源码已经包含在仓库内，但 Python 运行依赖仍需存在。建议使用独立评估环境，因为训练
环境固定了 `av==16.1.0`，而当前 lmms-eval 要求 `av<16`。无需额外安装脚本；可以直接
使用 vendored 项目自带的 `pyproject.toml`：

```bash
cd third_party/lmms-eval
uv sync
cd ../..
```

随后通过 `PYTHON_BIN` 指定该环境：

```bash
export PYTHON_BIN="$PWD/third_party/lmms-eval/.venv/bin/python"
```

如果当前激活环境已经安装 lmms-eval 所需依赖，可以省略 `PYTHON_BIN`。入口始终优先
加载仓库里的源码，不会误用 site-packages 中的另一版本。

## 第一阶段：生成

先在脚本外启动 OpenAI-compatible vLLM。API key 默认为 `EMPTY`，本机服务只需指定
端口；未指定 `GEN_MODEL_ID` 时，脚本会调用 `/v1/models` 自动发现唯一模型。

```bash
EVAL_PHASE=generate \
EXPERIMENT_NAME=sparse-teacher-step100 \
BENCHMARKS=vstar,mmstar,pope \
GEN_PORT=8000 \
PYTHON_BIN="$PWD/third_party/lmms-eval/.venv/bin/python" \
bash eval_lmms/run_eval.sh
```

远程服务示例：

```bash
EVAL_PHASE=generate \
EXPERIMENT_NAME=sparse-teacher-step100 \
BENCHMARKS=vstar,mmstar \
GEN_API_BASE=http://generator-host:8000/v1 \
GEN_API_KEY=EMPTY \
GEN_MODEL_ID=my-checkpoint \
bash eval_lmms/run_eval.sh
```

数据默认下载到 `eval_lmms/data`，结果默认写入
`eval_lmms/results/<EXPERIMENT_NAME>`。可分别用 `EVAL_DATA_ROOT` 和 `OUTPUT_ROOT`
覆盖。生成结束时会严格检查答案，并写入不含 API key 的 `manifest.json`。

## 第二阶段：评估

关闭生成 vLLM，释放 GPU。对于只使用规则评分的任务，第二阶段不需要任何 API：

```bash
EVAL_PHASE=evaluate \
EXPERIMENT_NAME=sparse-teacher-step100 \
bash eval_lmms/run_eval.sh
```

第二阶段会从 manifest 恢复 task 列表和生成日志位置，不需要再次传 `BENCHMARKS`、
`EVAL_DATA_ROOT` 或生成 API 参数。

HRBench、ZoomBench 和 VisualProbe 会调用 LLM Judge。先启动 Judge vLLM，再运行：

```bash
EVAL_PHASE=evaluate \
EXPERIMENT_NAME=sparse-teacher-step100 \
JUDGE_PORT=8000 \
JUDGE_API_KEY=EMPTY \
bash eval_lmms/run_eval.sh
```

`JUDGE_MODEL_ID` 同样支持 `/v1/models` 单模型自动发现。ZoomBench 和 VisualProbe
会并发提交 Judge 请求，并在实验目录保存逐样本 Judge cache；评估中断后重新执行时，
答案未变化的成功结果会直接复用，只重试缺失或失败样本。对于其他自定义、也会访问
Judge 的 lmms-eval task，请额外设置 `REQUIRE_JUDGE=true`。

## 单次顺序运行

`EVAL_PHASE=all` 会严格先完成所有生成，再开始后处理和 Judge，但不会替你启停外部
服务。如果生成模型与 Judge 是两个同时在线的端点，可以使用：

```bash
EVAL_PHASE=all \
EXPERIMENT_NAME=baseline \
BENCHMARKS=hrbench-4k,mmstar \
GEN_PORT=8000 \
JUDGE_PORT=8001 \
bash eval_lmms/run_eval.sh
```

需要复用同一组 GPU 时，应使用两个独立命令执行 `generate` 和 `evaluate`。

## 参数

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `EVAL_PHASE` | `all` | `generate`、`evaluate` 或 `all` |
| `EXPERIMENT_NAME` | 无 | 必填，作为阶段交接和结果目录名 |
| `BENCHMARKS` | 无 | 生成阶段必填，逗号分隔 |
| `OUTPUT_ROOT` | `eval_lmms/results` | 实验输出根目录 |
| `EVAL_DATA_ROOT` | `eval_lmms/data` | Hugging Face 数据和缓存根目录 |
| `GEN_API_BASE` | `http://127.0.0.1:$GEN_PORT/v1` | 生成服务地址 |
| `GEN_PORT` | `8000` | 本机生成服务端口简写 |
| `GEN_API_KEY` | `EMPTY` | 生成服务 API key |
| `GEN_MODEL_ID` | 自动发现 | 生成模型 ID |
| `GEN_PARALLEL_WORKERS` | `32` | 生成并发数 |
| `JUDGE_API_BASE` | `http://127.0.0.1:$JUDGE_PORT/v1` | Judge 服务地址 |
| `JUDGE_PORT` | `8000` | 本机 Judge 端口简写 |
| `JUDGE_API_KEY` | `EMPTY` | Judge API key |
| `JUDGE_MODEL_ID` | 自动发现 | Judge 模型 ID |
| `JUDGE_PARALLEL_WORKERS` | `32` | ZoomBench/VisualProbe Judge 并发数 |
| `JUDGE_MAX_RETRIES` | `3` | 单个 Judge 请求最大尝试次数 |
| `JUDGE_TIMEOUT` | `600` | Judge 请求超时秒数 |
| `JUDGE_MAX_TOKENS` | `2048` | Judge 最大输出 Token 数 |
| `PYTHON_BIN` | `python3` | lmms-eval 运行环境解释器 |
| `LIMIT` | 无 | 调试时限制每个 task 的样本数 |
| `SEED` | `0` | lmms-eval 随机种子 |

旧评估名会自动映射到 lmms-eval task：

| 输入名 | lmms-eval task |
|---|---|
| `vstar` | `vstar_bench` |
| `hrbench-4k` / `hrbench-8k` | `hrbench4k` / `hrbench8k` |
| `mme-realworld` | `mmerealworld` |
| `mme-realworld-cn` | `mmerealworld_cn` |
| `mme-realworld-lite` | `mmerealworld_lite` |
| `cv-bench` | `cv_bench` |
| `zoombench` | `zoombench`（Vision-OPD 新增） |
| `visualprobe` | `visualprobe` group（Vision-OPD 新增） |

`mmstar`、`pope`、`pope_adv`、`pope_pop`、`pope_random` 和 `mmvp` 可直接使用；
也可以在 `BENCHMARKS` 中直接传任意 vendored lmms-eval task 名。

新增 benchmark 保持旧 `eval/` 的定义：

- `zoombench` 使用 `inclusionAI/ZoomBench` 的完整 `image`，不使用 `crop_image`；问题和
  标准答案分别来自 `query`、`response`。
- `visualprobe` 是 `visualprobe_easy`、`visualprobe_medium`、`visualprobe_hard` 三个
  子任务组成的 group。每档分别统计准确率，最终分数为三档准确率的等权平均，而不是
  按样本数加权。
- 两者先执行规范化精确匹配和可选 MathRuler；无法规则确认的答案使用与旧脚本相同的
  Yes/No 语义一致性 Judge prompt。
- Judge 输出采用与 HRBench 选项解析相同的原则：`Yes.`、`(NO)`、`The answer is yes` 和
  `<answer>No</answer>` 等格式可以解析；同时出现 Yes/No 或没有明确结论时会重试，
  不会直接记为错误答案。
- Judge API 最终失败会使评估返回非零，不会被静默记成错误答案；已成功项保存在
  `judge_cache/`，可以续跑。缓存同时校验问题、标准答案、模型回答、Judge 模型和解析
  协议版本；旧解析协议产生的缓存会自动忽略。

## 实验目录与结果查看

不额外维护实验管理脚本。每个 `EXPERIMENT_NAME` 本身就是一个实验目录，例如：

```text
eval_lmms/results/
├── baseline/
├── sparse-teacher-step100/
└── sparse-teacher-step200/
```

建议直接用 checkpoint 或训练实验名作为 `EXPERIMENT_NAME`。同名实验重新生成时，task、
数据根目录、生成模型、seed 和 limit 必须与已有 `manifest.json` 一致；配置或 checkpoint
发生变化时使用新的实验名，避免覆盖和混合结果。manifest 只负责 generate/evaluate 两个
阶段之间的参数交接，不作为另一套实验数据库。

vendored `lmms-eval` 已自带 Web UI 和 Log Viewer。启动方式：

```bash
cd third_party/lmms-eval
uv run --extra tui lmms-eval-ui
```

浏览器打开后进入 **Log Viewer**，将 **Logs Path** 设置为结果根目录的绝对路径，例如：

```text
/path/to/Vision-OPD/eval_lmms/results
```

然后点击 **Scan**。Log Viewer 会递归查找所有实验中的 `*_results.json`，可直接查看：

- 实验/模型名、评估时间和 task 列表；
- 每个 benchmark 的聚合指标与运行配置；
- 使用 `--log_samples` 保存的逐样本回答、评分字段和分页记录。

当前两阶段流程的最终评估记录以 `EXPERIMENT_NAME` 作为原生 lmms-eval 的 model name，
因此在 Log Viewer 中可以直接按实验名识别。`generation/` 下的 predict-only 记录是阶段一
的原始生成日志，最终指标应查看 `evaluation/` 下、model name 等于实验名的记录。

如果需要云端实验追踪，lmms-eval 也原生支持 `--wandb_args` 和 `--wandb_log_samples`；
当前封装默认使用无需账号的本地 Log Viewer。

## 输出结构

```text
eval_lmms/results/<EXPERIMENT_NAME>/
├── manifest.json
├── generation/
│   └── <generation-model>/
│       ├── *_results.json
│       └── *_samples_<task>.jsonl
├── judge_cache/
│   └── <task>.json
└── evaluation/
    └── <EXPERIMENT_NAME>/
        ├── *_results.json
        └── *_samples_<task>.jsonl
```

最终指标位于 `evaluation/<EXPERIMENT_NAME>/*_results.json`；第二阶段 sample 日志包含
回放答案和每个任务的评分字段，便于逐样本审计。
