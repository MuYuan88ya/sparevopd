# Vision-OPD 评估套件

评估分为生成和 Judge 两个独立阶段。生成模型与 Judge 模型都应提前托管为 OpenAI-compatible vLLM API；本脚本不负责启动或停止模型服务。

## 最简使用方式

先启动待评估模型，例如监听本机 `8000` 端口，然后完成所有 benchmark 的生成：

```bash
EVAL_PHASE=generate \
EXPERIMENT_NAME=sparse-teacher-step100 \
EVAL_DATA_ROOT=/data/vision_eval \
BENCHMARKS=vstar,zoombench,mmstar \
GEN_PORT=8000 \
bash eval/run_eval.sh
```

生成完成后关闭生成服务，释放 GPU，再启动 Judge 模型。随后运行：

```bash
EVAL_PHASE=evaluate \
EXPERIMENT_NAME=sparse-teacher-step100 \
JUDGE_PORT=8000 \
bash eval/run_eval.sh
```

本地 vLLM 不需要填写 API key。脚本内部会使用兼容占位值；只有服务真正校验密钥时才需要设置 `GEN_API_KEY` 或 `JUDGE_API_KEY`。

## 数据和输出目录

`EVAL_DATA_ROOT` 是统一的数据集根目录。缺少某个 benchmark 时，脚本会将原始数据、图片和预处理 JSON 下载或生成到该目录；已经存在的标准 JSON 会直接复用。

默认输出位置不需要配置：

```text
eval/results/<EXPERIMENT_NAME>/
├── manifest.json
├── answers/
│   └── <benchmark>.jsonl
├── judge/
│   └── <benchmark>.json
└── metrics/
    ├── <benchmark>.json
    └── summary.json
```

如需使用其他位置，可以额外设置：

```bash
OUTPUT_ROOT=/path/to/eval-results
```

评估阶段只需要相同的 `EXPERIMENT_NAME` 和输出根目录。benchmark 列表、数据路径和生成文件位置都会从 `manifest.json` 恢复。

## API 配置

本地服务推荐直接传端口：

```bash
GEN_PORT=8000
JUDGE_PORT=8000
```

远程服务可以传完整地址和 API key：

```bash
GEN_API_BASE=https://generation.example.com/v1
GEN_API_KEY=secret

JUDGE_API_BASE=https://judge.example.com/v1
JUDGE_API_KEY=secret
```

脚本默认调用 `/v1/models` 发现模型。当端点只暴露一个模型时无需指定模型 ID；多模型端点需要增加：

```bash
GEN_MODEL_ID=model-to-evaluate
JUDGE_MODEL_ID=judge-model
```

## 阶段和常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `EVAL_PHASE` | `all` | `generate`、`evaluate` 或 `all` |
| `EXPERIMENT_NAME` | 无 | 必填，且必须是单个目录名 |
| `EVAL_DATA_ROOT` | 无 | `generate` 阶段必填 |
| `OUTPUT_ROOT` | `eval/results` | 实验输出根目录 |
| `BENCHMARKS` | 无 | `generate` 阶段必填，逗号分隔 |
| `GEN_API_KEY` | `EMPTY` | 生成 API 密钥，本地服务通常省略 |
| `JUDGE_API_KEY` | `EMPTY` | Judge API 密钥，本地服务通常省略 |
| `PARALLEL_WORKERS` | `256` | 生成请求并发数 |
| `JUDGE_PARALLEL_WORKERS` | `32` | Judge 请求并发数 |
| `MAX_TOKENS` | `32768` | 生成最大 Token 数 |
| `JUDGE_MAX_TOKENS` | `2048` | Judge 最大 Token 数 |
| `ENABLE_THINKING` | 未设置 | 可设为 `True` 或 `False` |

`all` 会先完成全部生成，再执行统一 Judge，但不会替用户切换外部模型服务。同一组 GPU 复用时，应分别运行 `generate` 和 `evaluate`。

## 续跑与完整性检查

- 生成结果按 `sample_uid` 续跑，只重试缺失、空回答或 API 失败样本。
- Judge 结果也按 `sample_uid` 续跑，只请求缺失、失败或回答发生变化的样本。
- 默认要求输入和输出样本完全对应，且不存在重复 UID 或生成错误。
- 需要对不完整结果进行临时评估时，可设置 `ALLOW_PARTIAL_EVAL=true`。
- 同一个实验名不能混用不同生成参数、生成模型或 Judge 模型；需要更换时请使用新的实验名。

## 支持的 benchmark

```text
vstar, zoombench, hrbench-4k, hrbench-8k,
mme-realworld, mme-realworld-cn, mme-realworld-lite,
mmstar, pope, pope_adv, pope_pop, pope_random,
cv-bench, mmvp, visualprobe
```

旧变量 `API_BASE`、`OPENAI_API_KEY`、`OPENAI_MODEL_ID`、`BENCHMARK`、`MODEL_NAME` 和 `OUT_DIR` 仍可作为兼容别名使用，但会输出弃用提示。
