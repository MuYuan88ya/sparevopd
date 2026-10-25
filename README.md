# SparseVOPD

**Same-Image Sparse Visual-Token Teachers for On-Policy Self-Distillation**

<p align="center">
  <a href="https://github.com/MuYuan88ya/sparevopd">Code</a> |
  <a href="https://github.com/VisionOPD/Vision-OPD">Original Vision-OPD Repository</a> |
  <a href="https://arxiv.org/abs/2605.18740">Vision-OPD Paper</a> |
  <a href="https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K">Training Data</a>
</p>

## Overview

SparseVOPD is an extension of
[Vision-OPD](https://github.com/VisionOPD/Vision-OPD) for fine-grained visual
reasoning with on-policy self-distillation. It replaces the crop-only teacher
condition with a **same-image sparse visual-token teacher**:

- The **student** receives the complete image and all visual tokens.
- The **positive teacher** receives the same image, text prompt, and student
  response, but its language model retains all visual tokens covered by the
  annotated bounding box and only a configurable, spatially distributed subset
  of background tokens.
- Optional token-reweighting strategies use either a paired negative teacher or
  the positive teacher's predictive entropy to concentrate distillation on more
  reliable response tokens.
- A shared visual-imagination prompt encourages both student and teacher to
  identify the relevant region, describe grounded local evidence, and reason
  without invoking image-manipulation tools at inference time.

The method preserves the on-policy training principle: the student first
generates a response, and the teacher evaluates that same response token by
token under a stronger visual condition. No external teacher model is required.

> **Relationship to Vision-OPD.** This repository is built on the official
> [Vision-OPD codebase](https://github.com/VisionOPD/Vision-OPD) and retains its
> training data, on-policy distillation objective, and evaluation targets. The
> sparse same-image teacher, token-selection session, contrastive and
> entropy-based weighting, visual-imagination prompt, and staged evaluation
> workflow are extensions implemented in this repository.

## Method

### Same-image sparse positive teacher

For each image, the bounding box is mapped to the merged visual-token grid. The
positive teacher keeps:

1. every visual token intersecting the target box; and
2. `ceil(background_token_count × background_keep_ratio)` background tokens.

The default `fixed_stratified` selector distributes retained background tokens
over the image using a deterministic Morton-order stratification. Selection is
stable for a fixed sample ID and seed. The student continues to use the dense
visual sequence.

```text
Original image
  ├─ Student: all visual tokens
  └─ Positive teacher: all target tokens + sparse global background tokens
                         └─ token-level distillation on the student's response
```

The teacher still runs the complete image through the vision encoder. Token
filtering is applied to the merged visual features before they enter the teacher
LLM, together with the matching attention-mask update. Consequently, the method
reduces the teacher LLM's effective visual sequence but should not be described
as vision-encoder pruning.

### Optional distillation-token weighting

The launcher exposes three mutually exclusive modes:

| Mode | Token weight | Description |
|---|---|---|
| `none` | `1` | Standard positive-teacher distillation. |
| `contrastive` | `max(0, log p+(y_t) - log p-(y_t))` | A negative teacher receives an equal-size, disjoint token set sampled from the positive mask's complement. |
| `entropy` | `exp(-β H_teacher)` | Low-entropy positive-teacher tokens receive greater weight; weights are normalized to mean 1 over the global optimizer mini-batch. |

Both weighting modes operate on valid response tokens only. Their training-time
diagnostics are documented in
[`docs/distillation_token_weight_metrics.md`](docs/distillation_token_weight_metrics.md).

### Shared visual-imagination prompt

[`prompts/imagine_shared.txt`](prompts/imagine_shared.txt) is injected as a
shared system prompt at runtime. It asks the model to identify the relevant
region, state what becomes visually clear under closer mental inspection, and
finish with exactly one `<answer>...</answer>`. Because the sparse teacher
reuses the student's tokenized prompt and response, prompt and response-token
alignment are preserved. Existing Parquet files do not need to be regenerated.

## Results

The current 4B experiment improves the six-benchmark average from **77.07** to
**77.92**. The six-benchmark suite consists of V* Bench, ZoomBench, HR-Bench 4K,
HR-Bench 8K, MME-RealWorld-EN, and MME-RealWorld-CN.

| Evaluation | Vision-OPD 4B | SparseVOPD 4B | Improvement |
|---|---:|---:|---:|
| Six-benchmark average | 77.07 | **77.92** | **+0.85** |
| V* Bench | 92.15 | **93.19** | **+1.04** |
| ZoomBench | 59.76 | **62.25** | **+2.49** |
| HR-Bench 4K | 84.50 | **86.98** | **+2.48** |
| HR-Bench 8K | 80.38 | **81.75** | **+1.37** |

All values are reported as percentage-point scores, and higher is better. The
table lists the four benchmark-level comparisons currently reported in addition
to the aggregate over all six evaluations.

## Installation

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation
```

## Training

### Prepare the data

Download and convert the
[Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K)
training set:

```bash
python scripts/prepare_data.py --data-dir ./data
```

The processed dataset is written to `data/train.parquet`. Bounding boxes remain
in image coordinates and are mapped to the model's visual-token grid during
training.

### Launch training

The default launcher enables the same-image sparse positive teacher, keeps 10%
of the non-target background tokens, and uses the shared visual-imagination
prompt:

```bash
bash scripts/run_vision_opd.sh
```

Select a distillation-token weighting mode with an environment variable:

```bash
# Equal-size positive/negative sparse teachers
DISTILLATION_TOKEN_WEIGHT_MODE=contrastive \
CONTRASTIVE_NEGATIVE_SEED=0 \
bash scripts/run_vision_opd.sh

# Positive-teacher entropy weighting
DISTILLATION_TOKEN_WEIGHT_MODE=entropy \
TEACHER_ENTROPY_BETA=1.0 \
bash scripts/run_vision_opd.sh
```

Important launcher options:

| Variable | Default | Description |
|---|---:|---|
| `DISTILLATION_TOKEN_WEIGHT_MODE` | `none` | `none`, `contrastive`, or `entropy` |
| `TEACHER_BACKGROUND_KEEP_RATIO` | `0.1` | Fraction of non-target visual tokens retained by the positive teacher |
| `TEACHER_VISUAL_SELECTION_SEED` | `0` | Seed for deterministic stratified background selection |
| `CONTRASTIVE_NEGATIVE_SEED` | `0` | Seed for complement-based negative masks |
| `TEACHER_ENTROPY_BETA` | `1.0` | Entropy-weight sensitivity |
| `SHARED_SYSTEM_PROMPT_FILE` | `prompts/imagine_shared.txt` | Shared student/teacher system prompt; set to an empty value to disable |

Additional Hydra overrides can be appended directly to the launcher command.
Model path, batch sizes, sequence lengths, GPU count, and checkpoint locations
can be adjusted at the top of [`scripts/run_vision_opd.sh`](scripts/run_vision_opd.sh).

### Merge a checkpoint

```bash
bash scripts/merge_checkpoint.sh \
  ./checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/
```

The merge script exports the FSDP actor checkpoint as a standard Hugging Face
model that can be loaded by Transformers or served with vLLM.

## Evaluation

The recommended evaluation entry point is the vendored, revision-pinned
`lmms-eval` workflow under `eval_lmms/`. Generation and scoring are separate
stages, allowing the same GPU pool to host the evaluated checkpoint first and
the LLM Judge afterward.

Prepare the evaluation environment once:

```bash
cd third_party/lmms-eval
uv sync
cd ../..
export PYTHON_BIN="$PWD/third_party/lmms-eval/.venv/bin/python"
```

Generate answers from an externally hosted OpenAI-compatible vLLM server:

```bash
EVAL_PHASE=generate \
EXPERIMENT_NAME=sparsevopd-4b \
BENCHMARKS=vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn \
GEN_PORT=8000 \
bash eval_lmms/run_eval.sh
```

After generation, stop that service, start the Judge service if required, and
replay the saved answers through the task evaluators:

```bash
EVAL_PHASE=evaluate \
EXPERIMENT_NAME=sparsevopd-4b \
JUDGE_PORT=8000 \
bash eval_lmms/run_eval.sh
```

Local API keys default to `EMPTY`. Data is stored under `eval_lmms/data`, while
isolated experiment outputs are written to
`eval_lmms/results/<EXPERIMENT_NAME>`. See
[`eval_lmms/README.md`](eval_lmms/README.md) for remote endpoints, Judge
requirements, resumability, task aliases, output structure, and the native
lmms-eval Log Viewer. The original project-specific evaluation implementation
is retained under `eval/` for compatibility.

## Serving

Serve a merged checkpoint through vLLM:

```bash
vllm serve /path/to/merged-checkpoint \
  --served-model-name SparseVOPD-4B \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

## Acknowledgements and Citation

SparseVOPD is derived from the official
[Vision-OPD](https://github.com/VisionOPD/Vision-OPD) implementation. We thank
the Vision-OPD authors and the open-source `verl` and `lmms-eval` communities.

If this repository is useful in your work, please cite the original Vision-OPD
paper:

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
