# Vision-OPD

**Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs via On-Policy Self-Distillation**

<p align="center">
📃 <a href="https://arxiv.org/pdf/2605.18740" target="_blank">Paper</a> | 
🤗 <a href="https://huggingface.co/collections/yuanqianhao/vision-opd">Hugging Face</a> 
</p>

## Overview

Vision-OPD is a regional-to-global on-policy self-distillation framework that transfers a model's own privileged regional perception to its full-image policy, enabling fine-grained visual understanding in a single forward pass — without external teachers, ground-truth labels, reward verifiers, or inference-time tool use.

<p align="center">
  <img src="figures/average_bar_chart.png" alt="Vision-OPD Average Scores" width="60%"/>
</p>

<p align="center"><i>Average scores across fine-grained visual understanding benchmarks, including V* Bench, ZoomBench,  HR-Bench 4K, HR-Bench 8K, MME-RealWorld-EN and MME-RealWorld-CN.</i></p>

## Environment Setup

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation
```

## Quick Start

### 1. Deployment

Serve the merged checkpoint with vLLM, for example:

```bash
vllm serve yuanqianhao/Vision-OPD-9B \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 8 \
    --served-model-name Vision-OPD-9B \
    --trust-remote-code
```

The server listens on port 8000 by default. You can then query the model via the OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`.

Please check out our [Hugging Face Collection](https://huggingface.co/collections/yuanqianhao/vision-opd) for all public Vision-OPD checkpoints.

### 2. Evaluation

Evaluation is split into generation and judging so the two externally hosted vLLM services do not need to share GPUs.
First generate every benchmark answer through the model API:

```bash
EVAL_PHASE=generate \
EXPERIMENT_NAME="vision-opd-9b-step100" \
EVAL_DATA_ROOT="/path/to/eval-data" \
BENCHMARKS="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
GEN_PORT=8000 \
bash eval/run_eval.sh
```

Stop the generation service, start the Judge service, and evaluate the saved experiment:

```bash
EVAL_PHASE=evaluate \
EXPERIMENT_NAME="vision-opd-9b-step100" \
JUDGE_PORT=8000 \
bash eval/run_eval.sh
```

Both services use the OpenAI-compatible `/v1/models` and `/v1/chat/completions` APIs. The model ID is discovered
automatically when the endpoint exposes exactly one model; use `GEN_MODEL_ID` or `JUDGE_MODEL_ID` to disambiguate a
multi-model endpoint. Full remote endpoints can be supplied with `GEN_API_BASE` and `JUDGE_API_BASE` instead of ports.
Missing datasets are downloaded under `EVAL_DATA_ROOT`. Answers, Judge results, metrics, and the resumable manifest are
stored under `eval/results/EXPERIMENT_NAME` by default. Set `OUTPUT_ROOT` only when a different output location is needed.
Local vLLM services do not require an API-key setting; `GEN_API_KEY` and `JUDGE_API_KEY` are optional overrides.
See [`eval/README.md`](eval/README.md) for the complete evaluation guide.

Supported benchmarks: `vstar`, `zoombench`, `hrbench-4k`, `hrbench-8k`, `mme-realworld`, `mme-realworld-cn`, `mme-realworld-lite`, `visualprobe`, `mmvp`, `cv-bench`, `mmstar`, `pope`.

To evaluate Qwen3.5 models as baselines, set `ENABLE_THINKING=False` to run in non-thinking mode, for example:

```bash
EVAL_PHASE=generate \
EXPERIMENT_NAME="qwen35-9b" \
EVAL_DATA_ROOT="/path/to/eval-data" \
BENCHMARKS="vstar,zoombench" \
GEN_PORT=8000 \
ENABLE_THINKING=False \
bash eval/run_eval.sh
```

## Training

### 1. Prepare Training Data

Download and preprocess the [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) dataset:

```bash
python scripts/prepare_data.py --data-dir ./data
```

This downloads images and metadata from HuggingFace, extracts archives, and converts `train.jsonl` to the parquet format expected by the training pipeline.

### 2. Training

Launch Vision-OPD training:

```bash
bash scripts/run_vision_opd.sh
```

The launcher injects `prompts/imagine_shared.txt` as a shared system prompt for both the student rollout and the
sparse teacher. The prompt asks the model to identify a relevant region, describe image-grounded local evidence,
and end with exactly one `<answer>...</answer>`. It is loaded at runtime, so existing Parquet files do not need to be
regenerated. To use another prompt file or disable the shared prompt, set:

```bash
SHARED_SYSTEM_PROMPT_FILE=/path/to/prompt.txt bash scripts/run_vision_opd.sh
SHARED_SYSTEM_PROMPT_FILE= bash scripts/run_vision_opd.sh
```

The default `none` mode keeps the existing single-positive-teacher objective. Enable the equal-size positive/negative
teacher weighting method with:

```bash
DISTILLATION_TOKEN_WEIGHT_MODE=contrastive \
CONTRASTIVE_NEGATIVE_SEED=0 \
bash scripts/run_vision_opd.sh
```

Enable positive-teacher entropy weighting with:

```bash
DISTILLATION_TOKEN_WEIGHT_MODE=entropy \
TEACHER_ENTROPY_BETA=1.0 \
bash scripts/run_vision_opd.sh
```

The two modes are mutually exclusive. `TEACHER_BACKGROUND_KEEP_RATIO` controls the positive teacher's sparse
background ratio (default `0.1`); contrastive mode additionally supports `CONTRASTIVE_NEGATIVE_MODE` and
`CONTRASTIVE_NEGATIVE_SEED`. Key hyperparameters can also be edited at the top of the script.
When entropy or positive/negative teacher token weighting is enabled, the trainer reports global mini-batch weight
distribution and supervision-quality metrics. See [`docs/distillation_token_weight_metrics.md`](docs/distillation_token_weight_metrics.md)
for formulas and interpretation.

### 3. Merge Checkpoints

After training, merge the FSDP-sharded checkpoint into a standard HuggingFace model:

```bash
bash scripts/merge_checkpoint.sh <path_to_checkpoint>
```

For example:

```bash
bash scripts/merge_checkpoint.sh ./checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/
```

This merges the FSDP actor shards, saves the model weights, config, tokenizer, and processor into the specified directory. The merged checkpoint can then be loaded directly with `transformers` or served with vLLM.

## Citation

If you find Vision-OPD useful for your research, please consider citing:

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License

Apache-2.0 License
