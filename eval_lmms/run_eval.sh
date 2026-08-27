#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMMS_ROOT="${PROJECT_ROOT}/third_party/lmms-eval"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EVAL_PHASE="${EVAL_PHASE:-all}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${SCRIPT_DIR}/data}"
BENCHMARKS="${BENCHMARKS:-}"
SEED="${SEED:-0}"
VERBOSITY="${VERBOSITY:-INFO}"
LIMIT="${LIMIT:-}"

GEN_API_KEY="${GEN_API_KEY:-EMPTY}"
GEN_API_BASE="${GEN_API_BASE:-http://127.0.0.1:${GEN_PORT:-8000}/v1}"
GEN_MODEL_ID="${GEN_MODEL_ID:-}"
GEN_PARALLEL_WORKERS="${GEN_PARALLEL_WORKERS:-32}"
GEN_TIMEOUT="${GEN_TIMEOUT:-120}"
GEN_MAX_RETRIES="${GEN_MAX_RETRIES:-5}"

JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:${JUDGE_PORT:-8000}/v1}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-}"
REQUIRE_JUDGE="${REQUIRE_JUDGE:-false}"

if [[ -z "${EXPERIMENT_NAME}" ]]; then
  echo "ERROR: EXPERIMENT_NAME is required." >&2
  exit 2
fi

case "${EVAL_PHASE}" in
  generate|evaluate|all) ;;
  *)
    echo "ERROR: EVAL_PHASE must be generate, evaluate, or all; got '${EVAL_PHASE}'." >&2
    exit 2
    ;;
esac

if [[ ! -f "${LMMS_ROOT}/lmms_eval/__main__.py" ]]; then
  echo "ERROR: vendored lmms-eval source is missing at ${LMMS_ROOT}." >&2
  exit 2
fi

RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
GENERATION_DIR="${RUN_DIR}/generation"
EVALUATION_DIR="${RUN_DIR}/evaluation"
MANIFEST_PATH="${RUN_DIR}/manifest.json"

mkdir -p "${RUN_DIR}" "${EVAL_DATA_ROOT}"
export HF_HOME="${HF_HOME:-${EVAL_DATA_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONPATH="${LMMS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

normalize_api_base() {
  local base="${1%/}"
  if [[ "${base}" == */chat/completions ]]; then
    base="${base%/chat/completions}"
  fi
  if [[ "${base}" != */v1 ]]; then
    base="${base}/v1"
  fi
  printf '%s' "${base}"
}

discover_model_id() {
  local api_base="$1"
  local api_key="$2"
  "${PYTHON_BIN}" - "${api_base}" "${api_key}" <<'PY'
import json
import sys
import urllib.request

base, key = sys.argv[1:]
request = urllib.request.Request(
    f"{base.rstrip('/')}/models",
    headers={"Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
except Exception as exc:
    raise SystemExit(f"Unable to query {base}/models: {exc}") from exc

model_ids = [item.get("id") for item in payload.get("data", []) if item.get("id")]
if len(model_ids) != 1:
    raise SystemExit(
        f"Expected exactly one model from {base}/models, found {len(model_ids)}: "
        f"{model_ids}. Set the model ID explicitly."
    )
print(model_ids[0])
PY
}

normalize_benchmarks() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys

aliases = {
    "vstar": "vstar_bench",
    "hrbench-4k": "hrbench4k",
    "hrbench-8k": "hrbench8k",
    "mme-realworld": "mmerealworld",
    "mme-realworld-cn": "mmerealworld_cn",
    "mme-realworld-lite": "mmerealworld_lite",
    "cv-bench": "cv_bench",
}
requested = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
normalized = []
for item in requested:
    task = aliases.get(item, item)
    if task not in normalized:
        normalized.append(task)
print(",".join(normalized))
PY
}

write_manifest() {
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${EXPERIMENT_NAME}" "${BENCHMARKS}" "${LMMS_TASKS}" \
    "${EVAL_DATA_ROOT}" "${GENERATION_DIR}" "${EVALUATION_DIR}" "${GEN_MODEL_ID}" \
    "${SEED}" "${LIMIT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    experiment_name,
    benchmarks,
    tasks,
    data_root,
    generation_dir,
    evaluation_dir,
    generation_model,
    seed,
    limit,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "experiment_name": experiment_name,
    "benchmarks": [item for item in benchmarks.split(",") if item],
    "lmms_tasks": [item for item in tasks.split(",") if item],
    "eval_data_root": os.path.abspath(data_root),
    "generation_dir": os.path.abspath(generation_dir),
    "evaluation_dir": os.path.abspath(evaluation_dir),
    "generation_model_id": generation_model,
    "seed": seed,
    "limit": limit or None,
    "lmms_eval_commit": "84770436f0c3b099efbb56f75c6c0db79bb91739",
    "generation_completed_at": datetime.now(timezone.utc).isoformat(),
}
path = Path(manifest_path)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

load_manifest_tasks() {
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
tasks = manifest.get("lmms_tasks", [])
if not tasks:
    raise SystemExit("Generation manifest contains no lmms_tasks")
print(",".join(tasks))
PY
}

load_manifest_value() {
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get(sys.argv[2])
print("" if value is None else value)
PY
}

run_generation() {
  if [[ -z "${BENCHMARKS}" ]]; then
    echo "ERROR: BENCHMARKS is required for EVAL_PHASE=${EVAL_PHASE}." >&2
    exit 2
  fi

  LMMS_TASKS="$(normalize_benchmarks "${BENCHMARKS}")"
  GEN_API_BASE="$(normalize_api_base "${GEN_API_BASE}")"
  if [[ -z "${GEN_MODEL_ID}" ]]; then
    GEN_MODEL_ID="$(discover_model_id "${GEN_API_BASE}" "${GEN_API_KEY}")"
  fi

  echo "[generate] experiment=${EXPERIMENT_NAME}"
  echo "[generate] tasks=${LMMS_TASKS}"
  echo "[generate] endpoint=${GEN_API_BASE} model=${GEN_MODEL_ID}"
  mkdir -p "${GENERATION_DIR}"

  local command=(
    "${PYTHON_BIN}" -m lmms_eval
    --model openai
    --model_args "model_version=${GEN_MODEL_ID},base_url=${GEN_API_BASE},num_concurrent=${GEN_PARALLEL_WORKERS},timeout=${GEN_TIMEOUT},max_retries=${GEN_MAX_RETRIES}"
    --tasks "${LMMS_TASKS}"
    --batch_size 1
    --predict_only
    --output_path "${GENERATION_DIR}"
    --seed "${SEED}"
    --verbosity "${VERBOSITY}"
  )
  if [[ -n "${LIMIT}" ]]; then
    command+=(--limit "${LIMIT}")
  fi
  OPENAI_API_KEY="${GEN_API_KEY}" "${command[@]}"

  # Loading in strict mode rejects empty/error answers before the generation
  # service is shut down. Newer rerun files override older records.
  "${PYTHON_BIN}" - "${GENERATION_DIR}" <<'PY'
import sys
from lmms_eval.models.simple.from_log import load_jsonl_responses

logs = load_jsonl_responses([sys.argv[1]], strict=True)
if not logs:
    raise SystemExit("Generation produced no replayable sample logs")
sample_count = sum(len(task_logs) for task_logs in logs.values())
print(f"[generate] validated {sample_count} generated samples across {len(logs)} tasks")
PY

  write_manifest
  echo "[generate] complete; manifest=${MANIFEST_PATH}"
}

run_evaluation() {
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "ERROR: generation manifest not found: ${MANIFEST_PATH}" >&2
    echo "Run EVAL_PHASE=generate for the same EXPERIMENT_NAME first." >&2
    exit 2
  fi

  LMMS_TASKS="$(load_manifest_tasks)"
  EVAL_DATA_ROOT="$(load_manifest_value eval_data_root)"
  GENERATION_DIR="$(load_manifest_value generation_dir)"
  EVALUATION_DIR="$(load_manifest_value evaluation_dir)"
  SEED="$(load_manifest_value seed)"
  LIMIT="$(load_manifest_value limit)"
  export HF_HOME="${EVAL_DATA_ROOT}/huggingface"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  local needs_judge="${REQUIRE_JUDGE}"
  if [[ ",${LMMS_TASKS}," == *,hrbench4k,* || ",${LMMS_TASKS}," == *,hrbench8k,* || ",${LMMS_TASKS}," == *,hrbench,* ]]; then
    needs_judge=true
  fi

  if [[ "${needs_judge}" == "true" ]]; then
    JUDGE_API_BASE="$(normalize_api_base "${JUDGE_API_BASE}")"
    if [[ -z "${JUDGE_MODEL_ID}" ]]; then
      JUDGE_MODEL_ID="$(discover_model_id "${JUDGE_API_BASE}" "${JUDGE_API_KEY}")"
    fi
    export OPENAI_API_KEY="${JUDGE_API_KEY}"
    export OPENAI_API_URL="${JUDGE_API_BASE}/chat/completions"
    export MODEL_VERSION="${JUDGE_MODEL_ID}"
    echo "[evaluate] judge_endpoint=${JUDGE_API_BASE} judge_model=${JUDGE_MODEL_ID}"
  fi

  echo "[evaluate] experiment=${EXPERIMENT_NAME}"
  echo "[evaluate] tasks=${LMMS_TASKS}"
  echo "[evaluate] replaying=${GENERATION_DIR}"
  mkdir -p "${EVALUATION_DIR}"

  local command=(
    "${PYTHON_BIN}" -m lmms_eval
    --model from_log
    --model_args "logs=${GENERATION_DIR},strict=True,model_name=${EXPERIMENT_NAME}"
    --tasks "${LMMS_TASKS}"
    --batch_size 1
    --log_samples
    --output_path "${EVALUATION_DIR}"
    --seed "${SEED}"
    --verbosity "${VERBOSITY}"
  )
  if [[ -n "${LIMIT}" ]]; then
    command+=(--limit "${LIMIT}")
  fi
  "${command[@]}"
  echo "[evaluate] complete; results=${EVALUATION_DIR}"
}

case "${EVAL_PHASE}" in
  generate)
    run_generation
    ;;
  evaluate)
    run_evaluation
    ;;
  all)
    run_generation
    run_evaluation
    ;;
esac
