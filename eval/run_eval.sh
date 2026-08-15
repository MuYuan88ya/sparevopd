#!/usr/bin/env bash
set -euo pipefail

# Vision-OPD staged evaluation.
#
# EVAL_PHASE=generate: prepare every dataset, then generate every answer.
# EVAL_PHASE=evaluate: judge and score an existing experiment manifest.
# EVAL_PHASE=all: run both phases without managing either external API server.

export MKL_SERVICE_FORCE_INTEL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PHASE="${EVAL_PHASE:-all}"

warn_legacy() {
  local old_name="$1"
  local new_name="$2"
  echo "WARNING: ${old_name} is deprecated; use ${new_name}." >&2
}

if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
  if [[ -n "${MODEL_NAME:-}" ]]; then
    warn_legacy MODEL_NAME EXPERIMENT_NAME
    EXPERIMENT_NAME="${MODEL_NAME}"
  else
    echo "ERROR: EXPERIMENT_NAME must be set." >&2
    exit 1
  fi
fi

if [[ -z "${BENCHMARKS:-}" && -n "${BENCHMARK:-}" ]]; then
  warn_legacy BENCHMARK BENCHMARKS
  BENCHMARKS="${BENCHMARK}"
fi
BENCHMARKS="${BENCHMARKS:-}"

if [[ -z "${OUTPUT_ROOT:-}" && -n "${OUT_DIR:-}" ]]; then
  warn_legacy OUT_DIR OUTPUT_ROOT
  OUTPUT_ROOT="${OUT_DIR}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results}"

if [[ -z "${EVAL_DATA_ROOT:-}" && -n "${DATA_DIR:-}" ]]; then
  warn_legacy DATA_DIR EVAL_DATA_ROOT
  EVAL_DATA_ROOT="${DATA_DIR}"
fi
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-}"

if [[ -z "${GEN_API_BASE:-}" && -n "${API_BASE:-}" ]]; then
  warn_legacy API_BASE GEN_API_BASE
  GEN_API_BASE="${API_BASE}"
fi
if [[ -z "${GEN_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  warn_legacy OPENAI_API_KEY GEN_API_KEY
  GEN_API_KEY="${OPENAI_API_KEY}"
fi
if [[ -z "${GEN_MODEL_ID:-}" && -n "${OPENAI_MODEL_ID:-}" ]]; then
  warn_legacy OPENAI_MODEL_ID GEN_MODEL_ID
  GEN_MODEL_ID="${OPENAI_MODEL_ID}"
fi
if [[ -z "${JUDGE_MODEL_ID:-}" && -n "${JUDGE_MODEL:-}" ]]; then
  warn_legacy JUDGE_MODEL JUDGE_MODEL_ID
  JUDGE_MODEL_ID="${JUDGE_MODEL}"
fi

if [[ -n "${JUDGE_MODEL_PATH:-}" ]]; then
  echo "ERROR: JUDGE_MODEL_PATH is no longer supported; configure a Judge OpenAI-compatible API." >&2
  exit 1
fi

GEN_API_BASE="${GEN_API_BASE:-}"
GEN_API_KEY="${GEN_API_KEY:-EMPTY}"
GEN_MODEL_ID="${GEN_MODEL_ID:-}"
JUDGE_API_BASE="${JUDGE_API_BASE:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-}"

if [[ -z "${GEN_API_BASE}" && -n "${GEN_PORT:-}" ]]; then
  GEN_API_BASE="http://127.0.0.1:${GEN_PORT}/v1"
fi
if [[ -z "${JUDGE_API_BASE}" && -n "${JUDGE_PORT:-}" ]]; then
  JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
fi

case "${EVAL_PHASE}" in
  generate|all)
    [[ -n "${EVAL_DATA_ROOT}" ]] || { echo "ERROR: EVAL_DATA_ROOT is required for ${EVAL_PHASE}." >&2; exit 1; }
    [[ -n "${BENCHMARKS}" ]] || { echo "ERROR: BENCHMARKS is required for ${EVAL_PHASE}." >&2; exit 1; }
    [[ -n "${GEN_API_BASE}" ]] || { echo "ERROR: GEN_API_BASE or GEN_PORT is required for ${EVAL_PHASE}." >&2; exit 1; }
    ;;
  evaluate)
    ;;
  *)
    echo "ERROR: EVAL_PHASE must be generate, evaluate, or all; got ${EVAL_PHASE}." >&2
    exit 1
    ;;
esac

if [[ "${EVAL_PHASE}" == "evaluate" || "${EVAL_PHASE}" == "all" ]]; then
  [[ -n "${JUDGE_API_BASE}" ]] || { echo "ERROR: JUDGE_API_BASE or JUDGE_PORT is required for ${EVAL_PHASE}." >&2; exit 1; }
fi

ARGS=(
  --phase "${EVAL_PHASE}"
  --experiment-name "${EXPERIMENT_NAME}"
  --output-root "${OUTPUT_ROOT}"
  --seed "${SEED:-42}"
  --max-tokens "${MAX_TOKENS:-32768}"
  --max-retries "${MAX_RETRIES:-3}"
  --parallel-workers "${PARALLEL_WORKERS:-256}"
  --gen-api-key "${GEN_API_KEY}"
  --judge-api-key "${JUDGE_API_KEY}"
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-2048}"
  --judge-parallel-workers "${JUDGE_PARALLEL_WORKERS:-32}"
  --judge-max-retries "${JUDGE_MAX_RETRIES:-3}"
  --judge-timeout "${JUDGE_TIMEOUT:-600}"
)

[[ -n "${EVAL_DATA_ROOT}" ]] && ARGS+=(--data-root "${EVAL_DATA_ROOT}")
[[ -n "${BENCHMARKS}" ]] && ARGS+=(--benchmarks "${BENCHMARKS}")
[[ -n "${GEN_API_BASE}" ]] && ARGS+=(--gen-api-base "${GEN_API_BASE}")
[[ -n "${GEN_MODEL_ID}" ]] && ARGS+=(--gen-model-id "${GEN_MODEL_ID}")
[[ -n "${JUDGE_API_BASE}" ]] && ARGS+=(--judge-api-base "${JUDGE_API_BASE}")
[[ -n "${JUDGE_MODEL_ID}" ]] && ARGS+=(--judge-model-id "${JUDGE_MODEL_ID}")
[[ -n "${ENABLE_THINKING:-}" ]] && ARGS+=(--enable-thinking "${ENABLE_THINKING}")
[[ "${ALLOW_PARTIAL_EVAL:-false}" == "true" ]] && ARGS+=(--allow-partial-eval)

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Phase: ${EVAL_PHASE}"
echo "Output root: ${OUTPUT_ROOT}"
python3 "${SCRIPT_DIR}/eval_pipeline.py" "${ARGS[@]}" "$@"
