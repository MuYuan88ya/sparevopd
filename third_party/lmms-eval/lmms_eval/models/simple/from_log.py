"""Replay model responses from lmms-eval sample logs.

The upstream implementation only understands the legacy monolithic JSON log
format. Current lmms-eval releases write one JSONL sample file per task, so a
``--predict_only`` run could not be consumed by ``from_log``. This backend
supports both formats and provides a strict mode suitable for a real
generation/evaluation phase boundary.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from accelerate import Accelerator, DistributedType
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger
from tqdm import tqdm

_SAMPLE_LOG_PATTERN = re.compile(r"_samples_(?P<task>.+)\.jsonl$")
_FAILURE_MARKERS = (
    "[LMMS_EVAL_REQUEST_FAILED",
    "API_ERROR",
    "FUTURE_ERROR",
    "$ERROR$",
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _normalize_responses(value: Any) -> list[str]:
    """Normalize JSONL ``resps`` wrappers without discarding repeated outputs."""
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, str) for item in value):
            return value
        if len(value) == 1:
            return _normalize_responses(value[0])
    if value is None:
        return []
    return [str(value)]


def _validate_responses(responses: list[str], *, source: str) -> None:
    if not responses:
        raise ValueError(f"No generated response found in {source}")
    for response in responses:
        if not response.strip():
            raise ValueError(f"Empty generated response found in {source}")
        marker = next((item for item in _FAILURE_MARKERS if item in response), None)
        if marker is not None:
            raise ValueError(f"Generation failure marker {marker!r} found in {source}")


def _sample_task_name(path: Path) -> str | None:
    match = _SAMPLE_LOG_PATTERN.search(path.name)
    return match.group("task") if match else None


def load_jsonl_responses(log_roots: list[str], *, strict: bool = True) -> dict[str, dict[str, list[str]]]:
    """Load modern lmms-eval sample JSONL files; the newest duplicate wins."""
    paths: list[Path] = []
    for root in log_roots:
        root_path = Path(root).expanduser()
        if root_path.is_file() and root_path.suffix == ".jsonl":
            paths.append(root_path)
        elif root_path.is_dir():
            paths.extend(root_path.rglob("*_samples_*.jsonl"))

    paths = sorted(set(paths), key=lambda path: (path.stat().st_mtime_ns, str(path)))
    responses_by_task: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for path in paths:
        filename_task = _sample_task_name(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                source = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if strict:
                        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
                    eval_logger.warning(f"Skipping invalid JSON in {source}: {exc}")
                    continue

                task = record.get("task_name") or record.get("task") or filename_task
                doc_id = record.get("doc_id")
                if task is None or doc_id is None:
                    message = f"Missing task or doc_id in {source}"
                    if strict:
                        raise ValueError(message)
                    eval_logger.warning(message)
                    continue

                raw_response = record["resps"] if "resps" in record else record.get("filtered_resps")
                responses = _normalize_responses(raw_response)
                if not responses and not strict:
                    eval_logger.warning(f"Skipping sample with no response in {source}")
                    continue

                task_logs = responses_by_task[str(task)]
                normalized_doc_id = str(doc_id)
                previous = task_logs.get(normalized_doc_id)
                if previous is not None and previous != responses:
                    eval_logger.warning(
                        f"Replacing response for task={task!r}, doc_id={doc_id!r} with newer record from {source}"
                    )
                task_logs[normalized_doc_id] = responses

    if strict:
        for task, task_logs in responses_by_task.items():
            for doc_id, responses in task_logs.items():
                _validate_responses(
                    responses,
                    source=f"latest record for task={task!r}, doc_id={doc_id!r}",
                )

    return dict(responses_by_task)


def _load_legacy_json_responses(log_roots: list[str]) -> dict[str, dict[str, list[str]]]:
    """Best-effort compatibility with the pre-JSONL from_log format."""
    result: dict[str, dict[str, list[str]]] = {}
    for log_root in log_roots:
        for root, _, files in os.walk(log_root):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                path = Path(root) / filename
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    task = str(data["model_configs"]["task"])
                    task_logs = {str(item["doc_id"]): _normalize_responses(item.get("resps")) for item in data["logs"]}
                    if task_logs:
                        result[task] = task_logs
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
    return result


@register_model("from_log")
class FromLog(lmms):
    """Offline backend that strictly replays a completed generation run."""

    def __init__(
        self,
        logs: str = "logs",
        model_name: str | None = None,
        model_args: str | None = None,
        have_limits: bool = False,
        strict: bool | str = True,
        **kwargs,
    ) -> None:
        super().__init__()
        del model_name, model_args, have_limits
        if kwargs:
            eval_logger.warning(f"Unknown from_log model_args ignored: {list(kwargs.keys())}")

        self.strict = _parse_bool(strict)
        log_roots = [item.strip() for item in logs.split(",") if item.strip()]
        if not log_roots:
            raise ValueError("from_log requires at least one path in model_args logs=...")

        self.logs = load_jsonl_responses(log_roots, strict=self.strict)
        if not self.logs:
            self.logs = _load_legacy_json_responses(log_roots)
        if not self.logs:
            raise ValueError(
                "No replayable lmms-eval sample logs found. Expected files named "
                "'*_samples_<task>.jsonl' below: " + ", ".join(log_roots)
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self.device = accelerator.device

    def generate_until(self, requests: list[Instance]) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        offsets: dict[tuple[str, str], int] = defaultdict(int)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Replaying responses")

        for request in requests:
            task = str(request.task_name)
            doc_id = str(request.doc_id)
            key = (task, doc_id)
            task_logs = self.logs.get(task, {})
            saved_responses = task_logs.get(doc_id)
            offset = offsets[key]
            if saved_responses is None or offset >= len(saved_responses):
                available = 0 if saved_responses is None else len(saved_responses)
                message = (
                    f"Missing replay response for task={task!r}, doc_id={doc_id!r}, "
                    f"repeat={offset}; available={available}"
                )
                if self.strict:
                    raise KeyError(message)
                eval_logger.error(message)
                text = ""
            else:
                text = saved_responses[offset]
                offsets[key] += 1
            results.append(GenerationResult(text=text, token_counts=TokenCounts()))
            pbar.update(1)

        pbar.close()
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("from_log only supports generative tasks")

    def generate_until_multi_round(self, requests: list[Instance]) -> list[GenerationResult]:
        return self.generate_until(requests)
