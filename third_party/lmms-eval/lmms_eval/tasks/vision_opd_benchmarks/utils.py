"""Vision-OPD tasks missing from the pinned upstream lmms-eval revision."""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from loguru import logger as eval_logger
from PIL import Image

JUDGE_PROMPT = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {ground_truth}\n"
    "The response is: {prediction}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)
JUDGE_PROTOCOL_VERSION = 2

_VISUALPROBE_REPOS = {
    "visual_probe_easy": "Mini-o3/VisualProbe_Easy",
    "visual_probe_medium": "Mini-o3/VisualProbe_Medium",
    "visual_probe_hard": "Mini-o3/VisualProbe_Hard",
}
_CATEGORY_NAMES = {
    "visual_probe_easy": "Easy",
    "visual_probe_medium": "Medium",
    "visual_probe_hard": "Hard",
}
_thread_local = threading.local()


def _rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            with Image.open(io.BytesIO(raw_bytes)) as image:
                return image.convert("RGB")
        path = value.get("path")
        if path:
            with Image.open(path) as image:
                return image.convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value).__name__}")


def zoombench_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    # The original Vision-OPD evaluation intentionally uses the full image,
    # not ZoomBench's crop_image field.
    return [_rgb_image(doc["image"])]


def zoombench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    del lmms_eval_specific_kwargs
    return str(doc.get("query", "")).strip()


def zoombench_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc.get("response", "")).strip()


def _visualprobe_repo(doc: dict[str, Any]) -> str:
    data_source = str(doc.get("data_source", "")).strip().lower()
    try:
        return _VISUALPROBE_REPOS[data_source]
    except KeyError as exc:
        raise ValueError(f"Unsupported VisualProbe data_source: {data_source!r}") from exc


def visualprobe_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    images = doc.get("images") or []
    if not images:
        raise ValueError(f"VisualProbe sample {doc.get('doc_id')!r} has no image")
    relative_path = str(images[0])
    # val.json stores paths such as VisualProbe_Easy/data/foo.jpg, while the
    # repository itself contains data/foo.jpg.
    if "/" in relative_path:
        relative_path = relative_path.split("/", 1)[1]
    local_path = hf_hub_download(
        repo_id=_visualprobe_repo(doc),
        filename=relative_path,
        repo_type="dataset",
    )
    return [_rgb_image(local_path)]


def visualprobe_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    del lmms_eval_specific_kwargs
    problem = str(doc.get("problem", "")).strip()
    return problem.replace("<image>\n", "").replace("<image>", "").strip()


def visualprobe_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc.get("solution", "")).strip()


def _extract_answer(raw_answer: str) -> str:
    if not isinstance(raw_answer, str):
        return ""
    answer = raw_answer.strip()
    think_end = answer.lower().rfind("</think>")
    if think_end != -1:
        answer = answer[think_end + len("</think>") :].strip()

    tagged_answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", answer, flags=re.IGNORECASE | re.DOTALL)
    if tagged_answers:
        return tagged_answers[-1].strip()

    answer_prefixes = list(re.finditer(r"\banswer\s*:\s*", answer, flags=re.IGNORECASE))
    if answer_prefixes:
        return answer[answer_prefixes[-1].end() :].strip()
    return answer


def _parse_judge_decision(raw_output: str) -> float | None:
    """Infer one unambiguous Yes/No decision, mirroring HRBench's option parser."""
    answer = _extract_answer(raw_output)
    decisions = {match.group(0).lower() for match in re.finditer(r"\b(?:yes|no)\b", answer, re.IGNORECASE)}
    if decisions == {"yes"}:
        return 1.0
    if decisions == {"no"}:
        return 0.0
    return None


def _record(
    *, task: str, uid: str, question: str, target: str, prediction: str, category: str | None = None
) -> dict[str, Any]:
    return {
        "task": task,
        "uid": f"{task}:{uid}",
        "question": question,
        "target": target,
        "prediction": _extract_answer(prediction),
        "category": category,
    }


def zoombench_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    uid = str(doc.get("id", doc.get("index", "")))
    return {
        "zoombench_accuracy": _record(
            task="zoombench",
            uid=uid,
            question=zoombench_doc_to_text(doc),
            target=zoombench_doc_to_target(doc),
            prediction=results[0] if results else "",
        )
    }


def visualprobe_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    source = str(doc.get("data_source", "")).strip().lower()
    task = source.replace("visual_probe_", "visualprobe_")
    return {
        "visualprobe_accuracy": _record(
            task=task,
            uid=str(doc.get("doc_id", "")),
            question=visualprobe_doc_to_text(doc),
            target=visualprobe_doc_to_target(doc),
            prediction=results[0] if results else "",
            category=_CATEGORY_NAMES.get(source),
        )
    }


def _normalized_text(value: str) -> str:
    normalized = value.strip().lower().strip("\"'`")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.rstrip(".").strip()


def _rule_match(target: str, prediction: str) -> bool:
    if _normalized_text(target) == _normalized_text(prediction):
        return True
    try:
        from mathruler.grader import grade_answer

        return bool(grade_answer(target, prediction))
    except (ImportError, ValueError, TypeError):
        return False


def _judge_config() -> dict[str, Any]:
    return {
        "api_base": os.getenv("VISION_OPD_JUDGE_API_BASE", "").rstrip("/"),
        "api_key": os.getenv("VISION_OPD_JUDGE_API_KEY", "EMPTY"),
        "model": os.getenv("VISION_OPD_JUDGE_MODEL_ID", ""),
        "workers": max(1, int(os.getenv("JUDGE_PARALLEL_WORKERS", "32"))),
        "max_retries": max(1, int(os.getenv("JUDGE_MAX_RETRIES", "3"))),
        "timeout": float(os.getenv("JUDGE_TIMEOUT", "600")),
        "max_tokens": int(os.getenv("JUDGE_MAX_TOKENS", "2048")),
    }


def _judge_client(config: dict[str, Any]):
    key = (config["api_base"], config["api_key"], config["timeout"])
    if getattr(_thread_local, "client_key", None) != key:
        from openai import OpenAI

        _thread_local.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["api_base"],
            timeout=config["timeout"],
        )
        _thread_local.client_key = key
    return _thread_local.client


def _judge_one(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=record["question"],
        ground_truth=record["target"],
        prediction=record["prediction"],
    )
    last_error = "unknown Judge error"
    for attempt in range(config["max_retries"]):
        try:
            response = _judge_client(config).chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=config["max_tokens"],
            )
            raw_output = (response.choices[0].message.content or "").strip()
            if not raw_output:
                raise ValueError("Judge API returned an empty response")
            score = _parse_judge_decision(raw_output)
            if score is None:
                raise ValueError(f"Judge API returned an ambiguous Yes/No response: {raw_output!r}")
            return {
                "question": record["question"],
                "prediction": record["prediction"],
                "target": record["target"],
                "score": score,
                "judge_output": raw_output,
                "judge_source": "llm",
            }
        except Exception as exc:  # persisted successes allow failed items to be retried
            last_error = str(exc)
            if attempt + 1 < config["max_retries"]:
                time.sleep(1.0)
    raise RuntimeError(f"Judge failed for {record['uid']}: {last_error}")


def _cache_path(task: str) -> Path | None:
    cache_root = os.getenv("VISION_OPD_JUDGE_CACHE_DIR", "")
    return Path(cache_root) / f"{task}.json" if cache_root else None


def _load_cache(task: str, judge_model: str) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    path = _cache_path(task)
    if path is None or not path.exists():
        return path, {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("judge_protocol_version") != JUDGE_PROTOCOL_VERSION:
        eval_logger.info(f"Ignoring stale Judge cache with an older parsing protocol: {path}")
        return path, {}
    cached_model = payload.get("judge_model")
    if cached_model and cached_model != judge_model:
        raise ValueError(
            f"Judge cache {path} uses model {cached_model!r}, not {judge_model!r}. Use a new EXPERIMENT_NAME."
        )
    return path, payload.get("items", {})


def _write_cache(path: Path | None, task: str, judge_model: str, items: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
                "task": task,
                "judge_model": judge_model,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def judged_accuracy(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    tasks = {record["task"] for record in results}
    if len(tasks) != 1:
        raise ValueError(f"Expected one task per aggregation, got {sorted(tasks)}")
    task = next(iter(tasks))
    config = _judge_config()
    cache_path, cache = _load_cache(task, config["model"])
    pending: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    seen_uids: set[str] = set()

    for record in results:
        if record["uid"] in seen_uids:
            raise ValueError(f"Duplicate sample UID in {task}: {record['uid']}")
        seen_uids.add(record["uid"])
        if _rule_match(record["target"], record["prediction"]):
            scores[record["uid"]] = 1.0
            record.update(score=1.0, judge_source="rule", judge_output="Yes")
            continue
        cached = cache.get(record["uid"])
        if (
            cached
            and cached.get("question") == record["question"]
            and cached.get("prediction") == record["prediction"]
            and cached.get("target") == record["target"]
        ):
            scores[record["uid"]] = float(cached["score"])
            record.update(cached)
            record["judge_source"] = "cache"
        else:
            pending.append(record)

    if pending and (not config["api_base"] or not config["model"]):
        raise RuntimeError(
            f"{task} requires an LLM Judge for {len(pending)} samples. "
            "Configure JUDGE_API_BASE/JUDGE_PORT and JUDGE_MODEL_ID."
        )

    errors: list[str] = []
    if pending:
        eval_logger.info(f"Judging {len(pending)} {task} samples with {config['workers']} workers")
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            futures = {executor.submit(_judge_one, record, config): record for record in pending}
            completed_since_checkpoint = 0
            for future in as_completed(futures):
                record = futures[future]
                try:
                    judgement = future.result()
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                cache[record["uid"]] = judgement
                scores[record["uid"]] = float(judgement["score"])
                record.update(judgement)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= 25:
                    _write_cache(cache_path, task, config["model"], cache)
                    completed_since_checkpoint = 0
        _write_cache(cache_path, task, config["model"], cache)

    if errors:
        preview = "; ".join(errors[:3])
        raise RuntimeError(
            f"Judge failed for {len(errors)} {task} samples; successful results were cached. "
            f"Rerun evaluate to retry. First errors: {preview}"
        )
    if len(scores) != len(results):
        raise RuntimeError(f"Only scored {len(scores)}/{len(results)} {task} samples")

    accuracy = sum(scores.values()) / len(results)
    eval_logger.info(f"{task} accuracy: {accuracy * 100:.2f}% ({len(results)} samples)")
    return accuracy
