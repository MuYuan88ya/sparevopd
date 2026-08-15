from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

BENCHMARK_JSON_MAP = {
    "zoombench": "zoombench.json",
    "vstar": "vstar.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "mme-realworld": "MME_RealWorld.json",
    "mme-realworld-cn": "MME_RealWorld_CN.json",
    "mme-realworld-lite": "MME_RealWorld_Lite.json",
    "mmstar": "mmstar.json",
    "pope": "POPE.json",
    "pope_adv": "POPE_adv.json",
    "pope_pop": "POPE_pop.json",
    "pope_random": "POPE_random.json",
    "cv-bench": "cv_bench.json",
    "mmvp": "mmvp.json",
    "visualprobe": "visualprobe.json",
}


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=indent)
        file.write("\n")
    os.replace(tmp_path, path)


def make_sample_uid(item: dict[str, Any], benchmark: str) -> str:
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    stable_obj = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(stable_obj, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def is_generation_error(record: dict[str, Any]) -> bool:
    answer = record.get("model_answer")
    if not isinstance(answer, str) or not answer.strip():
        return True
    answer = answer.strip()
    return answer.startswith("[API_ERROR]") or answer.startswith("[FUTURE_ERROR]")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def validate_answer_file(
    benchmark: str,
    benchmark_path: str | Path,
    answer_path: str | Path,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_path)
    answer_path = Path(answer_path)
    with open(benchmark_path, encoding="utf-8") as file:
        source_records = json.load(file)
    if not isinstance(source_records, list):
        raise ValueError(f"Benchmark JSON must contain a list: {benchmark_path}")

    answer_records = load_jsonl(answer_path) if answer_path.exists() else []
    source_uids = [make_sample_uid(item, benchmark) for item in source_records]
    answer_uids = [record.get("sample_uid") or make_sample_uid(record, benchmark) for record in answer_records]
    duplicate_source = len(source_uids) - len(set(source_uids))
    duplicate_answers = len(answer_uids) - len(set(answer_uids))
    source_uid_set = set(source_uids)
    answer_uid_set = set(answer_uids)
    missing = source_uid_set - answer_uid_set
    unexpected = answer_uid_set - source_uid_set
    error_count = sum(is_generation_error(record) for record in answer_records)

    return {
        "expected_samples": len(source_records),
        "completed_samples": len(answer_records),
        "error_samples": error_count,
        "duplicate_source_uids": duplicate_source,
        "duplicate_answer_uids": duplicate_answers,
        "missing_samples": len(missing),
        "unexpected_samples": len(unexpected),
        "complete": not any(
            (
                duplicate_source,
                duplicate_answers,
                len(missing),
                len(unexpected),
                error_count,
            )
        ),
    }


def normalize_api_base(api_base: str | None, port: str | int | None) -> str | None:
    if api_base:
        normalized = api_base.rstrip("/")
        return normalized if normalized.endswith("/v1") else normalized + "/v1"
    if port not in (None, ""):
        return f"http://127.0.0.1:{int(port)}/v1"
    return None


def discover_model_id(
    api_base: str,
    api_key: str,
    explicit_model_id: str | None = None,
    *,
    client_factory=None,
) -> str:
    if explicit_model_id:
        return explicit_model_id
    if client_factory is None:
        from openai import OpenAI

        client_factory = OpenAI
    client = client_factory(api_key=api_key, base_url=api_base, timeout=60)
    response = client.models.list()
    model_ids = [str(model.id) for model in response.data]
    if len(model_ids) != 1:
        visible = ", ".join(model_ids) if model_ids else "none"
        raise ValueError(
            f"Expected exactly one model from {api_base}/models, found {len(model_ids)} ({visible}). "
            "Set the model ID explicitly."
        )
    return model_ids[0]
