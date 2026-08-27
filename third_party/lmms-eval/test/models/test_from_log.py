import json
import os
from pathlib import Path

import pytest

from lmms_eval.api.instance import Instance
from lmms_eval.models.simple.from_log import FromLog, load_jsonl_responses


def _write_sample_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_loads_current_sample_jsonl_and_prefers_raw_responses(tmp_path: Path) -> None:
    path = tmp_path / "model" / "2026-01-01_samples_mmstar.jsonl"
    _write_sample_log(
        path,
        [
            {"doc_id": 7, "resps": ["raw answer"], "filtered_resps": "A"},
            {"doc_id": 8, "filtered_resps": "B"},
        ],
    )

    logs = load_jsonl_responses([str(tmp_path)])

    assert logs == {"mmstar": {"7": ["raw answer"], "8": ["B"]}}


def test_preserves_repeated_responses(tmp_path: Path) -> None:
    path = tmp_path / "2026-01-01_samples_vstar_bench.jsonl"
    _write_sample_log(path, [{"doc_id": "sample", "resps": [["A", "B"]]}])

    logs = load_jsonl_responses([str(path)])

    assert logs["vstar_bench"]["sample"] == ["A", "B"]


@pytest.mark.parametrize(
    "response",
    ["", "API_ERROR: timeout", "FUTURE_ERROR", "[LMMS_EVAL_REQUEST_FAILED after 5 retries]"],
)
def test_strict_mode_rejects_incomplete_generation(tmp_path: Path, response: str) -> None:
    path = tmp_path / "2026-01-01_samples_mmstar.jsonl"
    _write_sample_log(path, [{"doc_id": 0, "filtered_resps": response}])

    with pytest.raises(ValueError):
        load_jsonl_responses([str(tmp_path)], strict=True)


def test_newest_duplicate_wins(tmp_path: Path) -> None:
    old_path = tmp_path / "2026-01-01_samples_mmstar.jsonl"
    new_path = tmp_path / "2026-01-02_samples_mmstar.jsonl"
    _write_sample_log(old_path, [{"doc_id": 0, "filtered_resps": "old"}])
    _write_sample_log(new_path, [{"doc_id": 0, "filtered_resps": "new"}])
    os.utime(old_path, ns=(1, 1))
    os.utime(new_path, ns=(2, 2))

    logs = load_jsonl_responses([str(tmp_path)])

    assert logs["mmstar"]["0"] == ["new"]


def test_backend_replays_repeats_by_task_and_doc_id() -> None:
    backend = FromLog.__new__(FromLog)
    backend.logs = {"mmstar": {"3": ["first", "second"]}}
    backend.strict = True
    backend._rank = 0
    request = Instance(
        request_type="generate_until",
        arguments=("prompt", {}, None, 3, "mmstar", "test"),
        idx=0,
        metadata={"task": "mmstar", "doc_id": 3, "repeats": 2},
    )

    results = backend.generate_until([request, request])

    assert [result.text for result in results] == ["first", "second"]


def test_backend_strict_mode_rejects_missing_sample() -> None:
    backend = FromLog.__new__(FromLog)
    backend.logs = {"mmstar": {}}
    backend.strict = True
    backend._rank = 0
    request = Instance(
        request_type="generate_until",
        arguments=("prompt", {}, None, 9, "mmstar", "test"),
        idx=0,
        metadata={"task": "mmstar", "doc_id": 9, "repeats": 1},
    )

    with pytest.raises(KeyError, match="Missing replay response"):
        backend.generate_until([request])
