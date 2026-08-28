import importlib.util
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

_UTILS_PATH = Path(__file__).parents[2] / "lmms_eval" / "tasks" / "vision_opd_benchmarks" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("vision_opd_benchmark_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(utils)


def test_zoombench_uses_full_image_and_original_text() -> None:
    full_image = Image.new("RGB", (4, 4), "red")
    crop_image = Image.new("RGB", (2, 2), "blue")
    doc = {
        "id": "z1",
        "query": "  question  ",
        "response": " answer ",
        "image": full_image,
        "crop_image": crop_image,
    }

    visual = utils.zoombench_doc_to_visual(doc)[0]
    result = utils.zoombench_process_results(doc, ["model answer"])["zoombench_accuracy"]

    assert visual.size == (4, 4)
    assert utils.zoombench_doc_to_text(doc) == "question"
    assert result["target"] == "answer"
    assert result["prediction"] == "model answer"


def test_visualprobe_resolves_repository_image(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (3, 5), "green").save(image_path)
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(image_path)

    monkeypatch.setattr(utils, "hf_hub_download", fake_download)
    doc = {
        "images": ["VisualProbe_Easy/data/visual_probe_easy_40.jpg"],
        "doc_id": "visual_probe_easy_40",
        "problem": "<image>\nWhat is shown?",
        "solution": "Green",
        "data_source": "visual_probe_easy",
    }

    visual = utils.visualprobe_doc_to_visual(doc)[0]
    result = utils.visualprobe_process_results(doc, ["Green"])["visualprobe_accuracy"]

    assert captured["repo_id"] == "Mini-o3/VisualProbe_Easy"
    assert captured["filename"] == "data/visual_probe_easy_40.jpg"
    assert visual.size == (3, 5)
    assert utils.visualprobe_doc_to_text(doc) == "What is shown?"
    assert result["task"] == "visualprobe_easy"
    assert result["category"] == "Easy"


def test_judged_accuracy_uses_rules_without_api(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VISION_OPD_JUDGE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("VISION_OPD_JUDGE_API_BASE", raising=False)
    monkeypatch.delenv("VISION_OPD_JUDGE_MODEL_ID", raising=False)
    records = [
        {
            "task": "zoombench",
            "uid": "zoombench:1",
            "question": "q",
            "target": "White.",
            "prediction": "white",
            "category": None,
        }
    ]

    assert utils.judged_accuracy(records) == 1.0
    assert records[0]["judge_source"] == "rule"


def test_judged_accuracy_caches_successful_judgements(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VISION_OPD_JUDGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISION_OPD_JUDGE_API_BASE", "http://judge/v1")
    monkeypatch.setenv("VISION_OPD_JUDGE_MODEL_ID", "judge-model")
    calls = []

    def fake_judge(record, config):
        calls.append(record["uid"])
        return {
            "question": record["question"],
            "prediction": record["prediction"],
            "target": record["target"],
            "score": 1.0,
            "judge_output": "Yes",
            "judge_source": "llm",
        }

    monkeypatch.setattr(utils, "_judge_one", fake_judge)
    first = {
        "task": "zoombench",
        "uid": "zoombench:1",
        "question": "q",
        "target": "ground truth",
        "prediction": "semantic equivalent",
        "category": None,
    }

    assert utils.judged_accuracy([first.copy()]) == 1.0
    assert utils.judged_accuracy([first.copy()]) == 1.0
    assert calls == ["zoombench:1"]
    assert (tmp_path / "zoombench.json").exists()


def test_answer_extraction_prefers_final_structured_answer() -> None:
    raw = "<think>Answer: wrong</think>\n<answer>first</answer>\n<answer>Final value</answer>"

    assert utils._extract_answer(raw) == "Final value"
    assert utils._extract_answer("Reasoning\nANSWER:  Blue ") == "Blue"


def test_judge_decision_parser_accepts_formatting_but_rejects_ambiguity() -> None:
    positive = ["Yes", "Yes.", "(YES)", "The answer is yes", "<answer>Yes</answer>"]
    negative = ["No", "No.", "**NO**", "The answer is no", "<answer>No</answer>"]

    assert all(utils._parse_judge_decision(value) == 1.0 for value in positive)
    assert all(utils._parse_judge_decision(value) == 0.0 for value in negative)
    assert utils._parse_judge_decision("Yes or no cannot be determined") is None
    assert utils._parse_judge_decision("maybe") is None


def test_judge_retries_an_ambiguous_response(monkeypatch) -> None:
    outputs = iter(["Yes or No", "Yes."])

    class FakeCompletions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(outputs)))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(utils, "_judge_client", lambda _config: client)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    record = {
        "uid": "zoombench:1",
        "question": "q",
        "target": "ground truth",
        "prediction": "prediction",
    }
    config = {
        "model": "judge",
        "max_tokens": 8,
        "max_retries": 2,
    }

    judgement = utils._judge_one(record, config)

    assert judgement["score"] == 1.0
    assert judgement["judge_output"] == "Yes."


def test_old_judge_cache_is_not_reused(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "zoombench.json"
    cache_path.write_text(
        '{"schema_version": 1, "judge_model": "judge-model", "items": {'
        '"zoombench:1": {"score": 0.0, "judge_output": "Yes."}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("VISION_OPD_JUDGE_CACHE_DIR", str(tmp_path))

    path, cache = utils._load_cache("zoombench", "judge-model")

    assert path == cache_path
    assert cache == {}
