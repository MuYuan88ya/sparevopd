import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

import cal_acc  # noqa: E402
import eval_pipeline  # noqa: E402
import eval_utils  # noqa: E402
import judge_qwenlm  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class FakeModelsClient:
    def __init__(self, model_ids):
        self.model_ids = model_ids

    def __call__(self, **_kwargs):
        models = [SimpleNamespace(id=model_id) for model_id in self.model_ids]
        return SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=models)))


def test_model_discovery_and_explicit_override():
    factory = FakeModelsClient(["only-model"])
    assert eval_utils.discover_model_id("http://localhost:8000/v1", "key", client_factory=factory) == "only-model"
    assert eval_utils.discover_model_id(
        "http://localhost:8000/v1", "key", "explicit", client_factory=FakeModelsClient([])
    ) == "explicit"

    with pytest.raises(ValueError, match="found 0"):
        eval_utils.discover_model_id(
            "http://localhost:8000/v1", "key", client_factory=FakeModelsClient([])
        )
    with pytest.raises(ValueError, match="found 2"):
        eval_utils.discover_model_id(
            "http://localhost:8000/v1", "key", client_factory=FakeModelsClient(["one", "two"])
        )


def test_local_defaults_require_no_api_keys_or_output_root():
    args = eval_pipeline.build_parser().parse_args(["--experiment-name", "local-run"])
    assert args.gen_api_key == "EMPTY"
    assert args.judge_api_key == "EMPTY"
    assert Path(args.output_root) == EVAL_DIR / "results"


def test_answer_validation_detects_errors_and_uid_mismatches(tmp_path):
    benchmark_path = tmp_path / "benchmark.json"
    answer_path = tmp_path / "answers.jsonl"
    source = [
        {"index": 1, "query": "q1", "images": ["1.jpg"]},
        {"index": 2, "query": "q2", "images": ["2.jpg"]},
    ]
    write_json(benchmark_path, source)
    write_jsonl(
        answer_path,
        [
            {**source[0], "sample_uid": "test:index:1", "model_answer": "ok"},
            {**source[1], "sample_uid": "test:index:2", "model_answer": "[API_ERROR] failed"},
        ],
    )

    result = eval_utils.validate_answer_file("test", benchmark_path, answer_path)
    assert result["expected_samples"] == 2
    assert result["completed_samples"] == 2
    assert result["error_samples"] == 1
    assert result["complete"] is False


def test_generate_prepares_every_dataset_before_inference(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(eval_pipeline, "run_command", lambda command: commands.append(command))
    monkeypatch.setattr(eval_pipeline, "discover_model_id", lambda *_args: "served-model")
    monkeypatch.setattr(
        eval_pipeline,
        "validate_answer_file",
        lambda *_args: {
            "expected_samples": 1,
            "completed_samples": 1,
            "error_samples": 0,
            "duplicate_source_uids": 0,
            "duplicate_answer_uids": 0,
            "missing_samples": 0,
            "unexpected_samples": 0,
            "complete": True,
        },
    )
    args = SimpleNamespace(
        data_root=str(tmp_path / "data"),
        output_root=str(tmp_path / "output"),
        experiment_name="experiment",
        benchmarks="vstar,zoombench",
        seed=42,
        gen_api_base="http://localhost:8000/v1",
        gen_api_key="key",
        gen_model_id=None,
        max_tokens=64,
        max_retries=2,
        parallel_workers=4,
        enable_thinking=None,
    )

    eval_pipeline.run_generate(args)
    script_names = [Path(command[1]).name for command in commands]
    assert script_names == ["prepare_data.py", "prepare_data.py", "infer.py", "infer.py"]
    manifest = json.loads((tmp_path / "output" / "experiment" / "manifest.json").read_text())
    assert manifest["generation"]["status"] == "complete"
    assert manifest["generation"]["model_id"] == "served-model"


def test_unified_judge_reuses_existing_results_and_scatters_outputs(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    vstar_answers = run_dir / "answers" / "vstar.jsonl"
    zoom_answers = run_dir / "answers" / "zoombench.jsonl"
    vstar_judge = run_dir / "judge" / "vstar.json"
    zoom_judge = run_dir / "judge" / "zoombench.json"
    write_jsonl(
        vstar_answers,
        [{"sample_uid": "vstar:1", "query": "Pick one", "response": "(A)", "model_answer": "A."}],
    )
    write_jsonl(
        zoom_answers,
        [
            {"sample_uid": "zoom:1", "query": "What?", "response": "cat", "model_answer": "dog"},
            {"sample_uid": "zoom:2", "query": "Where?", "response": "left", "model_answer": "right"},
        ],
    )
    write_json(
        zoom_judge,
        [
            {
                "sample_uid": "zoom:1",
                "query": "What?",
                "response": "cat",
                "model_answer": "dog",
                "judge": "No",
                "judge_source": "llm",
            }
        ],
    )
    manifest_path = run_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "benchmarks": ["vstar", "zoombench"],
            "artifacts": {
                "vstar": {"answer_path": str(vstar_answers), "judge_path": str(vstar_judge)},
                "zoombench": {"answer_path": str(zoom_answers), "judge_path": str(zoom_judge)},
            },
        },
    )
    observed_prompts = []

    def fake_judge(prompts, *_args, **_kwargs):
        observed_prompts.extend(prompts)
        return [{"text": "Yes", "error": None} for _ in prompts]

    monkeypatch.setattr(judge_qwenlm, "judge_via_api", fake_judge)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_qwenlm.py",
            "--manifest",
            str(manifest_path),
            "--api_base",
            "http://localhost:8001/v1",
            "--judge_model",
            "judge-model",
        ],
    )
    judge_qwenlm.main()

    assert len(observed_prompts) == 1
    assert json.loads(vstar_judge.read_text())[0]["judge_source"] == "first letter"
    zoom_results = json.loads(zoom_judge.read_text())
    assert [item["judge"] for item in zoom_results] == ["No", "Yes"]


def test_original_eval_extracts_final_answer_without_reasoning_noise():
    raw = "<think>I considered A and B.</think>\n<answer>C</answer>"
    assert judge_qwenlm.extract_answer(raw) == "C"
    assert judge_qwenlm.extract_answer("Answer: Blue") == "Blue"
    assert judge_qwenlm.extract_answer("<answer>A</answer> then <answer>B</answer>") == "B"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The answer is (B).", "B"),
        ("A is wrong. Final answer: B.", "B"),
        ("答案是 c。", "C"),
        ("A or B", ""),
        ("The answer is A or B", ""),
        ("Apple", ""),
    ],
)
def test_original_eval_mcq_parser_requires_an_unambiguous_option(answer, expected):
    assert judge_qwenlm.extract_first_option(answer) == expected


def test_original_eval_binary_parsers_reject_ambiguous_answers():
    assert judge_qwenlm.pope_extract("Yes, the object is visible.") == "yes"
    assert judge_qwenlm.pope_extract("Yesterday was sunny.") == ""
    assert judge_qwenlm.pope_extract("Yes or no cannot be determined.") == ""
    assert judge_qwenlm.pope_extract("The answer is not yes.") == ""
    assert judge_qwenlm.mmvp_extract("The answer is (b).") == "(b)"
    assert judge_qwenlm.mmvp_extract("Either A or B") == ""


def test_original_eval_judge_parser_returns_only_canonical_decisions():
    assert judge_qwenlm.parse_judge_decision("Yes.") == "Yes"
    assert judge_qwenlm.parse_judge_decision("<answer>No</answer>") == "No"
    assert judge_qwenlm.parse_judge_decision("Yes or no") == ""
    assert judge_qwenlm.parse_judge_decision("I cannot determine this") == ""


def test_original_eval_rules_grade_unambiguous_wrong_mcq_without_judge():
    item = {"query": "Pick one", "response": "(A)", "model_answer": "Final answer: B"}
    assert judge_qwenlm.apply_rule_grading(item, "vstar", grade_answer=None) is None
    assert item["judge"] == "No"
    assert item["judge_source"] == "first letter"


def test_accuracy_writes_machine_readable_metrics(tmp_path):
    judge_path = tmp_path / "judge.json"
    write_json(judge_path, [{"judge": "Yes"}, {"judge": "No"}])
    result = cal_acc.calc_generic(judge_path, "example")
    assert result == {"benchmark": "example", "accuracy": 50.0, "correct": 1, "total": 2}


def test_evaluate_uses_manifest_without_generation_configuration(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    run_dir = output_root / "experiment"
    benchmark_path = tmp_path / "data" / "mmvp.json"
    answer_path = run_dir / "answers" / "mmvp.jsonl"
    judge_path = run_dir / "judge" / "mmvp.json"
    metric_path = run_dir / "metrics" / "mmvp.json"
    source = {"index": 1, "query": "q", "response": "(a)", "images": ["image.jpg"]}
    write_json(benchmark_path, [source])
    write_jsonl(answer_path, [{**source, "sample_uid": "mmvp:index:1", "model_answer": "(a)"}])
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "experiment_name": "experiment",
            "benchmarks": ["mmvp"],
            "artifacts": {
                "mmvp": {
                    "benchmark_json": str(benchmark_path),
                    "answer_path": str(answer_path),
                    "judge_path": str(judge_path),
                    "metric_path": str(metric_path),
                }
            },
        },
    )
    commands = []

    def fake_run(command):
        commands.append(Path(command[1]).name)
        if Path(command[1]).name == "judge_qwenlm.py":
            write_json(judge_path, [{**source, "judge": "Yes"}])
        elif Path(command[1]).name == "cal_acc.py":
            write_json(metric_path, {"benchmark": "mmvp", "accuracy": 100.0, "correct": 1, "total": 1})

    monkeypatch.setattr(eval_pipeline, "run_command", fake_run)
    monkeypatch.setattr(eval_pipeline, "discover_model_id", lambda *_args: "judge-model")
    args = SimpleNamespace(
        output_root=str(output_root),
        experiment_name="experiment",
        judge_api_base="http://localhost:8000/v1",
        judge_api_key="key",
        judge_model_id=None,
        judge_max_tokens=16,
        judge_parallel_workers=2,
        judge_max_retries=2,
        judge_timeout=30,
        allow_partial_eval=False,
    )

    summary_path = eval_pipeline.run_evaluate(args)
    assert commands == ["judge_qwenlm.py", "cal_acc.py"]
    summary = json.loads(summary_path.read_text())
    assert summary["benchmarks"]["mmvp"]["accuracy"] == 100.0
