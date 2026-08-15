import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval_utils import (
    BENCHMARK_JSON_MAP,
    atomic_write_json,
    discover_model_id,
    normalize_api_base,
    validate_answer_file,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_SCHEMA_VERSION = 1


def parse_benchmarks(value: str) -> list[str]:
    benchmarks = [item.strip() for item in value.split(",") if item.strip()]
    if not benchmarks:
        raise ValueError("At least one benchmark is required.")
    duplicates = sorted({item for item in benchmarks if benchmarks.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate benchmarks: {duplicates}")
    unsupported = [item for item in benchmarks if item not in BENCHMARK_JSON_MAP]
    if unsupported:
        raise ValueError(f"Unsupported benchmarks: {unsupported}")
    return benchmarks


def run_command(args: list[str]) -> None:
    redacted = []
    hide_next = False
    for value in args:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        redacted.append(value)
        hide_next = value in {"--api_key", "--api-key", "--gen-api-key", "--judge-api-key"}
    print("+", " ".join(redacted), flush=True)
    subprocess.run(args, check=True)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment manifest not found: {path}. Run EVAL_PHASE=generate first."
        )
    with open(path, encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema in {path}")
    return manifest


def benchmark_entry(run_dir: Path, data_root: Path, benchmark: str) -> dict:
    return {
        "benchmark_json": str((data_root / BENCHMARK_JSON_MAP[benchmark]).resolve()),
        "answer_path": str((run_dir / "answers" / f"{benchmark}.jsonl").resolve()),
        "judge_path": str((run_dir / "judge" / f"{benchmark}.json").resolve()),
        "metric_path": str((run_dir / "metrics" / f"{benchmark}.json").resolve()),
        "expected_samples": 0,
        "completed_samples": 0,
        "error_samples": 0,
        "status": "pending",
    }


def initialize_manifest(args, benchmarks: list[str], model_id: str) -> tuple[Path, dict]:
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / args.experiment_name
    manifest_path = run_dir / "manifest.json"
    data_root = Path(args.data_root).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("experiment_name") != args.experiment_name:
            raise ValueError("Existing manifest has a different experiment name.")
        if manifest.get("benchmarks") != benchmarks:
            raise ValueError(
                "Existing experiment uses different benchmarks. Choose a new EXPERIMENT_NAME "
                "or use the original benchmark list."
            )
        if Path(manifest["data_root"]).resolve() != data_root:
            raise ValueError("Existing experiment uses a different EVAL_DATA_ROOT.")
        previous_generation = manifest.get("generation", {})
        incompatible = {
            "seed": (manifest.get("seed"), args.seed),
            "model_id": (previous_generation.get("model_id"), model_id),
            "max_tokens": (previous_generation.get("max_tokens"), args.max_tokens),
            "enable_thinking": (previous_generation.get("enable_thinking"), args.enable_thinking),
        }
        changed = {
            name: values
            for name, values in incompatible.items()
            if values[0] is not None and values[0] != values[1]
        }
        if changed:
            raise ValueError(
                f"Existing experiment has incompatible generation settings: {changed}. "
                "Choose a new EXPERIMENT_NAME."
            )
    else:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_name": args.experiment_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmarks": benchmarks,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "run_dir": str(run_dir.resolve()),
            "seed": args.seed,
            "generation": {},
            "evaluation": {},
            "artifacts": {
                benchmark: benchmark_entry(run_dir, data_root, benchmark)
                for benchmark in benchmarks
            },
        }

    manifest["generation"] = {
        "api_base": args.gen_api_base,
        "model_id": model_id,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "parallel_workers": args.parallel_workers,
        "enable_thinking": args.enable_thinking,
        "status": "running",
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest


def run_generate(args) -> Path:
    if not args.data_root:
        raise ValueError("EVAL_DATA_ROOT/--data-root is required for generate.")
    if not args.benchmarks:
        raise ValueError("BENCHMARKS/--benchmarks is required for generate.")
    if not args.gen_api_base:
        raise ValueError("GEN_API_BASE or GEN_PORT is required for generate.")

    benchmarks = parse_benchmarks(args.benchmarks)
    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    for benchmark in benchmarks:
        run_command(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_data.py"),
                "--benchmark",
                benchmark,
                "--data_dir",
                str(data_root),
            ]
        )

    model_id = discover_model_id(args.gen_api_base, args.gen_api_key, args.gen_model_id)
    manifest_path, manifest = initialize_manifest(args, benchmarks, model_id)

    incomplete = []
    for benchmark in benchmarks:
        entry = manifest["artifacts"][benchmark]
        command = [
            sys.executable,
            str(SCRIPT_DIR / "infer.py"),
            "--benchmark",
            benchmark,
            "--benchmark_json",
            entry["benchmark_json"],
            "--output_path",
            entry["answer_path"],
            "--model_name",
            args.experiment_name,
            "--seed",
            str(args.seed),
            "--api_base",
            args.gen_api_base,
            "--api_key",
            args.gen_api_key,
            "--model_id",
            model_id,
            "--max_tokens",
            str(args.max_tokens),
            "--max_retries",
            str(args.max_retries),
            "--parallel_workers",
            str(args.parallel_workers),
        ]
        if args.enable_thinking is not None:
            command.extend(["--enable_thinking", args.enable_thinking])
        run_command(command)

        validation = validate_answer_file(
            benchmark,
            entry["benchmark_json"],
            entry["answer_path"],
        )
        entry.update(validation)
        entry["status"] = "complete" if validation["complete"] else "incomplete"
        if not validation["complete"]:
            incomplete.append(benchmark)
        atomic_write_json(manifest_path, manifest)

    manifest["generation"]["status"] = "complete" if not incomplete else "incomplete"
    manifest["generation"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)
    if incomplete:
        raise RuntimeError(
            f"Generation is incomplete for {incomplete}. Results were preserved; rerun generate to retry."
        )
    print(f"Generation complete. Manifest: {manifest_path}")
    return manifest_path


def validate_manifest_answers(manifest: dict, allow_partial: bool) -> None:
    invalid = []
    for benchmark in manifest["benchmarks"]:
        entry = manifest["artifacts"][benchmark]
        validation = validate_answer_file(
            benchmark,
            entry["benchmark_json"],
            entry["answer_path"],
        )
        entry.update(validation)
        entry["status"] = "complete" if validation["complete"] else "incomplete"
        if not validation["complete"]:
            invalid.append(benchmark)
    if invalid and not allow_partial:
        raise RuntimeError(
            f"Refusing to evaluate incomplete generation results for {invalid}. "
            "Rerun generate or set ALLOW_PARTIAL_EVAL=true."
        )


def run_evaluate(args) -> Path:
    if not args.judge_api_base:
        raise ValueError("JUDGE_API_BASE or JUDGE_PORT is required for evaluate.")
    run_dir = Path(args.output_root).expanduser().resolve() / args.experiment_name
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    validate_manifest_answers(manifest, args.allow_partial_eval)
    judge_model_id = discover_model_id(
        args.judge_api_base,
        args.judge_api_key,
        args.judge_model_id,
    )
    previous_judge_model = manifest.get("evaluation", {}).get("model_id")
    if previous_judge_model and previous_judge_model != judge_model_id:
        raise ValueError(
            f"Existing experiment was judged by {previous_judge_model!r}, not {judge_model_id!r}. "
            "Choose a new EXPERIMENT_NAME to avoid mixing Judge outputs."
        )
    manifest["evaluation"] = {
        "api_base": args.judge_api_base,
        "model_id": judge_model_id,
        "status": "running",
    }
    atomic_write_json(manifest_path, manifest)

    judge_command = [
        sys.executable,
        str(SCRIPT_DIR / "judge_qwenlm.py"),
        "--manifest",
        str(manifest_path),
        "--api_base",
        args.judge_api_base,
        "--api_key",
        args.judge_api_key,
        "--judge_model",
        judge_model_id,
        "--judge_max_tokens",
        str(args.judge_max_tokens),
        "--parallel_workers",
        str(args.judge_parallel_workers),
        "--max_retries",
        str(args.judge_max_retries),
        "--timeout",
        str(args.judge_timeout),
    ]
    if args.allow_partial_eval:
        judge_command.append("--allow_partial")
    try:
        run_command(judge_command)
    except Exception as exc:
        manifest["evaluation"].update(
            {
                "status": "incomplete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise

    metrics = {}
    for benchmark in manifest["benchmarks"]:
        entry = manifest["artifacts"][benchmark]
        command = [
            sys.executable,
            str(SCRIPT_DIR / "cal_acc.py"),
            "--benchmark",
            benchmark,
            "--judge_json",
            entry["judge_path"],
            "--benchmark_json",
            entry["benchmark_json"],
            "--output_json",
            entry["metric_path"],
        ]
        run_command(command)
        with open(entry["metric_path"], encoding="utf-8") as file:
            metrics[benchmark] = json.load(file)

    summary_path = run_dir / "metrics" / "summary.json"
    atomic_write_json(
        summary_path,
        {
            "experiment_name": args.experiment_name,
            "judge_model_id": judge_model_id,
            "benchmarks": metrics,
        },
    )
    manifest["evaluation"].update(
        {
            "status": "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "summary_path": str(summary_path.resolve()),
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(f"Evaluation complete. Summary: {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged Vision-OPD evaluation")
    parser.add_argument("--phase", choices=["generate", "evaluate", "all"], default="all")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int, default=256)
    parser.add_argument("--enable-thinking", choices=["True", "False"])
    parser.add_argument("--gen-api-base")
    parser.add_argument("--gen-api-key", default="EMPTY")
    parser.add_argument("--gen-model-id")
    parser.add_argument("--judge-api-base")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model-id")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-parallel-workers", type=int, default=32)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-timeout", type=float, default=600)
    parser.add_argument("--allow-partial-eval", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in {".", ".."}:
        raise ValueError("EXPERIMENT_NAME must be a single directory name, not a path.")
    args.gen_api_base = normalize_api_base(args.gen_api_base, None)
    args.judge_api_base = normalize_api_base(args.judge_api_base, None)
    if args.phase in {"generate", "all"}:
        run_generate(args)
    if args.phase in {"evaluate", "all"}:
        run_evaluate(args)


if __name__ == "__main__":
    main()
