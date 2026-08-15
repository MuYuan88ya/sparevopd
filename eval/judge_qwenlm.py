import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval_utils import (
    atomic_write_json,
    discover_model_id,
    is_generation_error,
    load_jsonl,
    make_sample_uid,
)
from tqdm import tqdm

MCQ_BENCHMARKS = {
    "hrbench-4k",
    "hrbench-8k",
    "vstar",
    "mme-realworld",
    "mme-realworld-cn",
    "mme-realworld-lite",
    "mmstar",
    "cv-bench",
}
POPE_BENCHMARKS = {"pope", "pope_adv", "pope_pop", "pope_random"}
MMVP_BENCHMARKS = {"mmvp"}

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def extract_first_option(text):
    if not text:
        return ""
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])[\.\)\s]", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])", text)
    return match.group(1) if match else ""


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    match = re.match(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))", answer.strip())
    return match.group(1) if match else ""


def pope_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    value = text.lstrip("*").strip()
    if value.lower().startswith("answer"):
        value = value.split("answer", 1)[1].lstrip(":").lstrip("*").strip()
    if value.lower().startswith("yes"):
        return "yes"
    if value.lower().startswith("no"):
        return "no"
    last = value.rstrip(".").rstrip("*").strip().split()[-1] if value.strip() else ""
    last = last.lstrip("*").rstrip("*").lower()
    return last if last in {"yes", "no"} else text.strip()


def mmvp_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    value = text.strip().lower()
    match = re.search(r"\(([ab])\)", value) or re.search(r"\b([ab])\b", value)
    return f"({match.group(1)})" if match else ""


def first_letter_match(gt, answer):
    gt_value = extract_mcq_option(gt)
    answer_value = extract_first_option(answer)
    return bool(gt_value and answer_value and gt_value == answer_value)


def extract_answer(model_answer_raw):
    if not isinstance(model_answer_raw, str):
        return ""
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>", start + len("<answer>"))
        if start != -1 and end != -1:
            return model_answer_raw[start + len("<answer>") : end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:") :].strip()
    return model_answer_raw.strip()


def judge_via_api(
    prompts,
    api_base,
    api_key,
    judge_model,
    judge_max_tokens,
    *,
    parallel_workers=32,
    max_retries=3,
    timeout=600,
):
    from openai import OpenAI

    thread_local = threading.local()

    def get_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
            thread_local.client = client
        return client

    def call_one(index, prompt):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = get_client().chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("Judge API returned an empty response")
                return index, text, None
            except Exception as exc:  # the exception text is persisted for resumability
                last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(1.0)
        return index, "", last_error or "unknown Judge API error"

    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, index, prompt) for index, prompt in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            index, text, error = future.result()
            results[index] = {"text": text, "error": error}
    return results


def load_existing_judgements(path: Path, benchmark: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Judge output must contain a JSON list: {path}")
    existing = {}
    for record in records:
        uid = record.get("sample_uid") or make_sample_uid(record, benchmark)
        if uid in existing:
            raise ValueError(f"Duplicate sample UID in {path}: {uid}")
        existing[uid] = record
    return existing


def apply_rule_grading(item, benchmark, grade_answer):
    question = str(item.get("query", "")).replace("<image>", "")
    extracted_answer = extract_answer(item.get("model_answer", ""))
    gt = str(item.get("response", ""))
    item["extracted_answer"] = extracted_answer

    if benchmark in POPE_BENCHMARKS and pope_extract(extracted_answer) == gt.strip().lower():
        item["judge"] = "Yes"
        item["judge_source"] = "pope_exact"
        return None
    if benchmark in MMVP_BENCHMARKS:
        prediction = mmvp_extract(extracted_answer)
        if prediction and prediction == gt.strip().lower():
            item["judge"] = "Yes"
            item["judge_source"] = "mmvp_option"
            return None
    if grade_answer is not None:
        try:
            if grade_answer(gt, extracted_answer):
                item["judge"] = "Yes"
                item["judge_source"] = "mathruler"
                return None
        except Exception:
            pass
    if benchmark in MCQ_BENCHMARKS and first_letter_match(gt, extracted_answer):
        item["judge"] = "Yes"
        item["judge_source"] = "first letter"
        return None
    return PROMPT_TEMPLATE.format(gt=gt, response=extracted_answer, question=question)


def manifest_jobs(manifest_path: Path):
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    return [
        {
            "benchmark": benchmark,
            "answer_path": Path(manifest["artifacts"][benchmark]["answer_path"]),
            "judge_path": Path(manifest["artifacts"][benchmark]["judge_path"]),
        }
        for benchmark in manifest["benchmarks"]
    ]


def legacy_job(args):
    if not args.benchmark or not args.model:
        raise ValueError("Use --manifest, or provide both --benchmark and --model.")
    return [{
        "benchmark": args.benchmark,
        "answer_path": Path(args.answer_dir) / args.benchmark / f"{args.model}_answer.jsonl",
        "judge_path": Path(args.judge_dir) / args.benchmark / f"{args.model}_answer.jsonl",
    }]


def main():
    parser = argparse.ArgumentParser(description="Unified API-based LLM Judge")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--model")
    parser.add_argument("--answer_dir", default="model_answer")
    parser.add_argument("--judge_dir", default="judge")
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--judge_model")
    parser.add_argument("--judge_max_tokens", default=2048, type=int)
    parser.add_argument("--parallel_workers", default=32, type=int)
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--timeout", default=600, type=float)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    judge_model = discover_model_id(args.api_base, args.api_key, args.judge_model)
    jobs = manifest_jobs(args.manifest) if args.manifest else legacy_job(args)
    try:
        from mathruler.grader import grade_answer
    except ImportError:
        grade_answer = None

    records_by_benchmark = {}
    pending = []
    prompts = []
    for job in jobs:
        benchmark = job["benchmark"]
        if job["answer_path"].exists():
            records = load_jsonl(job["answer_path"])
        elif args.allow_partial:
            records = []
        else:
            raise FileNotFoundError(f"Answer file not found: {job['answer_path']}")
        existing = load_existing_judgements(job["judge_path"], benchmark)
        seen = set()
        for index, item in enumerate(tqdm(records, desc=f"Rule grading: {benchmark}")):
            uid = item.get("sample_uid") or make_sample_uid(item, benchmark)
            if uid in seen:
                raise ValueError(f"Duplicate sample UID in {job['answer_path']}: {uid}")
            seen.add(uid)
            item["sample_uid"] = uid
            previous = existing.get(uid)
            can_reuse = (
                previous
                and previous.get("judge")
                and not previous.get("judge_error")
                and previous.get("model_answer") == item.get("model_answer")
            )
            if can_reuse:
                for key in ("extracted_answer", "judge", "judge_source"):
                    if key in previous:
                        item[key] = previous[key]
                continue
            if is_generation_error(item):
                if not args.allow_partial:
                    raise ValueError(f"Generation error for {benchmark} sample {uid}")
                item["judge"] = ""
                item["judge_source"] = "generation_error"
                item["judge_error"] = "generation_error"
                continue
            prompt = apply_rule_grading(item, benchmark, grade_answer)
            if prompt is not None:
                pending.append((benchmark, index))
                prompts.append(prompt)
        records_by_benchmark[benchmark] = {"records": records, "path": job["judge_path"]}

    print(f"Calling one unified LLM Judge job for {len(prompts)} cases across {len(jobs)} benchmarks.")
    api_errors = 0
    if prompts:
        results = judge_via_api(
            prompts,
            args.api_base,
            args.api_key,
            judge_model,
            args.judge_max_tokens,
            parallel_workers=args.parallel_workers,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        if len(results) != len(pending):
            raise RuntimeError("Judge API returned a different number of results than requested.")
        for result_index, (benchmark, index) in enumerate(pending):
            result = results[result_index]
            item = records_by_benchmark[benchmark]["records"][index]
            if result["error"]:
                item["judge"] = ""
                item["judge_source"] = "llm_error"
                item["judge_error"] = result["error"]
                api_errors += 1
            else:
                item["judge"] = result["text"]
                item["judge_source"] = "llm"
                item.pop("judge_error", None)

    for benchmark, value in records_by_benchmark.items():
        atomic_write_json(value["path"], value["records"], indent=2)
        print(f"Saved {benchmark} Judge results to: {value['path']}")
    if api_errors:
        raise RuntimeError(
            f"Judge API failed for {api_errors} samples. Successful results were preserved; rerun evaluate to retry."
        )


if __name__ == "__main__":
    main()
