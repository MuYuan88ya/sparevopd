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


def _extract_unique_option(text, valid_options="ABCDEFGH"):
    if not isinstance(text, str) or not text.strip():
        return ""

    value = text.strip()
    valid = re.escape(valid_options.upper())
    explicit_patterns = (
        rf"(?:final\s+answer|correct\s+answer|answer|option|choice)\s*"
        rf"(?:is|:|：|=)?\s*[\(\[]?([{valid}])[\)\]]?\b",
        rf"(?:最终答案|答案|选项)\s*(?:是|为|:|：)?\s*[\(\[]?([{valid}])[\)\]]?",
    )
    explicit_matches = []
    for pattern in explicit_patterns:
        explicit_matches.extend(re.finditer(pattern, value, flags=re.IGNORECASE))
    if explicit_matches:
        match = max(explicit_matches, key=lambda candidate: candidate.start())
        ambiguous_suffix = rf"\s*[\)\]]?\s*(?:or|/|或|和)\s*[\(\[]?[{valid}]\b"
        if re.match(ambiguous_suffix, value[match.end() :], flags=re.IGNORECASE):
            return ""
        return match.group(1).upper()

    candidates = {
        match.group(1).upper()
        for match in re.finditer(
            rf"(?<![A-Za-z0-9])([{valid}])(?![A-Za-z0-9])",
            value,
            flags=re.IGNORECASE,
        )
    }
    return next(iter(candidates)) if len(candidates) == 1 else ""


def extract_first_option(text):
    return _extract_unique_option(text)


def extract_mcq_option(answer):
    return _extract_unique_option(answer)


def _extract_unique_yes_no(text):
    if not isinstance(text, str) or not text:
        return ""
    if re.search(r"\b(?:not|neither)\s+(?:yes|no)\b", text, re.IGNORECASE):
        return ""
    decisions = {match.group(0).lower() for match in re.finditer(r"\b(?:yes|no)\b", text, re.IGNORECASE)}
    return next(iter(decisions)) if len(decisions) == 1 else ""


def pope_extract(text):
    return _extract_unique_yes_no(text)


def mmvp_extract(text):
    option = _extract_unique_option(text, valid_options="AB")
    return f"({option.lower()})" if option else ""


def first_letter_match(gt, answer):
    gt_value = extract_mcq_option(gt)
    answer_value = extract_first_option(answer)
    return bool(gt_value and answer_value and gt_value == answer_value)


def extract_answer(model_answer_raw):
    if not isinstance(model_answer_raw, str):
        return ""
    answer = model_answer_raw.strip()
    think_end = answer.lower().rfind("</think>")
    if think_end != -1:
        answer = answer[think_end + len("</think>") :].strip()

    tagged_answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", answer, flags=re.IGNORECASE | re.DOTALL)
    if tagged_answers:
        return tagged_answers[-1].strip()

    final_markers = list(
        re.finditer(
            r"(?:\b(?:the\s+)?final\s+answer|\banswer|最终答案|答案)\s*(?:is|为|是)?\s*[:：]\s*",
            answer,
            flags=re.IGNORECASE,
        )
    )
    if final_markers:
        return answer[final_markers[-1].end() :].strip()
    return answer


def parse_judge_decision(raw_output):
    answer = extract_answer(raw_output)
    decision = _extract_unique_yes_no(answer)
    if decision == "yes":
        return "Yes"
    if decision == "no":
        return "No"
    return ""


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
                decision = parse_judge_decision(text)
                if not decision:
                    raise ValueError(f"Judge API returned an ambiguous Yes/No response: {text!r}")
                return index, decision, None
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

    if benchmark in POPE_BENCHMARKS:
        prediction = pope_extract(extracted_answer)
        target = pope_extract(gt)
        if prediction and target:
            item["judge"] = "Yes" if prediction == target else "No"
            item["judge_source"] = "pope_exact"
            return None
    if benchmark in MMVP_BENCHMARKS:
        prediction = mmvp_extract(extracted_answer)
        target = mmvp_extract(gt)
        if prediction and target:
            item["judge"] = "Yes" if prediction == target else "No"
            item["judge_source"] = "mmvp_option"
            return None
    if benchmark in MCQ_BENCHMARKS:
        prediction = extract_first_option(extracted_answer)
        target = extract_mcq_option(gt)
        if prediction and target:
            item["judge"] = "Yes" if prediction == target else "No"
            item["judge_source"] = "first letter"
            return None
    if grade_answer is not None:
        try:
            if grade_answer(gt, extracted_answer):
                item["judge"] = "Yes"
                item["judge_source"] = "mathruler"
                return None
        except Exception:
            pass
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
            previous_judge = parse_judge_decision(previous.get("judge", "")) if previous else ""
            can_reuse = (
                previous
                and previous_judge
                and not previous.get("judge_error")
                and previous.get("model_answer") == item.get("model_answer")
            )
            if can_reuse:
                for key in ("extracted_answer", "judge", "judge_source"):
                    if key in previous:
                        item[key] = previous[key]
                item["judge"] = previous_judge
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
