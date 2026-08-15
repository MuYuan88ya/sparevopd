import argparse
import json
import sys
from collections import defaultdict

from eval_utils import atomic_write_json


def is_correct(item):
    return str(item.get("judge", "")).strip().lower() == "yes"


def acc_text(correct, total):
    pct = (100.0 * correct / total) if total else 0.0
    return f"{correct}/{total} = {pct:.2f}%"


def calc_vstar(judge_json, benchmark):
    with open(judge_json, encoding="utf-8") as f:
        data = json.load(f)

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in data:
        category = str(item.get("category", "unknown") or "unknown")
        category_stats[category]["total"] += 1
        if is_correct(item):
            overall_correct += 1
            category_stats[category]["correct"] += 1

    for category, stats in category_stats.items():
        total = stats["total"]
        acc = (100.0 * stats["correct"] / total) if total else 0.0
        print(f"{category}: {acc:.2f}% (n={total})")

    overall_acc = (100.0 * overall_correct / len(data)) if data else 0.0
    print(f"{benchmark} Acc: {overall_acc:.2f}% (n={len(data)})")
    return {
        "benchmark": benchmark,
        "accuracy": overall_acc,
        "correct": overall_correct,
        "total": len(data),
        "categories": {
            category: {
                **stats,
                "accuracy": (100.0 * stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for category, stats in category_stats.items()
        },
    }


def calc_hrbench(judge_json, benchmark, benchmark_json):
    with open(judge_json, encoding="utf-8") as f:
        records = json.load(f)
    with open(benchmark_json, encoding="utf-8") as f:
        benchmark_records = json.load(f)

    def image_key(item):
        images = item.get("images") or []
        return str(images[0]) if isinstance(images, list) and images else ""

    def text_key(item):
        return (str(item.get("query", "") or ""), str(item.get("response", "") or ""))

    category_by_image = {}
    category_by_text = {}
    for item in benchmark_records:
        category = str(item.get("category", "unknown") or "unknown")
        img = image_key(item)
        txt = text_key(item)
        if img:
            category_by_image[img] = category
        category_by_text[txt] = category

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in records:
        correct = is_correct(item)
        category = str(item.get("category", "") or "").strip()
        if not category:
            category = category_by_image.get(image_key(item)) or category_by_text.get(text_key(item)) or "unknown"
        category_stats[category]["total"] += 1
        if correct:
            overall_correct += 1
            category_stats[category]["correct"] += 1

    print(f"{benchmark} Acc: {acc_text(overall_correct, len(records))}")
    return {
        "benchmark": benchmark,
        "accuracy": (100.0 * overall_correct / len(records)) if records else 0.0,
        "correct": overall_correct,
        "total": len(records),
        "categories": {
            category: {
                **stats,
                "accuracy": (100.0 * stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for category, stats in category_stats.items()
        },
    }


def calc_mme_realworld(judge_json, benchmark, benchmark_json):
    with open(judge_json, encoding="utf-8") as f:
        records = json.load(f)
    with open(benchmark_json, encoding="utf-8") as f:
        benchmark_records = json.load(f)

    def image_key(item):
        images = item.get("images") or []
        return str(images[0]) if isinstance(images, list) and images else ""

    def text_key(item):
        return (str(item.get("query", "") or ""), str(item.get("response", "") or ""))

    category_by_image = {}
    category_by_text = {}
    for item in benchmark_records:
        category = str(item.get("category", "unknown") or "unknown")
        img = image_key(item)
        txt = text_key(item)
        if img:
            category_by_image[img] = category
        category_by_text[txt] = category

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    supercategory_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in records:
        correct = is_correct(item)
        category = str(item.get("category", "") or "").strip()
        if not category:
            category = category_by_image.get(image_key(item)) or category_by_text.get(text_key(item)) or "unknown"
        supercategory = category.split("/", 1)[0].strip() if "/" in category else category.strip()
        if not supercategory:
            supercategory = "unknown"
        category_stats[category]["total"] += 1
        supercategory_stats[supercategory]["total"] += 1
        if correct:
            overall_correct += 1
            category_stats[category]["correct"] += 1
            supercategory_stats[supercategory]["correct"] += 1

    print(f"{benchmark} Acc: {acc_text(overall_correct, len(records))}")
    def serialize_stats(stats_by_name):
        return {
            name: {
                **stats,
                "accuracy": (100.0 * stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for name, stats in stats_by_name.items()
        }

    return {
        "benchmark": benchmark,
        "accuracy": (100.0 * overall_correct / len(records)) if records else 0.0,
        "correct": overall_correct,
        "total": len(records),
        "categories": serialize_stats(category_stats),
        "supercategories": serialize_stats(supercategory_stats),
    }


def _pope_metrics(items):
    tp = fp = fn = tn = 0
    for item in items:
        gt = str(item.get("response", "")).strip().lower()
        correct = is_correct(item)
        if gt == "yes" and correct:
            tp += 1
        elif gt == "no" and not correct:
            fp += 1
        elif gt == "yes" and not correct:
            fn += 1
        else:
            tn += 1
    total = len(items)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    yes_count = sum(1 for x in items if str(x.get("response", "")).strip().lower() == "yes")
    yes_ratio = yes_count / total if total else 0.0
    return accuracy, precision, recall, f1, yes_ratio, total


def _pope_fmt(label, accuracy, precision, recall, f1, yes_ratio, total):
    return (
        f"{label}: accuracy={100*accuracy:.2f}% precision={100*precision:.2f}% "
        f"recall={100*recall:.2f}% f1={100*f1:.2f}% yes_ratio={100*yes_ratio:.2f}% (n={total})"
    )


def calc_pope(judge_json, benchmark):
    with open(judge_json, encoding="utf-8") as f:
        data = json.load(f)

    a, p, r, f1, yr, n = _pope_metrics(data)
    print(_pope_fmt(benchmark, a, p, r, f1, yr, n))
    return {
        "benchmark": benchmark,
        "accuracy": 100.0 * a,
        "precision": 100.0 * p,
        "recall": 100.0 * r,
        "f1": 100.0 * f1,
        "yes_ratio": 100.0 * yr,
        "total": n,
    }


def calc_cvbench(judge_json, benchmark):
    with open(judge_json, encoding="utf-8") as f:
        data = json.load(f)

    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in data:
        dim = str(item.get("type", "unknown") or "unknown")
        type_stats[dim]["total"] += 1
        if is_correct(item):
            type_stats[dim]["correct"] += 1

    acc_2d = type_stats["2D"]["correct"] / type_stats["2D"]["total"] if type_stats["2D"]["total"] else 0.0
    acc_3d = type_stats["3D"]["correct"] / type_stats["3D"]["total"] if type_stats["3D"]["total"] else 0.0
    acc_avg = (acc_2d + acc_3d) / 2.0

    print(f"{benchmark}: {100*acc_avg:.2f}%")
    return {
        "benchmark": benchmark,
        "accuracy": 100.0 * acc_avg,
        "types": {
            name: {
                **stats,
                "accuracy": (100.0 * stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for name, stats in type_stats.items()
        },
        "total": len(data),
    }


def calc_visualprobe(judge_json, benchmark):
    with open(judge_json, encoding="utf-8") as f:
        data = json.load(f)

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in data:
        category = str(item.get("category", "unknown") or "unknown")
        cat_stats[category]["total"] += 1
        if is_correct(item):
            cat_stats[category]["correct"] += 1

    accs = []
    for cat in ["Easy", "Medium", "Hard"]:
        stats = cat_stats[cat]
        acc = (100.0 * stats["correct"] / stats["total"]) if stats["total"] else 0.0
        accs.append(acc)
        print(f"  {cat}: {acc:.2f}% (n={stats['total']})")

    avg_acc = sum(accs) / len(accs) if accs else 0.0
    print(f"{benchmark} average: {avg_acc:.2f}%")
    return {
        "benchmark": benchmark,
        "accuracy": avg_acc,
        "total": len(data),
        "categories": {
            category: {
                **stats,
                "accuracy": (100.0 * stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for category, stats in cat_stats.items()
        },
    }


def calc_generic(judge_json, benchmark):
    with open(judge_json, encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    c = sum(1 for x in data if is_correct(x))
    accuracy = (100.0 * c / n) if n else 0.0
    print(f"{benchmark} Acc: {c}/{n} = {accuracy:.2f}%")
    return {"benchmark": benchmark, "accuracy": accuracy, "correct": c, "total": n}


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy from judge results")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--judge_json", required=True, type=str, help="Path to judge output JSON")
    parser.add_argument(
        "--benchmark_json",
        default=None,
        type=str,
        help="Path to original benchmark JSON (for category breakdown)",
    )
    parser.add_argument("--output_json", default=None, type=str, help="Optional machine-readable metric output")
    args = parser.parse_args()

    if args.benchmark == "visualprobe":
        result = calc_visualprobe(args.judge_json, args.benchmark)
    elif args.benchmark == "cv-bench":
        result = calc_cvbench(args.judge_json, args.benchmark)
    elif args.benchmark in ("pope", "pope_adv", "pope_pop", "pope_random"):
        result = calc_pope(args.judge_json, args.benchmark)
    elif args.benchmark == "vstar":
        result = calc_vstar(args.judge_json, args.benchmark)
    elif args.benchmark in ("hrbench-4k", "hrbench-8k"):
        if not args.benchmark_json:
            print("WARNING: --benchmark_json not provided, falling back to generic accuracy.", file=sys.stderr)
            result = calc_generic(args.judge_json, args.benchmark)
        else:
            result = calc_hrbench(args.judge_json, args.benchmark, args.benchmark_json)
    elif args.benchmark in ("mme-realworld", "mme-realworld-cn", "mme-realworld-lite"):
        if not args.benchmark_json:
            print("WARNING: --benchmark_json not provided, falling back to generic accuracy.", file=sys.stderr)
            result = calc_generic(args.judge_json, args.benchmark)
        else:
            result = calc_mme_realworld(args.judge_json, args.benchmark, args.benchmark_json)
    else:
        result = calc_generic(args.judge_json, args.benchmark)

    if args.output_json:
        atomic_write_json(args.output_json, result)


if __name__ == "__main__":
    main()
