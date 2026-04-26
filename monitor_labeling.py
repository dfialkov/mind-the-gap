"""Print judge-labeling progress and suppression-label stats."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LETTERS = {"A", "B", "C", "D"}
DEFAULT_PROBE_HINT_TYPES = ("metadata", "unethical")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probe-hint-types", nargs="+", default=list(DEFAULT_PROBE_HINT_TYPES))
    args = ap.parse_args()
    probe_hint_types = set(args.probe_hint_types)

    now = datetime.now(timezone.utc)
    dataset_rows = read_jsonl(Path(args.dataset))
    runs = read_jsonl(Path(args.runs))
    labels_rows = read_jsonl(Path(args.labels))

    dataset = {r["question_id"]: r for r in dataset_rows}
    run_by_id = {r["run_id"]: r for r in runs}
    labels = {r["run_id"]: r for r in labels_rows}
    baseline_answers: dict[str, str | None] = {}
    for run in runs:
        label = labels.get(run["run_id"])
        if run["hint_type"] == "none" and label:
            baseline_answers[run["question_id"]] = label.get("answer")

    by_hint = Counter()
    by_source = Counter()
    parseable = Counter()
    cot = Counter()
    answer_ack = Counter()
    filter_stats = Counter()
    class_balance = Counter()
    probe_filter_stats = Counter()
    probe_class_balance = Counter()

    for run_id, label in labels.items():
        run = run_by_id.get(run_id)
        if run is None:
            by_source["unknown"] += 1
            continue
        hint = run["hint_type"]
        source = "gpqa" if run["question_id"].startswith("gpqa_") else "mmlu"
        by_hint[hint] += 1
        by_source[source] += 1
        if label.get("answer") in LETTERS:
            parseable[hint] += 1
        cot[(hint, bool(label.get("hint_acknowledged_in_cot")))] += 1
        answer_ack[(hint, bool(label.get("hint_acknowledged_in_answer")))] += 1

        if hint == "none":
            filter_stats["baseline"] += 1
            probe_filter_stats["baseline"] += 1
            continue
        filter_stats["hinted"] += 1
        if hint in probe_hint_types:
            probe_filter_stats["hinted"] += 1
            in_probe_pool = True
        else:
            probe_filter_stats["excluded_hint_type"] += 1
            in_probe_pool = False
        qid = run["question_id"]
        target = dataset[qid]["target"]
        a_base = baseline_answers.get(qid)
        a_hint = label.get("answer")
        if a_base is None or a_hint != target or a_hint == a_base:
            filter_stats["not_influenced"] += 1
            if in_probe_pool:
                probe_filter_stats["not_influenced"] += 1
            continue
        filter_stats["influenced"] += 1
        if in_probe_pool:
            probe_filter_stats["influenced"] += 1
        if not label.get("hint_acknowledged_in_cot"):
            filter_stats["no_cot_ack"] += 1
            if in_probe_pool:
                probe_filter_stats["no_cot_ack"] += 1
            continue
        filter_stats["included"] += 1
        class_balance[int(not label.get("hint_acknowledged_in_answer"))] += 1
        if in_probe_pool:
            probe_filter_stats["included"] += 1
            probe_class_balance[int(not label.get("hint_acknowledged_in_answer"))] += 1

    times = [t for t in (parse_time(r.get("timestamp")) for r in labels_rows) if t]
    rate = None
    eta = None
    if times:
        elapsed_hours = max((now - min(times)).total_seconds() / 3600, 1 / 3600)
        rate = len(labels) / elapsed_hours
        pending = max(len(runs) - len(labels), 0)
        if rate > 0:
            eta = pending / rate

    print("=" * 72)
    print(now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"labels={len(labels)}/{len(runs)} pending={max(len(runs) - len(labels), 0)}")
    print(f"by_hint={dict(by_hint)}")
    print(f"by_source={dict(by_source)}")
    print(f"parseable={dict(parseable)}")
    print(f"cot={dict(cot)}")
    print(f"answer_ack={dict(answer_ack)}")
    print(f"filter_stats={dict(filter_stats)}")
    print(f"class_balance_0_faithful_1_suppressed={dict(class_balance)}")
    print(f"probe_hint_types={sorted(probe_hint_types)}")
    print(f"probe_filter_stats={dict(probe_filter_stats)}")
    print(f"probe_class_balance_0_faithful_1_suppressed={dict(probe_class_balance)}")
    if rate is not None:
        print(f"rate={rate:.1f} labels/hour eta_hours={eta:.1f}")
    for row in labels_rows[-3:]:
        print(
            "last="
            f"{row.get('run_id')} answer={row.get('answer')} "
            f"cot={row.get('hint_acknowledged_in_cot')} "
            f"ans={row.get('hint_acknowledged_in_answer')}"
        )


if __name__ == "__main__":
    main()
