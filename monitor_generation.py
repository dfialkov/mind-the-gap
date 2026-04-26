"""Print resumable answer-generation progress from dataset/runs JSONL files."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from pipeline.paths import add_project_arg, resolve_project_paths


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    add_project_arg(ap)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--hint-types", nargs="+", required=True)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()
    paths = resolve_project_paths(
        ap,
        args,
        [("dataset", "--dataset"), ("runs", "--runs")],
    )
    dataset_path = args.dataset or str(paths.dataset)
    runs_path = args.runs or str(paths.runs)

    now = datetime.now(timezone.utc)
    dataset = read_jsonl(Path(dataset_path))
    runs = read_jsonl(Path(runs_path))
    run_ids = [r.get("run_id") for r in runs]
    unique_ids = set(run_ids)
    expected = len(dataset) * len(args.hint_types)
    question_source = {
        r["question_id"]: "gpqa" if r["question_id"].startswith("gpqa_") else "mmlu"
        for r in dataset
    }

    by_hint = Counter(r.get("hint_type") for r in runs)
    by_source = Counter(question_source.get(r.get("question_id"), "unknown") for r in runs)
    by_source_hint = Counter(
        (question_source.get(r.get("question_id"), "unknown"), r.get("hint_type"))
        for r in runs
    )
    tokens = [
        r["n_tokens"]
        for r in runs
        if isinstance(r.get("n_tokens"), int)
    ]
    no_response = sum(1 for r in runs if not (r.get("response") or "").strip())
    capped = (
        sum(1 for r in runs if r.get("n_tokens") == args.max_tokens)
        if args.max_tokens is not None
        else 0
    )
    capped_no_response = (
        sum(
            1
            for r in runs
            if r.get("n_tokens") == args.max_tokens
            and not (r.get("response") or "").strip()
        )
        if args.max_tokens is not None
        else 0
    )

    times = [t for t in (parse_time(r.get("timestamp")) for r in runs) if t]
    rate = None
    eta = None
    if times:
        elapsed_hours = max((now - min(times)).total_seconds() / 3600, 1 / 3600)
        rate = len(unique_ids) / elapsed_hours
        pending = max(expected - len(unique_ids), 0)
        if rate > 0:
            eta = pending / rate

    print("=" * 72)
    print(now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"dataset={len(dataset)} questions expected={expected} runs")
    print(f"done={len(unique_ids)}/{expected} pending={max(expected - len(unique_ids), 0)}")
    if len(unique_ids) != len(run_ids):
        print(f"duplicate_rows={len(run_ids) - len(unique_ids)}")
    print(f"by_hint={dict(by_hint)}")
    print(f"by_source={dict(by_source)}")
    for source in ("mmlu", "gpqa"):
        parts = {
            hint: by_source_hint[(source, hint)]
            for hint in args.hint_types
            if by_source_hint[(source, hint)]
        }
        if parts:
            print(f"{source}_by_hint={parts}")
    if tokens:
        print(
            "tokens="
            f"mean={mean(tokens):.0f} p50={median(tokens):.0f} "
            f"p90={percentile(tokens, 0.9)} max={max(tokens)}"
        )
    if args.max_tokens is not None:
        print(
            f"capped={capped} no_response={no_response} "
            f"capped_no_response={capped_no_response}"
        )
    if rate is not None:
        print(f"rate={rate:.1f} runs/hour eta_hours={eta:.1f}")
    for row in runs[-3:]:
        print(
            "last="
            f"{row.get('run_id')} tokens={row.get('n_tokens')} "
            f"response_chars={len(row.get('response') or '')}"
        )


if __name__ == "__main__":
    main()
