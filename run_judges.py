"""Iterate runs.jsonl; judge each un-judged run with Haiku; append to labels.jsonl."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from pipeline.hints import HINTS
from pipeline.judges import JUDGE_MODEL, judge_run


def hint_text_for(record: dict, hint_type: str) -> str:
    if hint_type == "none":
        return "(none)"
    return HINTS[hint_type](record["target"])


def load_existing_label_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["run_id"])
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line {lineno} in {path}: {e}")
    return ids


def run_judges(
    runs_path: str = "data/runs.jsonl",
    dataset_path: str = "data/dataset.jsonl",
    labels_out: str = "data/labels.jsonl",
    limit: int | None = None,
) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in .env at the repo root "
            "(ANTHROPIC_API_KEY=sk-ant-...) or export it in your shell."
        )
    print(f"ANTHROPIC_API_KEY loaded (len={len(api_key)}).")
    client = Anthropic()

    with open(dataset_path) as f:
        records = {
            r["question_id"]: r
            for r in (json.loads(l) for l in f if l.strip())
        }

    labels_path = Path(labels_out)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_label_ids(labels_path)

    with open(runs_path) as f:
        runs = [json.loads(l) for l in f if l.strip()]

    total = len(runs)
    fresh = 0
    for i, run in enumerate(runs, 1):
        run_id = run["run_id"]
        if run_id in existing:
            print(f"[{i}/{total}] skip {run_id} (already judged)")
            continue
        if limit is not None and fresh >= limit:
            print(f"--limit {limit} reached, stopping.")
            return

        record = records.get(run["question_id"])
        if record is None:
            print(f"[{i}/{total}] ERROR {run_id}: question not in dataset")
            continue
        hint_text = hint_text_for(record, run["hint_type"])

        try:
            judgement = judge_run(client, hint_text, run["thinking"], run["response"])
        except Exception as e:
            print(f"[{i}/{total}] ERROR {run_id}: {type(e).__name__}: {e}")
            continue

        label = {
            "run_id": run_id,
            **judgement,
            "judge_model": JUDGE_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(labels_path, "a") as f:
            f.write(json.dumps(label) + "\n")
        existing.add(run_id)
        fresh += 1
        print(
            f"[{i}/{total}] {run_id} "
            f"cot={judgement['hint_acknowledged_in_cot']} "
            f"ans={judgement['hint_acknowledged_in_answer']} "
            f"pick={judgement['answer']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/runs.jsonl")
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--labels-out", default="data/labels.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many fresh judgements.")
    args = ap.parse_args()

    run_judges(
        runs_path=args.runs,
        dataset_path=args.dataset,
        labels_out=args.labels_out,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
