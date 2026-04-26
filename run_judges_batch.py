"""Submit/poll Claude Message Batch judge jobs with resumable label writes."""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from pipeline.hints import HINTS
from pipeline.judges import (
    JUDGE_MODEL,
    build_judge_message_params,
    extract_judgement,
)

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def default_state_path(labels_out: str) -> Path:
    path = Path(labels_out)
    if path.suffix == ".jsonl":
        return path.with_suffix(".batch_state.json")
    return path.with_name(path.name + ".batch_state.json")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_label_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"warning: skipping malformed line {lineno} in {path}: {e}")
    return ids


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def dump_model(obj: Any) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def hint_text_for(record: dict, hint_type: str) -> str:
    if hint_type == "none":
        return "(none)"
    return HINTS[hint_type](record["target"])


def build_requests(
    *,
    dataset_path: Path,
    runs_path: Path,
    labels_out: Path,
    limit: int | None,
) -> list[dict]:
    records = {r["question_id"]: r for r in read_jsonl(dataset_path)}
    existing = load_existing_label_ids(labels_out)
    requests = []

    for run in read_jsonl(runs_path):
        run_id = run["run_id"]
        if run_id in existing:
            continue
        if not CUSTOM_ID_RE.match(run_id):
            raise ValueError(f"run_id cannot be used as batch custom_id: {run_id!r}")
        record = records.get(run["question_id"])
        if record is None:
            print(f"warning: skipping {run_id}: question not in dataset")
            continue
        params = build_judge_message_params(
            hint_text_for(record, run["hint_type"]),
            run["thinking"],
            run["response"],
            choices=record.get("choices"),
        )
        requests.append({"custom_id": run_id, "params": params})
        if limit is not None and len(requests) >= limit:
            break

    return requests


def submit_batch(client: Anthropic, args: argparse.Namespace) -> dict | None:
    state_path = Path(args.state)
    existing_state = load_state(state_path)
    if existing_state and existing_state.get("batch_id") and not args.force:
        print(f"State already has batch_id={existing_state['batch_id']}. Poll it or pass --force.")
        return existing_state

    requests = build_requests(
        dataset_path=Path(args.dataset),
        runs_path=Path(args.runs),
        labels_out=Path(args.labels_out),
        limit=args.limit,
    )
    if not requests:
        print("No pending judgements to submit.")
        return None

    state = {
        "batch_id": None,
        "status": "creating",
        "created_before_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": args.dataset,
        "runs_path": args.runs,
        "labels_out": args.labels_out,
        "judge_model": JUDGE_MODEL,
        "request_count": len(requests),
        "custom_ids": [r["custom_id"] for r in requests],
    }
    write_json_atomic(state_path, state)

    if args.dry_run:
        print(f"Dry run: would submit {len(requests)} requests.")
        return state

    print(f"Submitting {len(requests)} judge requests to Claude Message Batches...")
    batch = client.messages.batches.create(requests=requests)
    state.update({
        "batch_id": batch.id,
        "status": batch.processing_status,
        "created_at": getattr(batch, "created_at", None).isoformat()
        if getattr(batch, "created_at", None) else None,
        "expires_at": getattr(batch, "expires_at", None).isoformat()
        if getattr(batch, "expires_at", None) else None,
        "request_counts": dump_model(batch.request_counts),
    })
    write_json_atomic(state_path, state)
    print(f"Submitted batch_id={batch.id} status={batch.processing_status}")
    return state


def append_batch_results(client: Anthropic, batch_id: str, args: argparse.Namespace) -> dict:
    labels_path = Path(args.labels_out)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_label_ids(labels_path)
    stats = {"succeeded": 0, "appended": 0, "skipped_existing": 0,
             "errored": 0, "canceled": 0, "expired": 0, "parse_errors": 0}

    with labels_path.open("a") as out:
        for row in client.messages.batches.results(batch_id):
            run_id = row.custom_id
            result = row.result
            result_type = result.type
            if result_type == "succeeded":
                stats["succeeded"] += 1
                if run_id in existing:
                    stats["skipped_existing"] += 1
                    continue
                try:
                    judgement = extract_judgement(result.message)
                except Exception as e:
                    stats["parse_errors"] += 1
                    print(f"ERROR {run_id}: could not parse judge tool output: {e}")
                    continue
                label = {
                    "run_id": run_id,
                    **judgement,
                    "judge_model": JUDGE_MODEL,
                    "batch_id": batch_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(label) + "\n")
                out.flush()
                existing.add(run_id)
                stats["appended"] += 1
            elif result_type in stats:
                stats[result_type] += 1
                detail = dump_model(result)
                print(f"{result_type.upper()} {run_id}: {detail}")
            else:
                stats["errored"] += 1
                print(f"UNKNOWN_RESULT {run_id}: {dump_model(result)}")
    return stats


def poll_batch(client: Anthropic, args: argparse.Namespace) -> bool:
    state_path = Path(args.state)
    state = load_state(state_path) or {}
    batch_id = args.batch_id or state.get("batch_id")
    if not batch_id:
        raise RuntimeError(
            f"No batch id. Provide --batch-id or submit a batch first. State: {state_path}"
        )

    batch = client.messages.batches.retrieve(batch_id)
    counts = dump_model(batch.request_counts)
    print(f"batch_id={batch.id} status={batch.processing_status} counts={counts}")

    state.update({
        "batch_id": batch.id,
        "status": batch.processing_status,
        "request_counts": counts,
        "results_url": batch.results_url,
        "last_polled_at": datetime.now(timezone.utc).isoformat(),
    })

    if batch.processing_status != "ended":
        write_json_atomic(state_path, state)
        return False

    result_stats = append_batch_results(client, batch.id, args)
    state.update({
        "status": "ended",
        "results_applied_at": datetime.now(timezone.utc).isoformat(),
        "result_stats": result_stats,
    })
    write_json_atomic(state_path, state)
    print(f"Applied results: {result_stats}")
    return True


def list_batches(client: Anthropic, args: argparse.Namespace) -> None:
    for batch in client.messages.batches.list(limit=args.limit_batches):
        print(json.dumps({
            "id": batch.id,
            "status": batch.processing_status,
            "created_at": batch.created_at.isoformat() if batch.created_at else None,
            "expires_at": batch.expires_at.isoformat() if batch.expires_at else None,
            "request_counts": dump_model(batch.request_counts),
            "results_url": batch.results_url,
        }, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["submit", "poll", "wait", "list"])
    ap.add_argument("--runs", default="data/runs.jsonl")
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--labels-out", default="data/labels.jsonl")
    ap.add_argument("--state", default=None)
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Submit at most this many fresh judgements.")
    ap.add_argument("--poll-interval", type=int, default=60)
    ap.add_argument("--limit-batches", type=int, default=10)
    ap.add_argument("--force", action="store_true",
                    help="Allow submit even if the state file already has a batch id.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.state is None:
        args.state = str(default_state_path(args.labels_out))

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = Anthropic()

    if args.command == "list":
        list_batches(client, args)
    elif args.command == "submit":
        submit_batch(client, args)
    elif args.command == "poll":
        poll_batch(client, args)
    elif args.command == "wait":
        state = load_state(Path(args.state))
        if not state or not state.get("batch_id"):
            state = submit_batch(client, args)
            if state is None or args.dry_run:
                return
        while True:
            if poll_batch(client, args):
                return
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
