"""Iterate runs.jsonl; judge each un-judged run with Haiku; append to labels.jsonl."""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from anthropic import Anthropic, APIStatusError, RateLimitError
from dotenv import load_dotenv

from pipeline.hints import HINTS
from pipeline.judges import JUDGE_MODEL, judge_run
from pipeline.paths import add_project_arg, project_paths, resolve_project_paths

MAX_RETRIES = 6
DEFAULT_PATHS = project_paths()


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


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
    runs_path: str = str(DEFAULT_PATHS.runs),
    dataset_path: str = str(DEFAULT_PATHS.dataset),
    labels_out: str = str(DEFAULT_PATHS.labels),
    limit: int | None = None,
    concurrency: int = 1,
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

    pending: list[tuple[int, dict]] = []
    n_skipped = 0
    for i, run in enumerate(runs, 1):
        run_id = run["run_id"]
        if run_id in existing:
            n_skipped += 1
            continue
        if limit is not None and len(pending) >= limit:
            break
        pending.append((i, run))

    total = len(runs)
    n_ok = 0
    n_errors = 0
    t_start = time.time()

    def judge_one(item: tuple[int, dict]) -> tuple[int, str, dict | None, str | None]:
        i, run = item
        run_id = run["run_id"]
        record = records.get(run["question_id"])
        if record is None:
            return i, run_id, None, "question not in dataset"
        hint_text = hint_text_for(record, run["hint_type"])

        for attempt in range(MAX_RETRIES):
            try:
                judgement = judge_run(
                    client, hint_text, run["thinking"], run["response"],
                    choices=record.get("choices"),
                )
                return i, run_id, judgement, None
            except RateLimitError:
                wait = 2 ** attempt
                print(f"[{i}/{total}] {run_id}: rate limited, "
                      f"retry {attempt + 1}/{MAX_RETRIES} in {wait}s...",
                      flush=True)
                time.sleep(wait)
            except APIStatusError as e:
                if e.status_code in (500, 529):
                    wait = 2 ** attempt
                    print(f"[{i}/{total}] {run_id}: server error {e.status_code}, "
                          f"retry {attempt + 1}/{MAX_RETRIES} in {wait}s...",
                          flush=True)
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                return i, run_id, None, f"{type(e).__name__}: {e}"
        return i, run_id, None, "max retries exceeded"

    if not pending:
        print("No pending judgements.")
    else:
        print(f"Judging {len(pending)} fresh runs (concurrency={concurrency})...")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(judge_one, item): item for item in pending}
        for future in as_completed(futures):
            i, run = futures[future]
            run_id = run["run_id"]
            try:
                _, _, judgement, error = future.result()
            except Exception as e:
                n_errors += 1
                print(f"[{i}/{total}] ERROR {run_id}: {type(e).__name__}: {e}",
                      flush=True)
                continue

            if error is not None or judgement is None:
                n_errors += 1
                print(f"[{i}/{total}] ERROR {run_id}: {error}", flush=True)
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
            n_ok += 1
            elapsed = _fmt_duration(time.time() - t_start)
            remaining = len(pending) - n_ok - n_errors
            avg = (time.time() - t_start) / n_ok if n_ok else 0
            eta = _fmt_duration(remaining * avg) if n_ok else "?"
            print(
                f"[{i}/{total}] {run_id} "
                f"cot={judgement['hint_acknowledged_in_cot']} "
                f"ans={judgement['hint_acknowledged_in_answer']} "
                f"pick={judgement['answer']} "
                f"| {n_ok}/{len(pending)} fresh "
                f"| elapsed={elapsed} eta={eta}",
                flush=True,
            )

    elapsed = _fmt_duration(time.time() - t_start)
    print(f"\nJudging done: {n_ok} ok, {n_errors} errors, "
          f"{n_skipped} already done | {elapsed}")


def main():
    ap = argparse.ArgumentParser()
    add_project_arg(ap)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many fresh judgements.")
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()
    paths = resolve_project_paths(
        ap,
        args,
        [
            ("runs", "--runs"),
            ("dataset", "--dataset"),
            ("labels_out", "--labels-out"),
        ],
    )

    run_judges(
        runs_path=args.runs or str(paths.runs),
        dataset_path=args.dataset or str(paths.dataset),
        labels_out=args.labels_out or str(paths.labels),
        limit=args.limit,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
