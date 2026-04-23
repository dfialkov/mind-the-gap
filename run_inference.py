"""Iterate over (question, hint_type) pairs; skip completed runs; append to runs.jsonl."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.hints import ALL_HINT_TYPES
from pipeline.inference import run_single


def load_existing_run_ids(path: Path) -> set:
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


def run_inference(
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    hint_types: list[str] | None = None,
    dataset_path: str = "data/dataset.jsonl",
    runs_out: str = "data/runs.jsonl",
    activations_dir: str = "data/activations",
    max_new_tokens: int = 8192,
    device: str = "mps",
    limit: int | None = None,
    run_timeout: int = 600,
) -> None:
    if hint_types is None:
        hint_types = list(ALL_HINT_TYPES)

    with open(dataset_path) as f:
        dataset = [json.loads(l) for l in f if l.strip()]
    runs_path = Path(runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    act_dir = Path(activations_dir)
    act_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_run_ids(runs_path)

    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if end_think_id is None or end_think_id == tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer has no </think> token id — wrong model?")
    print(f"end_think_id = {end_think_id}")

    total = len(dataset) * len(hint_types)
    done = 0
    fresh = 0
    for record in dataset:
        for hint_type in hint_types:
            done += 1
            run_id = f"{record['question_id']}__{hint_type}"
            if run_id in existing:
                print(f"[{done}/{total}] skip {run_id} (already done)")
                continue
            if limit is not None and fresh >= limit:
                print(f"--limit {limit} reached, stopping.")
                return
            try:
                t0 = time.time()
                result = run_single(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    hint_type=hint_type,
                    end_think_id=end_think_id,
                    activations_dir=act_dir,
                    model_id=model_id,
                    max_new_tokens=max_new_tokens,
                    run_timeout=run_timeout,
                )
            except Exception as e:
                print(f"[{done}/{total}] ERROR {run_id}: {type(e).__name__}: {e}")
                continue
            if result is None:
                print(f"[{done}/{total}] skip {run_id} (no </think> in output)")
                continue
            with open(runs_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            existing.add(run_id)
            fresh += 1
            dt = time.time() - t0
            print(f"[{done}/{total}] {run_id} ({dt:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--hint-types", nargs="+", default=list(ALL_HINT_TYPES))
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--runs-out", default="data/runs.jsonl")
    ap.add_argument("--activations-dir", default="data/activations")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many fresh runs (skipped runs don't count).")
    ap.add_argument("--run-timeout", type=int, default=600,
                    help="Skip any single run that takes longer than this (seconds).")
    args = ap.parse_args()

    run_inference(
        model_id=args.model,
        hint_types=args.hint_types,
        dataset_path=args.dataset,
        runs_out=args.runs_out,
        activations_dir=args.activations_dir,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        limit=args.limit,
        run_timeout=args.run_timeout,
    )


if __name__ == "__main__":
    main()
