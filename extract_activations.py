"""Extract local forward-pass activations for generated runs."""
import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.inference import PROBE_LOCATIONS, capture_activations_from_text
from pipeline.model_config import local_model_id, thinking_boundary_for_model


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _generated_text(run: dict, boundary: str) -> str | None:
    thinking = run.get("thinking")
    response = run.get("response")
    if thinking is None or response is None:
        return None
    return f"{thinking}{boundary}{response}"


def _has_all_activations(run: dict) -> bool:
    paths = run.get("activation_paths") or {}
    return all(
        paths.get(loc) is not None and Path(paths[loc]).exists()
        for loc in PROBE_LOCATIONS
    )


def extract_activations(
    model_id: str | None = None,
    dataset_path: str = "data/dataset.jsonl",
    runs_path: str = "data/runs.jsonl",
    activations_dir: str = "data/activations",
    device: str = "mps",
    limit: int | None = None,
    thinking_boundary: str | None = None,
) -> None:
    dataset = {
        row["question_id"]: row
        for row in _load_jsonl(Path(dataset_path))
    }
    runs_file = Path(runs_path)
    runs = _load_jsonl(runs_file)
    if not runs:
        print(
            f"No generated runs found at {runs_file}. "
            "Run generate_answers.py first."
        )
        return

    act_dir = Path(activations_dir)
    act_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        i for i, run in enumerate(runs)
        if not _has_all_activations(run)
    ]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("No pending activation captures.")
        return

    groups: dict[str, list[int]] = {}
    for idx in pending:
        run = runs[idx]
        activation_model_id = model_id or local_model_id(run.get("generation_model_id"))
        if activation_model_id is None:
            print(
                f"warning: skipping {run.get('run_id', f'row-{idx}')} "
                "because it has no generation_model_id and no --model override"
            )
            continue
        groups.setdefault(activation_model_id, []).append(idx)

    if not groups:
        print("No pending activation captures with a known model.")
        return

    n_ok = 0
    n_skipped = 0
    n_errors = 0
    t_start = time.time()
    total = sum(len(group) for group in groups.values())

    for activation_model_id, group in groups.items():
        print(
            f"\nLoading local activation model {activation_model_id} on {device} "
            f"for {len(group)} runs..."
        )
        tokenizer = AutoTokenizer.from_pretrained(activation_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            activation_model_id, torch_dtype=torch.bfloat16
        ).to(device)
        model.eval()

        boundary = thinking_boundary or thinking_boundary_for_model(activation_model_id)
        if boundary is None:
            raise RuntimeError(
                f"No thinking boundary configured for {activation_model_id}. "
                "Pass --thinking-boundary or add it to pipeline/model_config.py."
            )
        boundary_ids = tokenizer(boundary, add_special_tokens=False).input_ids
        if not boundary_ids:
            raise RuntimeError(
                f"Tokenizer for {activation_model_id} cannot tokenize boundary {boundary!r}"
            )
        print(f"thinking_boundary = {boundary!r}; boundary_ids = {boundary_ids}")

        for idx in group:
            n = n_ok + n_skipped + n_errors + 1
            run = runs[idx]
            run_id = run.get("run_id", f"row-{idx}")
            record = dataset.get(run.get("question_id"))
            generated_text = _generated_text(run, boundary)
            if record is None or generated_text is None:
                n_errors += 1
                print(f"[{n}/{total}] ERROR {run_id}: missing dataset row or text")
                continue

            t0 = time.time()
            result = capture_activations_from_text(
                model=model,
                tokenizer=tokenizer,
                record=record,
                hint_type=run["hint_type"],
                generated_text=generated_text,
                boundary_ids=boundary_ids,
                boundary=boundary,
                activations_dir=act_dir,
                model_id=activation_model_id,
            )
            dt = time.time() - t0

            if result is None:
                n_skipped += 1
                print(f"[{n}/{total}] skip {run_id} (boundary not found)")
                continue

            run["n_tokens"] = result["n_tokens"]
            run["probe_positions"] = result["probe_positions"]
            run["activation_paths"] = result["activation_paths"]
            run["activation_model_id"] = activation_model_id
            run["activation_timestamp"] = result["timestamp"]
            _write_jsonl_atomic(runs_file, runs)

            n_ok += 1
            elapsed = _fmt_duration(time.time() - t_start)
            print(
                f"[{n}/{total}] {run_id} "
                f"({dt:.1f}s, {result['n_tokens']} tok) | elapsed={elapsed}",
                flush=True,
            )

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    elapsed = _fmt_duration(time.time() - t_start)
    print(
        f"\nActivation capture done: {n_ok} ok, {n_skipped} no-think, "
        f"{n_errors} errors | {elapsed}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Override activation model for all runs.")
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--runs", default="data/runs.jsonl")
    ap.add_argument("--activations-dir", default="data/activations")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many activation captures.")
    ap.add_argument("--thinking-boundary", default=None,
                    help="Override the model-family thinking/response boundary.")
    args = ap.parse_args()

    extract_activations(
        model_id=args.model,
        dataset_path=args.dataset,
        runs_path=args.runs,
        activations_dir=args.activations_dir,
        device=args.device,
        limit=args.limit,
        thinking_boundary=args.thinking_boundary,
    )


if __name__ == "__main__":
    main()
