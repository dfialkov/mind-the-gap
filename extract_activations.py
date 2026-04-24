"""Extract activations from pre-generated model responses (phase 2 of API-based inference)."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.hints import build_user_message
from pipeline.inference import (
    _find_probe_position, _capture_activations, _save_tensor_atomic,
)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m{int(seconds) % 60:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m"


def _load_existing_run_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["run_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def extract_activations(
    model_id: str = "Qwen/QwQ-32B",
    generations_path: str = "data/generations.jsonl",
    dataset_path: str = "data/dataset.jsonl",
    runs_out: str = "data/runs.jsonl",
    activations_dir: str = "data/activations",
    device: str = "cuda",
    limit: int | None = None,
) -> None:
    with open(generations_path) as f:
        generations = [json.loads(l) for l in f if l.strip()]

    with open(dataset_path) as f:
        dataset = {
            r["question_id"]: r
            for r in (json.loads(l) for l in f if l.strip())
        }

    runs_path = Path(runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_run_ids(runs_path)

    pending = [g for g in generations if g["run_id"] not in existing]
    if limit is not None:
        pending = pending[:limit]

    if not pending:
        print("No pending activations to extract.")
        return

    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    act_dir = Path(activations_dir)
    act_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    t_start = time.time()

    for i, gen in enumerate(pending):
        record = dataset[gen["question_id"]]
        hint_type = gen["hint_type"]
        run_id = gen["run_id"]

        user_msg = build_user_message(record, hint_type)
        messages = [{"role": "user", "content": user_msg}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        generated_text = gen["generated_text"]
        if generated_text.startswith("<think>"):
            generated_text = generated_text[len("<think>"):]
            if generated_text.startswith("\n"):
                generated_text = generated_text[1:]

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        full_text = prompt_text + generated_text
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        prompt_len = len(prompt_ids)
        new_ids = full_ids[0, prompt_len:]

        end_pos_rel, answer_pos_rel = _find_probe_position(
            new_ids, tokenizer, end_think_id,
        )
        if end_pos_rel is None or answer_pos_rel is None or end_pos_rel < 1:
            n_skip += 1
            print(f"[{i+1}/{len(pending)}] skip {run_id} (no </think>)")
            continue

        think_first_rel = 0
        think_last_rel = end_pos_rel - 1
        think_mid_rel = (think_first_rel + think_last_rel) // 2

        target_indices = {
            "think_first": prompt_len + think_first_rel,
            "think_mid": prompt_len + think_mid_rel,
            "think_last": prompt_len + think_last_rel,
            "answer_first": prompt_len + answer_pos_rel,
        }

        t0 = time.time()
        with torch.no_grad():
            activations = _capture_activations(model, full_ids, target_indices)
        dt = time.time() - t0

        stem = f"{gen['question_id']}-{hint_type}"
        activation_paths = {}
        for loc, act in activations.items():
            act_path = act_dir / f"{loc}-{stem}.pt"
            _save_tensor_atomic(act, act_path)
            activation_paths[loc] = str(act_path)

        raw_text = gen["generated_text"]
        if "</think>" in raw_text:
            thinking, _, response = raw_text.partition("</think>")
            if thinking.startswith("<think>"):
                thinking = thinking[len("<think>"):]
            thinking = thinking.strip()
        else:
            thinking = raw_text
            response = ""

        result = {
            "run_id": run_id,
            "question_id": gen["question_id"],
            "hint_type": hint_type,
            "thinking": thinking,
            "response": response,
            "n_tokens": len(new_ids),
            "probe_positions": dict(target_indices),
            "activation_paths": activation_paths,
            "model_id": gen["model_id"],
            "timestamp": gen["timestamp"],
        }

        with open(runs_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        n_ok += 1

        elapsed = _fmt_duration(time.time() - t_start)
        print(f"[{i+1}/{len(pending)}] {run_id} ({dt:.1f}s, {len(new_ids)} tok) "
              f"| elapsed={elapsed}", flush=True)

    elapsed = _fmt_duration(time.time() - t_start)
    print(f"\nDone: {n_ok} ok, {n_skip} no-think | {elapsed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/QwQ-32B")
    ap.add_argument("--generations", default="data/generations.jsonl")
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--runs-out", default="data/runs.jsonl")
    ap.add_argument("--activations-dir", default="data/activations")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    extract_activations(
        model_id=args.model,
        generations_path=args.generations,
        dataset_path=args.dataset,
        runs_out=args.runs_out,
        activations_dir=args.activations_dir,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
