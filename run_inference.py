"""Iterate over (question, hint_type) pairs; skip completed runs; append to runs.jsonl."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.hints import ALL_HINT_TYPES
from pipeline.inference import run_single, PROBE_LOCATIONS


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m{int(seconds) % 60:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m"


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
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
        model_id, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if end_think_id is None or end_think_id == tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer has no </think> token id — wrong model?")
    print(f"end_think_id = {end_think_id}")

    print("Preflight: verifying chat template and generation...")
    preflight_msg = [{"role": "user", "content": "What is 2+2? Answer with a single letter."}]
    preflight_prompt = tokenizer.apply_chat_template(
        preflight_msg, tokenize=False, add_generation_prompt=True,
    )
    if "<think>" not in preflight_prompt:
        raise RuntimeError(
            f"Chat template does not include <think> in generation prompt. "
            f"Template tail: ...{preflight_prompt[-80:]!r}"
        )
    preflight_ids = tokenizer(preflight_prompt, return_tensors="pt").input_ids.to(device)
    preflight_config = model.generation_config
    preflight_config.do_sample = True
    preflight_config.temperature = 0.6
    preflight_config.top_p = 0.95
    preflight_config.max_new_tokens = 1024
    preflight_config.top_k = 20
    preflight_config.pad_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        preflight_gen = model.generate(
            preflight_ids, generation_config=preflight_config,
        )
    preflight_new = preflight_gen[0, preflight_ids.shape[1]:]
    preflight_text = tokenizer.decode(preflight_new, skip_special_tokens=False)
    if "</think>" not in preflight_text:
        raise RuntimeError(
            f"Preflight generation did not produce </think>. "
            f"Output: {preflight_text[:200]!r}"
        )
    think_part, _, answer_part = preflight_text.partition("</think>")
    print(f"Preflight passed: think={len(think_part)} chars, "
          f"answer={answer_part.strip()[:50]!r}")

    total = len(dataset) * len(hint_types)
    done = 0
    fresh = 0
    n_ok = 0
    n_skipped_existing = 0
    n_skipped_no_think = 0
    n_errors = 0
    run_times: list[float] = []
    t_start = time.time()

    for record in dataset:
        for hint_type in hint_types:
            done += 1
            run_id = f"{record['question_id']}__{hint_type}"
            if run_id in existing:
                n_skipped_existing += 1
                continue
            if limit is not None and fresh >= limit:
                print(f"--limit {limit} reached, stopping.")
                break
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
                n_errors += 1
                print(f"[{done}/{total}] ERROR {run_id}: {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0
            if result is None:
                n_skipped_no_think += 1
                print(f"[{done}/{total}] skip {run_id} (no </think>, {dt:.0f}s)")
                continue
            with open(runs_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            existing.add(run_id)
            fresh += 1
            n_ok += 1
            run_times.append(dt)
            avg = sum(run_times) / len(run_times)
            remaining = total - done
            eta = _fmt_duration(remaining * avg) if run_times else "?"
            elapsed = _fmt_duration(time.time() - t_start)
            n_tok = result.get("n_tokens", "?")
            print(f"[{done}/{total}] {run_id} ({dt:.1f}s, {n_tok} tok) "
                  f"| avg={avg:.0f}s elapsed={elapsed} eta={eta}",
                  flush=True)
        else:
            continue
        break

    elapsed = _fmt_duration(time.time() - t_start)
    print(f"\nInference done: {n_ok} ok, {n_skipped_no_think} no-think, "
          f"{n_errors} errors, {n_skipped_existing} already done | {elapsed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
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
