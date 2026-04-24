"""One-run inference: generate, locate probe position, capture activations."""
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from pipeline.hints import build_user_message

PROBE_LOCATIONS = ("think_first", "think_mid", "think_last", "answer_first")


class _TimeLimitCriteria(StoppingCriteria):
    def __init__(self, max_seconds: int):
        self.max_seconds = max_seconds
        self.start = time.monotonic()

    def __call__(self, input_ids, scores, **kwargs):
        return (time.monotonic() - self.start) > self.max_seconds


def _find_probe_position(new_ids: torch.Tensor, tokenizer, end_think_id: int):
    """Return (end_think_pos, probe_target_pos). Either may be None if not found."""
    matches = (new_ids == end_think_id).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return None, None
    end_pos = matches[0].item()
    for i in range(end_pos + 1, len(new_ids)):
        if tokenizer.decode([new_ids[i].item()]).strip():
            return end_pos, i - 1
    return end_pos, None


def _capture_activations(
    model, full_ids: torch.Tensor, target_indices: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Single forward pass; capture multiple positions per layer, cast bf16.

    target_indices: {location_name: absolute_token_index}
    Returns {location_name: tensor of shape (n_hidden_states, hidden_size)}.
    """
    num_layers = len(model.model.layers)
    captured: dict[str, dict[int, torch.Tensor]] = {n: {} for n in target_indices}

    def make_hook(layer_idx: int):
        def hook(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            for name, idx in target_indices.items():
                captured[name][layer_idx] = (
                    h[0, idx, :].detach().to("cpu", dtype=torch.bfloat16)
                )
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(make_hook(0))]
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i + 1)))

    try:
        with torch.no_grad():
            model(input_ids=full_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {
        name: torch.stack([captured[name][i] for i in range(num_layers + 1)], dim=0)
        for name in target_indices
    }


def _save_tensor_atomic(tensor: torch.Tensor, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def run_single(
    *,
    model,
    tokenizer,
    record: dict,
    hint_type: str,
    end_think_id: int,
    activations_dir: Path,
    model_id: str,
    max_new_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    run_timeout: int = 600,
) -> dict | None:
    """One (question, hint) inference. Returns run dict, or None if </think> missing."""
    device = next(model.parameters()).device
    user_msg = build_user_message(record, hint_type)
    messages = [{"role": "user", "content": user_msg}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    stopping = StoppingCriteriaList([_TimeLimitCriteria(run_timeout)])
    gen_config = model.generation_config
    gen_config.do_sample = True
    gen_config.temperature = temperature
    gen_config.top_p = top_p
    gen_config.top_k = 20
    gen_config.max_new_tokens = max_new_tokens
    gen_config.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            generation_config=gen_config,
            stopping_criteria=stopping,
        )

    new_ids = gen[0, prompt_len:]
    end_pos_rel, answer_pos_rel = _find_probe_position(new_ids, tokenizer, end_think_id)
    if end_pos_rel is None or answer_pos_rel is None or end_pos_rel < 1:
        return None

    think_first_rel = 0
    think_last_rel = end_pos_rel - 1
    think_mid_rel = (think_first_rel + think_last_rel) // 2

    target_indices = {
        "think_first": prompt_len + think_first_rel,
        "think_mid": prompt_len + think_mid_rel,
        "think_last": prompt_len + think_last_rel,
        "answer_first": prompt_len + answer_pos_rel,
    }

    activations = _capture_activations(model, gen, target_indices)

    stem = f"{record['question_id']}-{hint_type}"
    activation_paths = {}
    for loc, act in activations.items():
        act_path = activations_dir / f"{loc}-{stem}.pt"
        _save_tensor_atomic(act, act_path)
        activation_paths[loc] = str(act_path)

    generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)
    thinking, _, response = generated_text.partition("</think>")

    return {
        "run_id": f"{record['question_id']}__{hint_type}",
        "question_id": record["question_id"],
        "hint_type": hint_type,
        "thinking": thinking,
        "response": response,
        "probe_positions": {k: v for k, v in target_indices.items()},
        "activation_paths": activation_paths,
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
