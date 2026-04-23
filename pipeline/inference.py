"""One-run inference: generate, locate probe position, capture activations."""
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from pipeline.hints import build_user_message


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


def _capture_activations(model, full_ids: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Second forward pass; forward hooks capture one position per layer, cast bf16.

    Returns tensor of shape (n_hidden_states, hidden_size) where n_hidden_states
    = 1 (embedding) + num_decoder_layers.
    """
    num_layers = len(model.model.layers)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[idx] = h[0, target_idx, :].detach().to("cpu", dtype=torch.bfloat16)
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

    return torch.stack([captured[i] for i in range(num_layers + 1)], dim=0)


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

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = gen[0, prompt_len:]
    end_pos_rel, probe_pos_rel = _find_probe_position(new_ids, tokenizer, end_think_id)
    if end_pos_rel is None or probe_pos_rel is None:
        return None

    probe_target_idx = prompt_len + probe_pos_rel
    end_think_idx = prompt_len + end_pos_rel

    act = _capture_activations(model, gen, probe_target_idx)

    act_path = activations_dir / f"{record['question_id']}-{hint_type}.pt"
    _save_tensor_atomic(act, act_path)

    generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)
    thinking, _, response = generated_text.partition("</think>")

    return {
        "run_id": f"{record['question_id']}__{hint_type}",
        "question_id": record["question_id"],
        "hint_type": hint_type,
        "thinking": thinking,
        "response": response,
        "end_think_pos": end_think_idx,
        "probe_target_pos": probe_target_idx,
        "activation_path": str(act_path),
        "activation_shape": list(act.shape),
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
