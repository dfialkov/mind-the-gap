"""Generate answer traces through Hugging Face Inference Providers."""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, get_token

from pipeline.hints import ALL_HINT_TYPES, build_user_message
from pipeline.model_config import thinking_boundary_for_model
from pipeline.paths import add_project_arg, project_paths, resolve_project_paths

DEFAULT_PATHS = project_paths()


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _existing_run_ids(path: Path) -> set[str]:
    return {row["run_id"] for row in _load_jsonl(path) if "run_id" in row}


def _split_thinking_response(text: str, boundary: str | None) -> tuple[str, str]:
    if boundary is None:
        return "", text
    thinking, sep, response = text.partition(boundary)
    if not sep:
        return "", text
    return thinking, response


def _message_parts(message, boundary: str | None) -> tuple[str, str]:
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return str(reasoning), content
    return _split_thinking_response(content, boundary)


def _split_model_provider(model_id: str, provider: str | None) -> tuple[str, str | None]:
    if provider is not None or "://" in model_id or ":" not in model_id:
        return model_id, provider
    model, parsed_provider = model_id.rsplit(":", 1)
    return model, parsed_provider


def _api_key(explicit: str | None) -> str:
    key = (
        explicit
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
        or get_token()
    )
    if not key:
        raise RuntimeError(
            "Set HF_TOKEN or HUGGINGFACE_API_KEY, or pass --api-key."
        )
    return key


def generate_answers(
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:nscale",
    provider: str | None = None,
    endpoint_url: str | None = None,
    hint_types: list[str] | None = None,
    dataset_path: str = str(DEFAULT_PATHS.dataset),
    runs_out: str = str(DEFAULT_PATHS.runs),
    api_key: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    concurrency: int = 8,
    limit: int | None = None,
    thinking_boundary: str | None = None,
) -> None:
    load_dotenv()

    if hint_types is None:
        hint_types = list(ALL_HINT_TYPES)

    dataset = _load_jsonl(Path(dataset_path))
    runs_path = Path(runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_run_ids(runs_path)

    pending = []
    for record in dataset:
        for hint_type in hint_types:
            run_id = f"{record['question_id']}__{hint_type}"
            if run_id in existing:
                continue
            pending.append((record, hint_type, run_id))
            if limit is not None and len(pending) >= limit:
                break
        if limit is not None and len(pending) >= limit:
            break

    if not pending:
        print("No pending answer generations.")
        return

    hf_model_id, hf_provider = _split_model_provider(model_id, provider)
    if endpoint_url is not None and hf_provider is not None:
        raise ValueError(
            "--endpoint-url cannot be combined with --provider or a ':provider' "
            "model suffix."
        )
    boundary = thinking_boundary or thinking_boundary_for_model(hf_model_id)
    if endpoint_url is not None:
        client = InferenceClient(base_url=endpoint_url, token=_api_key(api_key))
    else:
        client = InferenceClient(
            model=hf_model_id,
            provider=hf_provider,
            token=_api_key(api_key),
        )
    print(
        f"Generating {len(pending)} runs via Hugging Face "
        f"(model={hf_model_id}, "
        f"{'endpoint=' + endpoint_url if endpoint_url else 'provider=' + (hf_provider or 'auto')}, "
        f"concurrency={concurrency})..."
    )

    def call_api(item):
        record, hint_type, run_id = item
        messages = [{"role": "user", "content": build_user_message(record, hint_type)}]
        response = client.chat_completion(
            messages=messages,
            model=hf_model_id if endpoint_url is not None else None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        thinking, final_response = _message_parts(
            response.choices[0].message, boundary
        )
        return {
            "run_id": run_id,
            "question_id": record["question_id"],
            "hint_type": hint_type,
            "correct_answer": record["correct"],
            "target_answer": record["target"],
            "correct_text": record["choices"][record["correct"]],
            "target_text": record["choices"][record["target"]],
            "thinking": thinking,
            "response": final_response,
            "thinking_boundary": boundary,
            "n_tokens": getattr(response.usage, "completion_tokens", None),
            "generation_model_id": hf_model_id,
            "generation_provider": hf_provider,
            "generation_endpoint_url": endpoint_url,
            "generation_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    n_ok = 0
    n_errors = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call_api, item): item[2] for item in pending}
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                n_errors += 1
                print(f"ERROR {run_id}: {type(e).__name__}: {e}", flush=True)
                continue

            with open(runs_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            n_ok += 1
            if n_ok % 10 == 0 or n_ok == len(pending):
                elapsed = _fmt_duration(time.time() - t_start)
                print(f"  {n_ok}/{len(pending)} generated | elapsed={elapsed}",
                      flush=True)

    elapsed = _fmt_duration(time.time() - t_start)
    print(f"\nGeneration done: {n_ok} ok, {n_errors} errors | {elapsed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:nscale")
    ap.add_argument("--provider", default=None,
                    help="HF inference provider. Overrides a ':provider' model suffix.")
    ap.add_argument("--endpoint-url", default=None,
                    help="Dedicated HF Inference Endpoint/OpenAI-compatible base URL.")
    ap.add_argument("--hint-types", nargs="+", default=list(ALL_HINT_TYPES))
    add_project_arg(ap)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--runs-out", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many fresh answer generations.")
    ap.add_argument("--thinking-boundary", default=None,
                    help="Override the model-family thinking/response boundary.")
    args = ap.parse_args()
    paths = resolve_project_paths(
        ap,
        args,
        [("dataset", "--dataset"), ("runs_out", "--runs-out")],
    )

    generate_answers(
        model_id=args.model,
        provider=args.provider,
        endpoint_url=args.endpoint_url,
        hint_types=args.hint_types,
        dataset_path=args.dataset or str(paths.dataset),
        runs_out=args.runs_out or str(paths.runs),
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        concurrency=args.concurrency,
        limit=args.limit,
        thinking_boundary=args.thinking_boundary,
    )


if __name__ == "__main__":
    main()
