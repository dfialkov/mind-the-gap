"""End-to-end pipeline: build dataset -> generate answers -> activations -> judges."""
import argparse

from build_dataset import build_dataset_file
from extract_activations import extract_activations
from generate_answers import generate_answers
from run_judges import run_judges
from train_probes import train_probes

DEFAULT_GENERATION_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:nscale"


def _kw(**kwargs):
    """Drop keys whose value is None so downstream defaults take effect."""
    return {k: v for k, v in kwargs.items() if v is not None}


def run_pipeline(
    n_mmlu: int | None = None,
    n_gpqa: int | None = None,
    seed: int | None = None,
    dataset_path: str | None = None,
    generation_model_id: str | None = None,
    activation_model_id: str | None = None,
    hint_types: list[str] | None = None,
    runs_out: str | None = None,
    activations_dir: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    concurrency: int | None = None,
    thinking_boundary: str | None = None,
    device: str | None = None,
    labels_out: str | None = None,
    generation_limit: int | None = None,
    activation_limit: int | None = None,
    judge_limit: int | None = None,
    results_out: str | None = None,
    probe_C: float | None = None,
) -> None:
    print("=== Stage 1/5: build dataset ===")
    build_dataset_file(**_kw(
        n_mmlu=n_mmlu, n_gpqa=n_gpqa, seed=seed, out=dataset_path,
    ))

    generation_model = generation_model_id or DEFAULT_GENERATION_MODEL

    print("\n=== Stage 2/5: generate answers ===")
    generate_answers(**_kw(
        model_id=generation_model,
        provider=provider,
        hint_types=hint_types,
        dataset_path=dataset_path,
        runs_out=runs_out,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        concurrency=concurrency,
        limit=generation_limit,
        thinking_boundary=thinking_boundary,
    ))

    print("\n=== Stage 3/5: extract activations ===")
    extract_activations(**_kw(
        model_id=activation_model_id,
        dataset_path=dataset_path,
        runs_path=runs_out,
        activations_dir=activations_dir,
        device=device,
        limit=activation_limit,
        thinking_boundary=thinking_boundary,
    ))

    print("\n=== Stage 4/5: run judges ===")
    run_judges(**_kw(
        runs_path=runs_out,
        dataset_path=dataset_path,
        labels_out=labels_out,
        limit=judge_limit,
    ))

    print("\n=== Stage 5/5: train probes ===")
    train_probes(**_kw(
        dataset_path=dataset_path,
        runs_path=runs_out,
        labels_path=labels_out,
        results_out=results_out,
        seed=seed,
        C=probe_C,
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mmlu", type=int, default=None)
    ap.add_argument("--n-gpqa", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--generation-model", default=None)
    ap.add_argument("--activation-model", default=None)
    ap.add_argument("--hint-types", nargs="+", default=None)
    ap.add_argument("--runs-out", default=None)
    ap.add_argument("--activations-dir", default=None)
    ap.add_argument("--provider", default=None,
                    help="HF inference provider. Overrides a ':provider' model suffix.")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--thinking-boundary", default=None,
                    help="Override the model-family thinking/response boundary.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--generation-limit", type=int, default=None,
                    help="Stop after this many fresh answer generations.")
    ap.add_argument("--activation-limit", type=int, default=None,
                    help="Stop after this many activation captures.")
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--judge-limit", type=int, default=None,
                    help="Stop judging after this many fresh judgements.")
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--probe-C", type=float, default=None)
    args = ap.parse_args()

    run_pipeline(
        n_mmlu=args.n_mmlu,
        n_gpqa=args.n_gpqa,
        seed=args.seed,
        dataset_path=args.dataset,
        generation_model_id=args.generation_model,
        activation_model_id=args.activation_model,
        hint_types=args.hint_types,
        runs_out=args.runs_out,
        activations_dir=args.activations_dir,
        provider=args.provider,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        concurrency=args.concurrency,
        thinking_boundary=args.thinking_boundary,
        device=args.device,
        labels_out=args.labels_out,
        generation_limit=args.generation_limit,
        activation_limit=args.activation_limit,
        judge_limit=args.judge_limit,
        results_out=args.results_out,
        probe_C=args.probe_C,
    )


if __name__ == "__main__":
    main()
