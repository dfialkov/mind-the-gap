"""End-to-end pipeline: build dataset -> run inference -> run judges."""
import argparse

from build_dataset import build_dataset_file
from run_inference import run_inference
from run_judges import run_judges
from train_probes import train_probes


def _kw(**kwargs):
    """Drop keys whose value is None so downstream defaults take effect."""
    return {k: v for k, v in kwargs.items() if v is not None}


def run_pipeline(
    n_mmlu: int | None = None,
    n_gpqa: int | None = None,
    seed: int | None = None,
    dataset_path: str | None = None,
    model_id: str | None = None,
    hint_types: list[str] | None = None,
    runs_out: str | None = None,
    activations_dir: str | None = None,
    max_new_tokens: int | None = None,
    device: str | None = None,
    labels_out: str | None = None,
    inference_limit: int | None = None,
    judge_limit: int | None = None,
    run_timeout: int | None = None,
    results_out: str | None = None,
    probe_C: float | None = None,
    backend: str | None = None,
) -> None:
    print("=== Stage 1/4: build dataset ===")
    build_dataset_file(**_kw(
        n_mmlu=n_mmlu, n_gpqa=n_gpqa, seed=seed, out=dataset_path,
    ))

    print("\n=== Stage 2/4: run inference ===")
    run_inference(**_kw(
        model_id=model_id,
        hint_types=hint_types,
        dataset_path=dataset_path,
        runs_out=runs_out,
        activations_dir=activations_dir,
        max_new_tokens=max_new_tokens,
        device=device,
        limit=inference_limit,
        run_timeout=run_timeout,
        backend=backend,
    ))

    print("\n=== Stage 3/4: run judges ===")
    run_judges(**_kw(
        runs_path=runs_out,
        dataset_path=dataset_path,
        labels_out=labels_out,
        limit=judge_limit,
    ))

    print("\n=== Stage 4/4: train probes ===")
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
    ap.add_argument("--model", default=None)
    ap.add_argument("--hint-types", nargs="+", default=None)
    ap.add_argument("--runs-out", default=None)
    ap.add_argument("--activations-dir", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--inference-limit", type=int, default=None,
                    help="Stop inference after this many fresh runs.")
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--judge-limit", type=int, default=None,
                    help="Stop judging after this many fresh judgements.")
    ap.add_argument("--run-timeout", type=int, default=None,
                    help="Skip any single inference run taking longer than this (seconds).")
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--probe-C", type=float, default=None)
    ap.add_argument("--backend", choices=["hf", "vllm"], default=None)
    args = ap.parse_args()

    run_pipeline(
        n_mmlu=args.n_mmlu,
        n_gpqa=args.n_gpqa,
        seed=args.seed,
        dataset_path=args.dataset,
        model_id=args.model,
        hint_types=args.hint_types,
        runs_out=args.runs_out,
        activations_dir=args.activations_dir,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        labels_out=args.labels_out,
        inference_limit=args.inference_limit,
        judge_limit=args.judge_limit,
        run_timeout=args.run_timeout,
        results_out=args.results_out,
        probe_C=args.probe_C,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
