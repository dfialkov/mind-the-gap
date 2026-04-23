"""Train per-layer logistic-regression probes for CoT->answer suppression."""
import argparse
import json

from pipeline.probes import load_examples, train_per_layer_probes


def train_probes(
    dataset_path: str = "data/dataset.jsonl",
    runs_path: str = "data/runs.jsonl",
    labels_path: str = "data/labels.jsonl",
    results_out: str = "data/probe_results.json",
    seed: int = 103,
    C: float = 1.0,
) -> None:
    examples, baselines = load_examples(dataset_path, runs_path, labels_path)
    results = train_per_layer_probes(examples, baselines, seed=seed, C=C)

    with open(results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--runs", default="data/runs.jsonl")
    ap.add_argument("--labels", default="data/labels.jsonl")
    ap.add_argument("--results-out", default="data/probe_results.json")
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    train_probes(
        dataset_path=args.dataset,
        runs_path=args.runs,
        labels_path=args.labels,
        results_out=args.results_out,
        seed=args.seed,
        C=args.C,
    )


if __name__ == "__main__":
    main()
