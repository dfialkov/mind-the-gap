"""Train per-layer logistic-regression probes for CoT->answer suppression."""
import argparse
import json
from pathlib import Path

from pipeline.inference import PROBE_LOCATIONS
from pipeline.paths import add_project_arg, project_paths, resolve_project_paths
from pipeline.probes import (
    DEFAULT_PROBE_HINT_TYPES,
    load_examples,
    train_per_layer_probes,
)

DEFAULT_PATHS = project_paths()


def train_probes(
    dataset_path: str = str(DEFAULT_PATHS.dataset),
    runs_path: str = str(DEFAULT_PATHS.runs),
    labels_path: str = str(DEFAULT_PATHS.labels),
    results_out: str = str(DEFAULT_PATHS.probe_results),
    seed: int = 103,
    C: float = 1.0,
    hint_types: tuple[str, ...] | list[str] | None = DEFAULT_PROBE_HINT_TYPES,
) -> None:
    all_results = {}
    for loc in PROBE_LOCATIONS:
        print(f"\n=== Probe location: {loc} ===")
        examples, baselines = load_examples(
            dataset_path, runs_path, labels_path, probe_location=loc,
            hint_types=hint_types,
        )
        result = train_per_layer_probes(
            examples, baselines, seed=seed, C=C,
        )
        result["probe_hint_types"] = list(hint_types) if hint_types is not None else None
        all_results[loc] = result

    results_path = Path(results_out)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {results_path}")


def main():
    ap = argparse.ArgumentParser()
    add_project_arg(ap)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument(
        "--probe-hint-types",
        nargs="+",
        default=list(DEFAULT_PROBE_HINT_TYPES),
        help="Hint types to include in probe examples. Use 'all' to include every non-baseline hint.",
    )
    args = ap.parse_args()
    paths = resolve_project_paths(
        ap,
        args,
        [
            ("dataset", "--dataset"),
            ("runs", "--runs"),
            ("labels", "--labels"),
            ("results_out", "--results-out"),
        ],
    )
    hint_types = None if args.probe_hint_types == ["all"] else args.probe_hint_types

    train_probes(
        dataset_path=args.dataset or str(paths.dataset),
        runs_path=args.runs or str(paths.runs),
        labels_path=args.labels or str(paths.labels),
        results_out=args.results_out or str(paths.probe_results),
        seed=args.seed,
        C=args.C,
        hint_types=hint_types,
    )


if __name__ == "__main__":
    main()
