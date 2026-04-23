"""Build the evaluation dataset and write atomically to data/dataset.jsonl."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from pipeline.dataset import build_dataset


def build_dataset_file(
    n_mmlu: int = 300,
    n_gpqa: int = 198,
    seed: int = 103,
    out: str = "data/dataset.jsonl",
) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_dataset(n_mmlu, n_gpqa, seed)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=out_path.parent, delete=False, suffix=".tmp"
    )
    try:
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()
        os.replace(tmp.name, out_path)
    except Exception:
        os.unlink(tmp.name)
        raise

    print(f"Wrote {len(records)} records to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mmlu", type=int, default=300)
    ap.add_argument("--n-gpqa", type=int, default=198)
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--out", default="data/dataset.jsonl")
    args = ap.parse_args()

    build_dataset_file(
        n_mmlu=args.n_mmlu,
        n_gpqa=args.n_gpqa,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
