"""Delete all run artifacts: dataset, run records, labels, activations."""
import argparse
import shutil
from pathlib import Path

ARTIFACTS = [
    Path("data/dataset.jsonl"),
    Path("data/runs.jsonl"),
    Path("data/labels.jsonl"),
    Path("data/activations"),
    Path("data/probe_results.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yes", action="store_true",
                    help="Skip the confirmation prompt.")
    args = ap.parse_args()

    existing = [p for p in ARTIFACTS if p.exists()]
    if not existing:
        print("Nothing to delete; all artifacts already absent.")
        return

    print("Will delete:")
    for p in existing:
        kind = "dir" if p.is_dir() else "file"
        print(f"  {p} ({kind})")

    if not args.yes:
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    for p in existing:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  removed {p}")

    print("Done.")


if __name__ == "__main__":
    main()
