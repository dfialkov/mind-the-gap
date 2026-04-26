"""Delete all run artifacts: dataset, run records, labels, activations."""
import argparse
import shutil

from pipeline.paths import DEFAULT_PROJECT, add_project_arg, project_paths


def artifacts_for_project(project: str):
    paths = project_paths(project)
    return [
        paths.dataset,
        paths.runs,
        paths.labels,
        paths.activations_dir,
        paths.probe_results,
        paths.batch_state,
    ]


def main():
    ap = argparse.ArgumentParser()
    add_project_arg(ap)
    ap.add_argument("-y", "--yes", action="store_true",
                    help="Skip the confirmation prompt.")
    args = ap.parse_args()

    project = args.project or DEFAULT_PROJECT
    try:
        project_paths(project)
    except ValueError as e:
        ap.error(str(e))
    existing = [p for p in artifacts_for_project(project) if p.exists()]
    if not existing:
        print(f"Nothing to delete; all artifacts for project {project!r} already absent.")
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
