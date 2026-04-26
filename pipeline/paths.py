"""Project-scoped artifact paths for datasets, runs, labels, and probes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROJECT = "default"
DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class ProjectPaths:
    project: str
    data_root: Path
    dataset: Path
    runs: Path
    labels: Path
    activations_dir: Path
    probe_results: Path
    batch_state: Path


def validate_project_name(project: str) -> str:
    if not project:
        raise ValueError("project name cannot be empty")
    if project in {".", ".."} or "/" in project or "\\" in project:
        raise ValueError(
            "project name must be a simple name, not a path: "
            f"{project!r}"
        )
    return project


def project_paths(
    project: str = DEFAULT_PROJECT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> ProjectPaths:
    project = validate_project_name(project)
    root = Path(data_root)
    return ProjectPaths(
        project=project,
        data_root=root,
        dataset=root / "datasets" / f"{project}.jsonl",
        runs=root / "runs" / f"{project}.jsonl",
        labels=root / "labels" / f"{project}.jsonl",
        activations_dir=root / "activations" / project,
        probe_results=root / "probe_results" / f"{project}.json",
        batch_state=root / "batch_states" / f"{project}.json",
    )


def add_project_arg(parser) -> None:
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Project name used to construct artifact paths under data/. "
            "Cannot be combined with explicit path arguments."
        ),
    )


def ensure_project_compatible(
    project: str | None,
    explicit_paths: list[tuple[str, object | None]],
) -> None:
    if project is None:
        return
    conflicts = [name for name, value in explicit_paths if value is not None]
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(
            f"--project cannot be combined with explicit path arguments: {joined}"
        )


def resolve_project_paths(parser, args, explicit_args: list[tuple[str, str]]) -> ProjectPaths:
    explicit_flags = [
        flag for attr, flag in explicit_args
        if getattr(args, attr) is not None
    ]
    if args.project is not None and explicit_flags:
        parser.error(
            "--project cannot be combined with explicit path arguments: "
            + ", ".join(explicit_flags)
        )
    try:
        return project_paths(args.project or DEFAULT_PROJECT)
    except ValueError as e:
        parser.error(str(e))
