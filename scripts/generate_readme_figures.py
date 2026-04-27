"""Generate README figures from local probe artifacts."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.paths import project_paths
from pipeline.probes import DEFAULT_PROBE_HINT_TYPES


PROJECT = "qwen35_27b_full"
SEED = 103
C = 1.0
README_IMAGE_DIR = Path("docs/images")

LOCATION_LABELS = {
    "answer_first": "answer first",
    "think_last": "CoT last",
    "think_first": "CoT first",
    "think_mid": "CoT middle",
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def activation_path(run: dict, probe_location: str) -> str | None:
    return (run.get("activation_paths") or {}).get(probe_location)


def has_activation(path: str | None) -> bool:
    return path is not None and Path(path).exists()


def balanced_subset(mask: np.ndarray, y_all: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    idxs = np.flatnonzero(mask)
    by_class = [idxs[y_all[idxs] == cls] for cls in np.unique(y_all[idxs])]
    if len(by_class) < 2:
        return idxs
    n_min = min(len(cls_idxs) for cls_idxs in by_class)
    keep = np.concatenate([
        rng.choice(cls_idxs, size=n_min, replace=False)
        for cls_idxs in by_class
    ])
    rng.shuffle(keep)
    return keep


def tpr_at_control_fpr(
    positive_scores: np.ndarray,
    control_scores: np.ndarray,
    fpr_cap: float,
) -> tuple[float, float, float]:
    y_true = np.concatenate([
        np.zeros(len(control_scores), dtype=int),
        np.ones(len(positive_scores), dtype=int),
    ])
    y_score = np.concatenate([control_scores, positive_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    valid = np.flatnonzero(fpr <= fpr_cap)
    best_tpr = tpr[valid].max()
    best_candidates = valid[tpr[valid] == best_tpr]
    best = best_candidates[np.argmax(fpr[best_candidates])]
    return float(tpr[best]), float(fpr[best]), float(thresholds[best])


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#d1d5db", alpha=0.65, linewidth=0.8)
    ax.set_axisbelow(True)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def make_probe_stability_plot() -> None:
    rows = [
        {
            "location": "answer_first",
            "mean": 0.731,
            "p10": 0.599,
            "median": 0.740,
            "p90": 0.845,
        },
        {
            "location": "think_last",
            "mean": 0.713,
            "p10": 0.545,
            "median": 0.743,
            "p90": 0.844,
        },
        {
            "location": "think_first",
            "mean": 0.407,
            "p10": 0.315,
            "median": 0.405,
            "p90": 0.524,
        },
        {
            "location": "think_mid",
            "mean": 0.321,
            "p10": 0.200,
            "median": 0.326,
            "p90": 0.459,
        },
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(rows))
    means = np.array([row["mean"] for row in rows])
    p10 = np.array([row["p10"] for row in rows])
    p90 = np.array([row["p90"] for row in rows])
    colors = ["#2563eb", "#2563eb", "#93c5fd", "#93c5fd"]

    ax.barh(y, means, color=colors, edgecolor="#1f2937", linewidth=0.6)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - p10, p90 - means]),
        fmt="none",
        ecolor="#111827",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
    )
    ax.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1.1)
    ax.text(
        0.505,
        -0.62,
        "50%",
        color="#4b5563",
        fontsize=9,
        va="center",
    )

    for idx, row in enumerate(rows):
        ax.text(
            row["mean"] + 0.025,
            idx,
            f"{row['mean']:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#111827",
            fontweight="bold" if idx < 2 else "normal",
        )

    ax.set_yticks(y, [LOCATION_LABELS[row["location"]] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Mean TPR at <=20% super-honest FPR")
    fig.suptitle(
        "Paired-difference probe stability",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.90,
        "Fixed layer held constant over 30 random question splits; whiskers show p10-p90.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    savefig(fig, README_IMAGE_DIR / "probe_stability.png")


def load_delta_examples_and_controls(probe_location: str) -> tuple[list[dict], list[dict]]:
    paths = project_paths(PROJECT)
    dataset_records = load_jsonl(paths.dataset)
    runs_by_id = {run["run_id"]: run for run in load_jsonl(paths.runs)}
    labels_by_id = {label["run_id"]: label for label in load_jsonl(paths.labels)}
    dataset_by_qid = {record["question_id"]: record for record in dataset_records}
    allowed_hint_types = set(DEFAULT_PROBE_HINT_TYPES)

    baseline_by_qid: dict[str, dict] = {}
    for run_id, run in runs_by_id.items():
        if run["hint_type"] != "none" or run_id not in labels_by_id:
            continue
        path = activation_path(run, probe_location)
        if not has_activation(path):
            continue
        baseline_by_qid[run["question_id"]] = {
            "run_id": run_id,
            "question_id": run["question_id"],
            "answer": labels_by_id[run_id]["answer"],
            "activation_path": path,
        }

    examples: list[dict] = []
    super_honest_controls: list[dict] = []
    for run_id, run in runs_by_id.items():
        if run["hint_type"] == "none" or run["hint_type"] not in allowed_hint_types:
            continue

        label = labels_by_id.get(run_id)
        if label is None:
            continue

        qid = run["question_id"]
        baseline = baseline_by_qid.get(qid)
        if baseline is None:
            continue

        path = activation_path(run, probe_location)
        if not has_activation(path):
            continue

        target = dataset_by_qid[qid]["target"]
        baseline_answer = baseline["answer"]
        hinted_answer = label["answer"]

        if hinted_answer == target and hinted_answer != baseline_answer:
            if not label["hint_acknowledged_in_cot"]:
                continue
            examples.append({
                "run_id": run_id,
                "question_id": qid,
                "activation_path": path,
                "baseline_activation_path": baseline["activation_path"],
                "label": int(not label["hint_acknowledged_in_answer"]),
            })
            continue

        if hinted_answer == baseline_answer and label["hint_acknowledged_in_answer"]:
            super_honest_controls.append({
                "run_id": run_id,
                "question_id": qid,
                "activation_path": path,
                "baseline_activation_path": baseline["activation_path"],
            })

    return examples, super_honest_controls


def load_delta_stack(records: list[dict]) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}

    def load_activation(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = torch.load(path, weights_only=True).float().numpy()
        return cache[path]

    return np.stack([
        load_activation(record["activation_path"])
        - load_activation(record["baseline_activation_path"])
        for record in records
    ])


def score_answer_first_controls() -> dict[str, np.ndarray | float]:
    probe_location = "answer_first"
    layer = 2
    examples, super_honest_controls = load_delta_examples_and_controls(probe_location)

    X_delta = load_delta_stack(examples)
    y_all = np.array([example["label"] for example in examples])
    qids = np.array([example["question_id"] for example in examples])
    X_control = load_delta_stack(super_honest_controls)
    control_qids = np.array([control["question_id"] for control in super_honest_controls])

    rng = np.random.RandomState(SEED)
    unique_qids = np.unique(qids)
    rng.shuffle(unique_qids)
    n_train = max(1, int(len(unique_qids) * 0.75))
    train_qids = set(unique_qids[:n_train])

    train_mask = np.array([qid in train_qids for qid in qids])
    test_mask = ~train_mask
    train_idxs = balanced_subset(train_mask, y_all, rng)
    heldout_suppression_idxs = np.flatnonzero(test_mask & (y_all == 1))
    heldout_faithful_idxs = np.flatnonzero(test_mask & (y_all == 0))
    control_heldout_mask = np.array([
        qid not in train_qids for qid in control_qids
    ])

    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=SEED,
        class_weight="balanced",
    )
    clf.fit(X_delta[train_idxs, layer, :], y_all[train_idxs])

    suppression_scores = clf.decision_function(
        X_delta[heldout_suppression_idxs, layer, :]
    )
    faithful_scores = clf.decision_function(
        X_delta[heldout_faithful_idxs, layer, :]
    )
    super_honest_scores = clf.decision_function(
        X_control[control_heldout_mask, layer, :]
    )
    tpr, fpr, threshold = tpr_at_control_fpr(
        suppression_scores,
        super_honest_scores,
        0.20,
    )

    print(
        "answer_first layer 2 counts:",
        {
            "train_balanced": int(len(train_idxs)),
            "suppression": int(len(suppression_scores)),
            "faithful": int(len(faithful_scores)),
            "super_honest": int(len(super_honest_scores)),
            "class_balance": dict(Counter(y_all.tolist())),
        },
    )
    print(
        "answer_first layer 2 threshold:",
        {"tpr": round(tpr, 3), "fpr": round(fpr, 3), "threshold": round(threshold, 3)},
    )

    return {
        "suppression": suppression_scores,
        "faithful": faithful_scores,
        "super_honest": super_honest_scores,
        "threshold": threshold,
        "tpr": tpr,
        "fpr": fpr,
    }


def make_super_honest_score_plot() -> None:
    scores = score_answer_first_controls()
    groups = [
        ("Hint suppression", scores["suppression"], "#dc2626"),
        ("Faithful influenced", scores["faithful"], "#2563eb"),
        ("Super-honest controls", scores["super_honest"], "#16a34a"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    positions = np.arange(len(groups))[::-1]
    rng = np.random.RandomState(7)

    box = ax.boxplot(
        [group[1] for group in groups],
        positions=positions,
        vert=False,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.3},
        whiskerprops={"color": "#4b5563"},
        capprops={"color": "#4b5563"},
    )
    for patch, (_, _, color) in zip(box["boxes"], groups):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    for position, (_, values, color) in zip(positions, groups):
        jitter = rng.normal(0, 0.045, size=len(values))
        alpha = 0.42 if len(values) < 100 else 0.26
        ax.scatter(
            values,
            np.full(len(values), position) + jitter,
            s=18,
            color=color,
            alpha=alpha,
            linewidths=0,
        )

    threshold = float(scores["threshold"])
    ax.axvline(threshold, color="#7f1d1d", linestyle="--", linewidth=1.4)
    ax.text(
        threshold + 0.035,
        positions[0] + 0.45,
        (
            f"threshold at <=20% SH-FPR\n"
            f"TPR={scores['tpr']:.3f}, FPR={scores['fpr']:.3f}"
        ),
        fontsize=9,
        color="#7f1d1d",
        va="top",
    )

    ax.set_yticks(
        positions,
        [f"{name}\n(n={len(values)})" for name, values, _ in groups],
    )
    ax.set_xlabel("Probe decision score (higher = more suppression-like)")
    fig.suptitle(
        "Super-honest controls as a false-positive check",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.90,
        "Scores are from the paired-difference answer_first probe at fixed layer 2.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    savefig(fig, README_IMAGE_DIR / "super_honest_scores.png")


def main() -> None:
    make_probe_stability_plot()
    make_super_honest_score_plot()


if __name__ == "__main__":
    main()
