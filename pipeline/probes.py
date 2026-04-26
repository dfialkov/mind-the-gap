"""Per-layer logistic-regression probes for CoT->answer suppression."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from pipeline.hints import HINTS
from pipeline.paths import project_paths

EXCLUDED_PROBE_HINT_TYPES = ("grader_hacking",)
ALL_PROBE_HINT_TYPES = tuple(
    hint_type for hint_type in HINTS if hint_type not in EXCLUDED_PROBE_HINT_TYPES
)
DEFAULT_PROBE_HINT_TYPES = ALL_PROBE_HINT_TYPES
DEFAULT_PATHS = project_paths()


def load_examples(
    dataset_path: str = str(DEFAULT_PATHS.dataset),
    runs_path: str = str(DEFAULT_PATHS.runs),
    labels_path: str = str(DEFAULT_PATHS.labels),
    probe_location: str = "answer_first",
    hint_types: tuple[str, ...] | list[str] | None = DEFAULT_PROBE_HINT_TYPES,
) -> tuple[list[dict], list[dict]]:
    """Join dataset + runs + labels; apply inclusion filter.

    Returns (examples, baselines) where examples carry a binary label
    (1 = suppression, 0 = faithful) and baselines are no-hint runs for
    the confound control.
    """
    with open(dataset_path) as f:
        dataset = {r["question_id"]: r for r in (json.loads(l) for l in f if l.strip())}

    with open(runs_path) as f:
        runs = {r["run_id"]: r for r in (json.loads(l) for l in f if l.strip())}

    with open(labels_path) as f:
        labels = {r["run_id"]: r for r in (json.loads(l) for l in f if l.strip())}

    baseline_answers: dict[str, str | None] = {}
    for run_id, run in runs.items():
        if run["hint_type"] == "none" and run_id in labels:
            baseline_answers[run["question_id"]] = labels[run_id]["answer"]

    allowed_hint_types = set(hint_types) if hint_types is not None else None

    examples: list[dict] = []
    baselines: list[dict] = []
    stats = {"total": 0, "baseline": 0, "not_influenced": 0,
             "excluded_hint_type": 0, "no_cot_ack": 0, "included": 0,
             "missing_activation": 0}

    def _activation_path(run: dict) -> str | None:
        paths = run.get("activation_paths") or {}
        return paths.get(probe_location)

    for run_id, run in runs.items():
        qid = run["question_id"]
        label = labels.get(run_id)
        if label is None:
            continue
        stats["total"] += 1

        if run["hint_type"] == "none":
            stats["baseline"] += 1
            activation_path = _activation_path(run)
            if activation_path is None:
                stats["missing_activation"] += 1
                continue
            baselines.append({
                "run_id": run_id,
                "question_id": qid,
                "hint_type": "none",
                "activation_path": activation_path,
            })
            continue

        if allowed_hint_types is not None and run["hint_type"] not in allowed_hint_types:
            stats["excluded_hint_type"] += 1
            continue

        if qid not in dataset:
            stats["not_influenced"] += 1
            continue
        target = dataset[qid]["target"]
        a_base = baseline_answers.get(qid)
        a_hint = label["answer"]

        if a_base is None or a_hint != target or a_hint == a_base:
            stats["not_influenced"] += 1
            continue

        if not label["hint_acknowledged_in_cot"]:
            stats["no_cot_ack"] += 1
            continue

        activation_path = _activation_path(run)
        if activation_path is None:
            stats["missing_activation"] += 1
            continue

        stats["included"] += 1
        examples.append({
            "run_id": run_id,
            "question_id": qid,
            "hint_type": run["hint_type"],
            "activation_path": activation_path,
            "label": int(not label["hint_acknowledged_in_answer"]),
        })

    def _has_activation(entry: dict) -> bool:
        if not Path(entry["activation_path"]).exists():
            print(f"  warning: missing activation {entry['activation_path']}, skipping")
            return False
        return True

    examples = [e for e in examples if _has_activation(e)]
    baselines = [b for b in baselines if _has_activation(b)]

    print(f"Filter stats: {stats}")
    if allowed_hint_types is not None:
        print(f"Probe hint types: {sorted(allowed_hint_types)}")
    class_counts: dict[int, int] = {}
    for ex in examples:
        class_counts[ex["label"]] = class_counts.get(ex["label"], 0) + 1
    print(f"Class balance: {class_counts}")

    return examples, baselines


def train_per_layer_probes(
    examples: list[dict],
    baselines: list[dict],
    *,
    seed: int = 103,
    C: float = 1.0,
) -> dict:
    """Train 49 per-layer logistic regression probes. Returns results dict."""
    unique_question_ids = {ex["question_id"] for ex in examples}
    if len(examples) < 4 or len(unique_question_ids) < 2:
        reason = (f"only {len(examples)} examples"
                  if len(examples) < 4
                  else f"only {len(unique_question_ids)} unique question(s)")
        print(f"{reason} — too few for train/test split. Returning empty results.")
        return {"per_layer": [], "n_examples": len(examples),
                "n_baselines": len(baselines),
                "error": f"insufficient data: {reason}"}

    X_all = []
    y_all = []
    qids = []
    for ex in examples:
        act = torch.load(ex["activation_path"], weights_only=True).float().numpy()
        X_all.append(act)
        y_all.append(ex["label"])
        qids.append(ex["question_id"])

    X_all = np.stack(X_all)
    y_all = np.array(y_all)
    qids = np.array(qids)

    X_base = []
    for b in baselines:
        act = torch.load(b["activation_path"], weights_only=True).float().numpy()
        X_base.append(act)
    X_base = np.stack(X_base) if X_base else None

    unique_qids = np.unique(qids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_qids)

    n = len(unique_qids)
    n_train = max(1, int(n * 0.75))
    train_qids = set(unique_qids[:n_train])
    test_qids = set(unique_qids[n_train:])

    train_mask = np.array([q in train_qids for q in qids])
    test_mask = np.array([q in test_qids for q in qids])

    def _balanced_subset(mask: np.ndarray) -> np.ndarray:
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

    train_idxs = _balanced_subset(train_mask)
    test_idxs = _balanced_subset(test_mask)

    n_layers = X_all.shape[1]
    results = []

    for layer in range(n_layers):
        X_train = X_all[train_idxs, layer, :]
        y_train = y_all[train_idxs]
        X_test = X_all[test_idxs, layer, :]
        y_test = y_all[test_idxs]

        if len(np.unique(y_train)) < 2:
            results.append({
                "layer": layer, "accuracy": None, "balanced_accuracy": None,
                "auc": None,
                "baseline_positive_rate": None,
                "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                "error": "single class in training set",
            })
            continue

        clf = LogisticRegression(
            C=C, max_iter=1000, random_state=seed, class_weight="balanced"
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        auc = None
        if len(np.unique(y_test)) >= 2:
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))

        baseline_pos_rate = None
        if X_base is not None:
            baseline_pred = clf.predict(X_base[:, layer, :])
            baseline_pos_rate = float(baseline_pred.mean())

        results.append({
            "layer": layer,
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "auc": auc,
            "baseline_positive_rate": baseline_pos_rate,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        })

    print(f"\n{'Layer':>5} {'Acc':>6} {'AUC':>6} {'Base+':>6} {'Ntrn':>5} {'Ntst':>5}")
    print("-" * 40)
    for r in results:
        acc_s = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "  n/a"
        auc_s = f"{r['auc']:.3f}" if r["auc"] is not None else "  n/a"
        base_s = (f"{r['baseline_positive_rate']:.3f}"
                  if r["baseline_positive_rate"] is not None else "  n/a")
        print(f"{r['layer']:>5} {acc_s:>6} {auc_s:>6} {base_s:>6} "
              f"{r['n_train']:>5} {r['n_test']:>5}")

    return {
        "per_layer": results,
        "n_examples": len(examples),
        "n_baselines": len(baselines),
        "n_train_balanced": int(len(train_idxs)),
        "n_test_balanced": int(len(test_idxs)),
        "seed": seed,
        "C": C,
        "class_weight": "balanced",
        "class_balancing": "downsampled within train/test split",
    }
