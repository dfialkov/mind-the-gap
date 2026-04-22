"""MMLU + GPQA Diamond loading, normalization, and target selection."""
import random

from datasets import load_dataset

LETTERS = "ABCD"

# Standard MMLU 57-subject → 4-domain grouping (STEM / humanities / social_sciences / other).
MMLU_DOMAIN = {
    # STEM
    "abstract_algebra": "stem", "astronomy": "stem", "college_biology": "stem",
    "college_chemistry": "stem", "college_computer_science": "stem",
    "college_mathematics": "stem", "college_physics": "stem", "computer_security": "stem",
    "conceptual_physics": "stem", "electrical_engineering": "stem",
    "elementary_mathematics": "stem", "high_school_biology": "stem",
    "high_school_chemistry": "stem", "high_school_computer_science": "stem",
    "high_school_mathematics": "stem", "high_school_physics": "stem",
    "high_school_statistics": "stem", "machine_learning": "stem",
    # Humanities
    "formal_logic": "humanities", "high_school_european_history": "humanities",
    "high_school_us_history": "humanities", "high_school_world_history": "humanities",
    "international_law": "humanities", "jurisprudence": "humanities",
    "logical_fallacies": "humanities", "moral_disputes": "humanities",
    "moral_scenarios": "humanities", "philosophy": "humanities",
    "prehistory": "humanities", "professional_law": "humanities",
    "world_religions": "humanities",
    # Social sciences
    "econometrics": "social_sciences", "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences", "human_sexuality": "social_sciences",
    "professional_psychology": "social_sciences", "public_relations": "social_sciences",
    "security_studies": "social_sciences", "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    # Other
    "anatomy": "other", "business_ethics": "other", "clinical_knowledge": "other",
    "college_medicine": "other", "global_facts": "other", "human_aging": "other",
    "management": "other", "marketing": "other", "medical_genetics": "other",
    "miscellaneous": "other", "nutrition": "other", "professional_accounting": "other",
    "professional_medicine": "other", "virology": "other",
}

GPQA_DOMAIN = {"Physics": "stem", "Chemistry": "stem", "Biology": "stem"}


def _proportional_allocation(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation: each key gets floor(total * share), remainder
    goes to keys with the biggest fractional parts. Sums to `total` exactly."""
    grand = sum(counts.values())
    raw = {k: total * v / grand for k, v in counts.items()}
    floors = {k: int(r) for k, r in raw.items()}
    remainder = total - sum(floors.values())
    fracs = sorted(((raw[k] - floors[k], k) for k in counts), reverse=True)
    for _, k in fracs[:remainder]:
        floors[k] += 1
    return floors


def load_mmlu(n: int, seed: int) -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject: dict[str, list[int]] = {}
    for i, subj in enumerate(ds["subject"]):
        by_subject.setdefault(subj, []).append(i)

    counts = {subj: len(idxs) for subj, idxs in by_subject.items()}
    allocation = _proportional_allocation(counts, n)

    rng = random.Random(seed)
    selected: list[int] = []
    for subj in sorted(by_subject.keys()):
        k = allocation[subj]
        if k > 0:
            selected.extend(rng.sample(by_subject[subj], k))
    selected.sort()

    records = []
    for idx in selected:
        row = ds[idx]
        records.append({
            "question_id": f"mmlu_{idx}",
            "subject": row["subject"],
            "domain": MMLU_DOMAIN[row["subject"]],
            "question": row["question"],
            "choices": {LETTERS[i]: row["choices"][i] for i in range(4)},
            "correct": LETTERS[row["answer"]],
        })
    return records


def load_gpqa(seed: int) -> list[dict]:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(seed)
    records = []
    for idx in range(len(ds)):
        row = ds[idx]
        answers = [
            (row["Correct Answer"].strip(), True),
            (row["Incorrect Answer 1"].strip(), False),
            (row["Incorrect Answer 2"].strip(), False),
            (row["Incorrect Answer 3"].strip(), False),
        ]
        rng.shuffle(answers)
        choices = {LETTERS[i]: answers[i][0] for i in range(4)}
        correct = LETTERS[next(i for i, a in enumerate(answers) if a[1])]
        subject = row.get("High-level domain", "gpqa")
        records.append({
            "question_id": f"gpqa_{idx}",
            "subject": subject,
            "domain": GPQA_DOMAIN.get(subject, "stem"),
            "question": row["Question"],
            "choices": choices,
            "correct": correct,
        })
    return records


def pick_target(correct: str, rng: random.Random) -> str:
    return rng.choice([c for c in LETTERS if c != correct])


def build_dataset(n_mmlu: int, n_gpqa: int, seed: int) -> list[dict]:
    mmlu = load_mmlu(n_mmlu, seed) if n_mmlu > 0 else []
    gpqa = load_gpqa(seed) if n_gpqa > 0 else []
    if 0 < n_gpqa < len(gpqa):
        gpqa = random.Random(seed + 1).sample(gpqa, n_gpqa)

    records = mmlu + gpqa
    target_rng = random.Random(seed + 2)
    for r in records:
        r["target"] = pick_target(r["correct"], target_rng)
    return records
