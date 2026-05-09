"""Stage 5: Compute KL-divergence and accuracy deltas between baseline and ablated runs.

For Race-Name and Gender-Name tasks, we score accuracy against ground-truth
demographic labels. For profession tasks, we measure KL divergence between the
predicted demographic distribution and a reference (uniform for race/gender;
empirical BLS distribution for education).

The reference distributions come from a CSV of names with race/gender labels and
the BLS occupation/education table cited in the paper.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DEMOGRAPHIC_VALUES


ALLOWED_LABELS: Dict[str, set] = {k: set(v) for k, v in DEMOGRAPHIC_VALUES.items()}


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """Compute KL(P || Q) over discrete distributions, with smoothing."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def normalize_to_distribution(counts: Dict[str, int], allowed: Sequence[str]) -> np.ndarray:
    total = sum(counts.get(lbl, 0) for lbl in allowed)
    if total == 0:
        return np.full(len(allowed), 1.0 / len(allowed))
    return np.asarray([counts.get(lbl, 0) / total for lbl in allowed], dtype=float)


def name_accuracy(
    pairs: Sequence[Dict[str, str]],
    name_to_demo: Dict[str, str],
    *,
    name_field: str = "lhs",
    demo_field: str = "rhs",
) -> Tuple[float, int]:
    """Accuracy of predicted demographic vs ground-truth name → demographic mapping."""
    n_correct = n_total = 0
    for pr in pairs:
        name = pr.get(name_field, "").strip()
        pred = pr.get(demo_field, "").strip()
        truth = name_to_demo.get(name)
        if not name or not pred or truth is None:
            continue
        n_total += 1
        if pred.lower() == truth.lower():
            n_correct += 1
    return (n_correct / n_total if n_total else 0.0), n_total


def profession_kl(
    pairs: Sequence[Dict[str, str]],
    category: str,
    reference: Dict[str, np.ndarray] | None = None,
    *,
    profession_field: str = "lhs",
    demo_field: str = "rhs",
) -> Tuple[float, int]:
    """Mean KL divergence per profession between predicted distribution and ``reference``.

    When ``reference`` is ``None`` (race/gender), a uniform distribution is used.
    """
    allowed = list(DEMOGRAPHIC_VALUES[category])
    by_prof: Dict[str, Counter] = defaultdict(Counter)
    for pr in pairs:
        prof = pr.get(profession_field, "").strip()
        demo = pr.get(demo_field, "").strip()
        if prof and demo in allowed:
            by_prof[prof][demo] += 1

    if not by_prof:
        return 0.0, 0

    kls = []
    uniform = np.full(len(allowed), 1.0 / len(allowed))
    for prof, counts in by_prof.items():
        p = normalize_to_distribution(counts, allowed)
        q = (reference.get(prof) if reference else None)
        if q is None:
            q = uniform
        else:
            q = np.asarray(q, dtype=float)
            q = q / q.sum() if q.sum() > 0 else uniform
        kls.append(kl_divergence(p, q))
    return float(np.mean(kls)), len(by_prof)


def relative_pct_delta(new: float, baseline: float) -> float:
    if baseline in (0, None) or new is None:
        return 0.0
    return (new - baseline) / baseline * 100.0


def load_name_to_demographic(csv_path: Path, axis: str) -> Dict[str, str]:
    """Load a name → demographic dictionary for either ``race`` or ``gender``."""
    df = pd.read_csv(csv_path)
    name_col = "name" if "name" in df.columns else df.columns[0]
    if axis == "race":
        target_col = "race"
    elif axis == "gender":
        target_col = "gender"
    else:
        raise ValueError(f"axis must be 'race' or 'gender', got {axis!r}")
    return {row[name_col]: str(row[target_col]) for _, row in df.iterrows()}


def load_education_reference(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load BLS empirical education distributions per profession."""
    df = pd.read_csv(csv_path)
    out: Dict[str, np.ndarray] = {}
    edu_cols = [c for c in df.columns if c != "profession"]
    for _, row in df.iterrows():
        prof = str(row["profession"]).strip()
        out[prof] = np.asarray([float(row[c]) for c in edu_cols], dtype=float)
    return out


def evaluate_run(
    *,
    baseline_pairs_by_category: Dict[str, List[Dict[str, str]]],
    ablated_pairs_by_category: Dict[str, List[Dict[str, str]]],
    name_to_race: Dict[str, str] | None = None,
    name_to_gender: Dict[str, str] | None = None,
    education_reference: Dict[str, np.ndarray] | None = None,
) -> Dict[str, dict]:
    """Compute the per-category metrics that populate the paper's figures."""
    summary: Dict[str, dict] = {}
    for category in DEMOGRAPHIC_VALUES:
        baseline_pairs = baseline_pairs_by_category.get(category, [])
        ablated_pairs = ablated_pairs_by_category.get(category, [])
        if not baseline_pairs and not ablated_pairs:
            continue

        entry: Dict[str, float | int] = {}
        if category == "Race-Name" and name_to_race is not None:
            base_acc, base_n = name_accuracy(baseline_pairs, name_to_race)
            abl_acc, abl_n = name_accuracy(ablated_pairs, name_to_race)
            entry.update({
                "baseline_acc": base_acc, "ablated_acc": abl_acc,
                "delta_acc_pct": relative_pct_delta(abl_acc, base_acc),
                "n_baseline": base_n, "n_ablated": abl_n,
            })
        elif category == "Gender-Name" and name_to_gender is not None:
            base_acc, base_n = name_accuracy(baseline_pairs, name_to_gender)
            abl_acc, abl_n = name_accuracy(ablated_pairs, name_to_gender)
            entry.update({
                "baseline_acc": base_acc, "ablated_acc": abl_acc,
                "delta_acc_pct": relative_pct_delta(abl_acc, base_acc),
                "n_baseline": base_n, "n_ablated": abl_n,
            })
        else:
            ref = education_reference if category == "Education-Profession" else None
            base_kl, base_n = profession_kl(baseline_pairs, category, reference=ref)
            abl_kl, abl_n = profession_kl(ablated_pairs, category, reference=ref)
            entry.update({
                "baseline_kl": base_kl, "ablated_kl": abl_kl,
                "delta_kl_pct": relative_pct_delta(abl_kl, base_kl),
                "n_baseline_professions": base_n, "n_ablated_professions": abl_n,
            })
        summary[category] = entry
    return summary


def load_pairs_by_category(jsonl_path: Path, key: str = "pairs_parsed") -> Dict[str, List[dict]]:
    """Group pairs from an ablation results file by evaluation category."""
    out: Dict[str, List[dict]] = defaultdict(list)
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["category"]].extend(rec.get(key, []))
    return out


def parse_run_name(run_name: str) -> Tuple[str, str]:
    """Split a run directory name like ``"Race-Name__attribution"`` into its parts."""
    parts = run_name.split("__", 1)
    if len(parts) != 2:
        return run_name, "unknown"
    ablate_src, abl_type = parts
    return ablate_src, abl_type.split("_")[0]
