"""Stage 3: Aggregate per-pair attribution/correlation records into top-K feature sets.

The output is a nested ``{category: {target: {submodule: set(int)}}}`` structure that
the ablation pipeline consumes directly. Set operations (intersection, non-overlap,
union) over these sets produce the four feature-localization strategies evaluated
in the paper.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data_utils import read_jsonl, sets_to_sorted_lists, write_json


def avg_topk_abs(vectors: List[np.ndarray], k: int = 100) -> set:
    """Return the indices of the top-``k`` features by mean absolute value."""
    if not vectors:
        return set()
    X = np.vstack(vectors)
    avg = np.abs(X).mean(axis=0)
    if avg.size == 0:
        return set()
    k = min(k, avg.size)
    idx = np.argpartition(avg, -k)[-k:]
    idx = idx[np.argsort(avg[idx])[::-1]]
    return set(int(i) for i in idx.tolist())


def _expand_topk_record(rec_attr: dict, length_by_sub: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Re-densify the ``topk_idx``/``topk_vals`` representation into full feature vectors."""
    out: Dict[str, np.ndarray] = {}
    for sub_name, payload in rec_attr.items():
        if isinstance(payload, list):
            out[sub_name] = np.asarray(payload, dtype=float)
            length_by_sub.setdefault(sub_name, len(payload))
            continue
        if not isinstance(payload, dict):
            continue
        length = int(payload.get("len", length_by_sub.get(sub_name, 0)))
        if length <= 0:
            continue
        length_by_sub[sub_name] = length
        v = np.zeros(length, dtype=float)
        idxs = payload.get("topk_idx", [])
        vals = payload.get("topk_vals", [])
        for i, val in zip(idxs, vals):
            if 0 <= int(i) < length:
                v[int(i)] = float(val)
        out[sub_name] = v
    return out


def aggregate_attribution(jsonl_path: str | Path, k: int = 100) -> dict:
    """Build top-K attribution feature sets from per-pair records."""
    bucket: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    length_by_sub: Dict[str, int] = {}

    for rec in read_jsonl(jsonl_path):
        cat = rec.get("category")
        tgt = rec.get("demo_token")
        att = rec.get("attribution", {})
        if cat is None or tgt is None or not isinstance(att, dict):
            continue
        densified = _expand_topk_record(att, length_by_sub)
        for sub_name, vec in densified.items():
            bucket[cat][tgt][sub_name].append(np.abs(vec))

    out: dict = defaultdict(lambda: defaultdict(dict))
    for cat, by_tgt in bucket.items():
        for tgt, by_sub in by_tgt.items():
            for sub_name, vectors in by_sub.items():
                out[cat][tgt][sub_name] = avg_topk_abs(vectors, k=k)
    return out


def aggregate_correlation(jsonl_path: str | Path, k: int = 100) -> dict:
    """Build top-K correlation feature sets from per-(target, submodule) records."""
    bucket: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for rec in read_jsonl(jsonl_path):
        cat = rec.get("category")
        tgt = rec.get("demo_token")
        sub = rec.get("submodule")
        corrs = rec.get("correlations")
        if cat is None or tgt is None or sub is None or corrs is None:
            continue
        bucket[cat][tgt][sub].append(np.asarray(corrs, dtype=float))

    out: dict = defaultdict(lambda: defaultdict(dict))
    for cat, by_tgt in bucket.items():
        for tgt, by_sub in by_tgt.items():
            for sub_name, vectors in by_sub.items():
                out[cat][tgt][sub_name] = avg_topk_abs(vectors, k=k)
    return out


def save_topk(attr_dict: dict, corr_dict: dict, attr_path: Path, corr_path: Path) -> None:
    write_json(attr_path, sets_to_sorted_lists(attr_dict))
    write_json(corr_path, sets_to_sorted_lists(corr_dict))


def get_ablation_indices(
    ablate_task: str,
    ablation_type: str,
    top_attr: dict,
    top_corr: dict,
) -> Dict[str, set]:
    """Combine per-target feature sets into one ``{submodule: indices}`` ablation mask.

    Supported ablation types (canonicalised):
      * ``attribution``        – attribution features only
      * ``correlation``        – correlation features only
      * ``intersection``       – features in both sets (causal ∩ representational)
      * ``union``              – features in either set
      * ``attr_minus_corr``    – attribution \\ correlation (non-overlapping causal)
      * ``corr_minus_attr``    – correlation \\ attribution (non-overlapping representational)
    """
    canonical = _canonical_ablation_type(ablation_type)

    by_sub_attr: Dict[str, set] = {}
    by_sub_corr: Dict[str, set] = {}
    for _, sub_dict in top_attr.get(ablate_task, {}).items():
        for sub_name, idxs in sub_dict.items():
            by_sub_attr.setdefault(sub_name, set()).update(int(i) for i in idxs)
    for _, sub_dict in top_corr.get(ablate_task, {}).items():
        for sub_name, idxs in sub_dict.items():
            by_sub_corr.setdefault(sub_name, set()).update(int(i) for i in idxs)

    out: Dict[str, set] = {}
    for sub in set(by_sub_attr) | set(by_sub_corr):
        a, c = by_sub_attr.get(sub, set()), by_sub_corr.get(sub, set())
        if canonical in ("attribution", "attr"):
            out[sub] = set(a)
        elif canonical in ("correlation", "corr"):
            out[sub] = set(c)
        elif canonical == "intersection":
            out[sub] = a & c
        elif canonical == "union":
            out[sub] = a | c
        elif canonical in ("attr_minus_corr", "nonoverlap"):
            out[sub] = a - c
        elif canonical == "corr_minus_attr":
            out[sub] = c - a
        else:
            raise ValueError(f"Unknown ablation_type: {ablation_type!r}")
    return out


def _canonical_ablation_type(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("attr_minus_corr"):
        return "attr_minus_corr"
    if s.startswith("corr_minus_attr"):
        return "corr_minus_attr"
    if s.startswith("attribution") or s == "attr":
        return "attribution"
    if s.startswith("correlation") or s == "corr":
        return "correlation"
    if s.startswith("intersection"):
        return "intersection"
    if s.startswith("union"):
        return "union"
    if s == "nonoverlap":
        return "attr_minus_corr"
    return s
