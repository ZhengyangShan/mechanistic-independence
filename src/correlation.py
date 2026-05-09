"""Stage 2b: Pearson correlations between SAE activations and demographic indicators.

For each ``(category, demographic)`` pair we collect SAE feature activations at the
attribute-token position across every observed pair, then compute Pearson r against
a binary indicator that the pair's demographic equals the target value. Top-k
features by absolute correlation define the correlation-based feature set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch as t
from scipy.stats import pearsonr
from tqdm import tqdm

from .config import DEMOGRAPHIC_VALUES
from .data_utils import clear_cuda_cache
from .prompts import get_attribute_and_demographic


def _find_subsequence(big, small):
    if not small:
        return None
    for i in range(len(big) - len(small) + 1):
        if big[i:i + len(small)] == small:
            return i
    return None


def _extract_features(model, submodules, sae_dicts, prompt: str, anchor: str):
    """Return ``{submod_name: feature_vector}`` measured at the last token of ``anchor``."""
    enc = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    token_ids = enc["input_ids"][0].tolist()
    anchor_tokens = model.tokenizer.encode(anchor, add_special_tokens=False)
    start = _find_subsequence(token_ids, anchor_tokens)
    if start is None:
        return None
    anchor_pos = start + len(anchor_tokens) - 1

    saved = {}
    with model.trace(prompt, scan=False, validate=False):
        for sm in submodules:
            sae = sae_dicts[sm]
            x = sm.get_activation()
            x_tok = x[0, anchor_pos]
            f_proxy = sae(x_tok, output_features=True)
            saved[sm.name] = f_proxy.save()

    out = {}
    for sub_name, proxy in saved.items():
        data = proxy.value if hasattr(proxy, "value") else proxy
        # Some SAE wrappers return (sparse_features, reconstruction); take the sparse view.
        sparse_features = data[0] if isinstance(data, tuple) else data
        out[sub_name] = sparse_features.detach().float().cpu().numpy().tolist()
    return out


def _pearson_per_feature(X: np.ndarray, y: np.ndarray) -> list:
    corrs = []
    for j in range(X.shape[1]):
        v = X[:, j]
        if np.std(v) == 0:
            corrs.append(0.0)
        else:
            c, _ = pearsonr(v, y)
            corrs.append(0.0 if np.isnan(c) else float(c))
    return corrs


def run_correlation(
    *,
    model,
    submodules,
    sae_dicts,
    pairs_records: Sequence[dict],
    out_path: Path,
    prompt_format: str,
) -> int:
    """Stream correlation records to ``out_path`` (one record per category/target/submodule)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    for category in tqdm(DEMOGRAPHIC_VALUES, desc="Correlation: categories"):
        entries = [e for e in pairs_records if e.get("category") == category]
        targets = DEMOGRAPHIC_VALUES[category]

        for target_value in tqdm(targets, desc=f"  {category}", leave=False):
            labels: list[int] = []
            neuron_acts: dict[str, list[list[float]]] = defaultdict(list)

            for entry in entries:
                pairs = entry.get("pairs", [])
                if isinstance(pairs, dict):
                    pairs = [{"lhs": v, "rhs": k} for k, v in pairs.items()]

                for pr in pairs:
                    attr_token, demo_token = get_attribute_and_demographic(pr, prompt_format)
                    if not attr_token or not demo_token:
                        continue

                    prompt = f"{attr_token} -"
                    feats = _extract_features(model, submodules, sae_dicts, prompt, attr_token)
                    if feats is None:
                        continue

                    labels.append(1 if demo_token == target_value else 0)
                    for sub_name, vec in feats.items():
                        neuron_acts[sub_name].append(vec)

            if not labels:
                continue
            y = np.asarray(labels, dtype=np.float32)

            with out_path.open("a", encoding="utf-8") as f:
                for sub_name, acts in neuron_acts.items():
                    X = np.asarray(acts, dtype=np.float32)
                    if X.ndim == 1:
                        X = X[:, None]
                    corrs = _pearson_per_feature(X, y)
                    f.write(json.dumps({
                        "category": category,
                        "demo_token": target_value,
                        "submodule": sub_name,
                        "correlations": corrs,
                    }) + "\n")
                    written += 1

            clear_cuda_cache()

    print(f"✅ Wrote {written} correlation records → {out_path}")
    return written
