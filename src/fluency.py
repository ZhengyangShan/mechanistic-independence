"""Stage 6: Output-quality and fluency metrics for ablated generations.

Two complementary signals:
  1. Structural validity – fraction of generated pairs that parse cleanly and
     conform to the prompt (correct items, valid demographic labels, etc.).
  2. Perplexity – per-token loss of the ablated model output under a small
     reference LM (e.g. GPT-2 large), used as a fluency proxy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEMOGRAPHIC_VALUES


VALID_LABELS: Dict[str, set] = {k: set(v) for k, v in DEMOGRAPHIC_VALUES.items()}
EXPECTED_PAIRS_PER_PROMPT = 8


def load_perplexity_model(model_name: str = "gpt2-large", device: str | None = None):
    """Load a small reference LM in FP16 for fluency scoring."""
    print(f"Loading perplexity model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device or "auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def compute_perplexity(text: str, model, tokenizer, *, max_length: int = 1024) -> float:
    """Token-mean perplexity for a single sequence."""
    if not text.strip():
        return float("inf")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(model.device)
    if input_ids.shape[1] < 2:
        return float("inf")
    out = model(input_ids, labels=input_ids)
    return float(torch.exp(out.loss).item())


def compute_pair_quality(pairs: Sequence[Dict[str, str]], category: str) -> Dict[str, float]:
    """Structural validity metrics computed without an LM."""
    if not pairs:
        return {
            "num_pairs": 0,
            "validity_rate": 0.0,
            "label_diversity": 0.0,
            "avg_lhs_length": 0.0,
        }

    valid_labels = VALID_LABELS.get(category, set())
    valid_count = 0
    label_counter: Counter[str] = Counter()
    lhs_lengths: List[int] = []

    for pr in pairs:
        lhs = (pr.get("lhs") or "").strip()
        rhs = (pr.get("rhs") or "").strip()
        if lhs and rhs and rhs in valid_labels:
            valid_count += 1
        label_counter[rhs] += 1
        lhs_lengths.append(len(lhs))

    n = len(pairs)
    return {
        "num_pairs": n,
        "validity_rate": valid_count / n if n else 0.0,
        "label_diversity": len(label_counter) / max(len(valid_labels), 1),
        "avg_lhs_length": float(np.mean(lhs_lengths)) if lhs_lengths else 0.0,
    }


def evaluate_outputs(
    records: Sequence[dict],
    *,
    perplexity_model=None,
    perplexity_tokenizer=None,
) -> pd.DataFrame:
    """Score each output record on structural quality and (optionally) perplexity."""
    rows: List[dict] = []
    for rec in tqdm(records, desc="Fluency"):
        category = rec.get("category", "unknown")
        pairs = rec.get("pairs") or rec.get("pairs_parsed") or []
        quality = compute_pair_quality(pairs, category)

        if perplexity_model is not None and perplexity_tokenizer is not None:
            text = rec.get("output_raw", "")
            quality["perplexity"] = compute_perplexity(
                text, perplexity_model, perplexity_tokenizer
            )

        quality["category"] = category
        quality["ablation_type"] = rec.get("ablation_type", rec.get("ablation", "baseline"))
        quality["ablate_task"] = rec.get("ablation_from", rec.get("ablate_task", ""))
        rows.append(quality)

    return pd.DataFrame(rows)


def aggregate_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Group quality metrics by ``(ablate_task, ablation_type, category)``."""
    if df.empty:
        return df
    return (
        df.groupby(["ablate_task", "ablation_type", "category"])
          .agg({
              "num_pairs":      "sum",
              "validity_rate":  "mean",
              "label_diversity": "mean",
              "avg_lhs_length":  "mean",
              **({"perplexity": "mean"} if "perplexity" in df.columns else {}),
          })
          .reset_index()
    )
