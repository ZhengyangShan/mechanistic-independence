"""WinoGender ablation: re-run the A/B coreference task under SAE feature ablations.

Reuses the per-step ablation primitives from :mod:`src.ablation`. The features
to ablate come from the top-K sets discovered on the synthetic word-pair tasks
(typically the gender-name attribution set), so a drop in WinoGender accuracy
under intervention provides external evidence that the localized features carry
gender-stereotype information beyond our synthetic prompts.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict

import torch as t
from datasets import load_dataset
from tqdm.auto import tqdm

from ..ablation import generate_with_ablation
from ..feature_aggregation import get_ablation_indices
from .baseline import _parse_answer, build_prompt


MAX_NEW_TOKENS = 3
MAX_INPUT_LENGTH = 256


def run_ablation(
    *,
    model,
    tokenizer,
    submodules,
    sae_dicts,
    out_dir: Path,
    ablation_specs: Dict[str, str],
    top_attr: dict,
    top_corr: dict,
    ablate_task: str = "Gender-Name",
    seed: int = 42,
) -> None:
    """Run WinoGender under each ``ablation_specs`` configuration.

    ``ablation_specs`` maps a label that goes into output filenames to one of the
    canonical ablation types (``attribution``, ``correlation``, ``intersection``,
    ``attr_minus_corr``, ...).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.manual_seed(seed)

    dataset = load_dataset("oskarvanderwal/winogender", split="train")

    for label, ablation_type in ablation_specs.items():
        ablation_indices = get_ablation_indices(
            ablate_task=ablate_task,
            ablation_type=ablation_type,
            top_attr=top_attr,
            top_corr=top_corr,
        )
        n_idx = sum(len(v) for v in ablation_indices.values())
        if n_idx == 0:
            print(f"⚠️  No indices for {ablation_type}; skipping")
            continue
        print(f"\n=== Ablation: {label} ({ablation_type}) — {n_idx} features ===")

        pred_path = out_dir / f"predictions__{label}.jsonl"
        if pred_path.exists():
            pred_path.unlink()

        counts: Counter = Counter()
        correct: Counter = Counter()
        t0 = time.time()

        for row in tqdm(dataset, desc=f"WinoGender {label}"):
            prompt = build_prompt(row)
            ans = generate_with_ablation(
                prompt, model, tokenizer, submodules, sae_dicts,
                ablation_indices, max_new_tokens=MAX_NEW_TOKENS,
            )
            pred = _parse_answer(ans)
            gender = row.get("pronoun_class", row.get("gender", "unknown"))
            counts[gender] += 1
            true_label = row.get("answer", "A")
            if pred == true_label:
                correct[gender] += 1
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": row.get("sentid", row.get("id")),
                    "gender": gender,
                    "pred": pred,
                    "answer": true_label,
                    "raw": ans,
                    "ablation": label,
                }) + "\n")

        accuracy = {g: correct[g] / counts[g] for g in counts if counts[g] > 0}
        accuracy["overall"] = sum(correct.values()) / max(sum(counts.values()), 1)
        with (out_dir / f"summary__{label}.json").open("w", encoding="utf-8") as f:
            json.dump({
                "ablation_type": ablation_type,
                "accuracy": accuracy,
                "n_features_ablated": n_idx,
                "elapsed_sec": time.time() - t0,
            }, f, indent=2)
        print(f"  accuracy = {json.dumps(accuracy)}")
