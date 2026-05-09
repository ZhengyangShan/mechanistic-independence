"""Stage 1: Generate demographic-attribute word pairs with the target LM.

For each prompt template (one per task category), the model is asked to label
a batch of names/professions with one of the allowed demographic categories.
Generations are parsed, validated for orientation, and persisted as JSONL with
checkpoint/resume support.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEMOGRAPHIC_VALUES, PROMPT_FILE_BASENAMES, SUPPORTED_MODELS
from .data_utils import append_jsonl, write_json
from .prompts import (
    build_prompt,
    extract_items_from_prompt,
    load_prompt_blocks,
    parse_pairs,
    short_hash,
    validate_orientation,
)


BASE_MAX_NEW_TOKENS = 180
TOKENS_PER_LINE = 16


def load_lm(model_key: str, device: str | torch.device | None = None):
    """Load a HuggingFace causal LM and tokenizer for greedy generation."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    spec = SUPPORTED_MODELS[model_key]
    model_id = spec["hf_id"]
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device


def generate_batch(
    prompts: Sequence[str],
    model,
    tokenizer,
    device,
    *,
    max_new_tokens: int,
    batch_size: int = 32,
) -> List[str]:
    """Greedy-decode a list of prompts and return the new-token continuations."""
    out_strs: List[str] = []
    gen_kwargs = dict(do_sample=False, pad_token_id=tokenizer.eos_token_id)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, **gen_kwargs)
        decoded = tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        out_strs.extend(decoded)
    return out_strs


def _load_resume_index(success_path: Path) -> Dict[tuple, int]:
    """Re-read prior success file to enable checkpoint/resume."""
    done: Dict[tuple, int] = {}
    if not success_path.exists():
        return done
    with success_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat = rec.get("category")
            item_hash = (rec.get("provenance") or {}).get("rhs_hash")
            if cat and item_hash:
                done[(cat, item_hash)] = done.get((cat, item_hash), 0) + 1
    return done


def run_pair_generation(
    *,
    model,
    tokenizer,
    device,
    prompts_dir: Path,
    out_dir: Path,
    prompt_format: str,
    categories: Sequence[str] = tuple(PROMPT_FILE_BASENAMES.keys()),
    base_max_new_tokens: int = BASE_MAX_NEW_TOKENS,
    tokens_per_line: int = TOKENS_PER_LINE,
) -> None:
    """Drive the full pair-generation loop for ``prompt_format`` ∈ {demoR, demoL}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_on_left = (prompt_format == "demoL")
    success_path = out_dir / "pairs.jsonl"
    results_path = out_dir / "results.jsonl"
    summary_csv = out_dir / "summary.csv"

    done_counts = _load_resume_index(success_path)

    summary_rows = []
    total_saved = sum(done_counts.values())

    for category in categories:
        prompt_path = prompts_dir / PROMPT_FILE_BASENAMES[category]
        blocks = load_prompt_blocks(prompt_path)
        if not blocks:
            print(f"⚠️  No prompts for {category} at {prompt_path}")
            continue

        labels = DEMOGRAPHIC_VALUES[category]

        kept_sum = swapped_sum = invalid_sum = total_sum = 0
        cat_saved = cat_skipped = 0

        print(f"\n=== {category} ({len(blocks)} prompts) ===")
        pbar = tqdm(blocks, desc=category, dynamic_ncols=True)
        for raw_prompt in pbar:
            items = extract_items_from_prompt(raw_prompt)
            item_hash = short_hash(",".join(items))

            if done_counts.get((category, item_hash), 0) >= len(items):
                cat_skipped += 1
                pbar.set_postfix(saved=cat_saved, skipped=cat_skipped)
                continue

            built = build_prompt(prompt_format, labels, items)
            output = generate_batch(
                [built],
                model, tokenizer, device,
                max_new_tokens=max(base_max_new_tokens, len(items) * tokens_per_line),
                batch_size=1,
            )[0]

            raw_pairs = parse_pairs(output)
            kept, total, swapped, invalid = validate_orientation(
                category, raw_pairs, items, label_on_left=label_on_left
            )

            kept_sum += len(kept)
            total_sum += total
            swapped_sum += swapped
            invalid_sum += invalid

            append_jsonl(results_path, [{
                "category": category,
                "prompt": built,
                "output_raw": output,
                "pairs_parsed": raw_pairs,
                "stats": {
                    "lines_total": total, "kept": len(kept),
                    "swapped": swapped, "invalid": invalid,
                },
            }])

            if kept:
                provenance = {"category": category, "rhs_hash": item_hash, "method": "batch"}
                append_jsonl(success_path, [
                    {"category": category, "lhs": p["lhs"], "rhs": p["rhs"], "provenance": provenance}
                    for p in kept
                ])
                cat_saved += len(kept)
                total_saved += len(kept)
                done_counts[(category, item_hash)] = done_counts.get((category, item_hash), 0) + len(kept)

            pbar.set_postfix(saved=cat_saved, skipped=cat_skipped, total=total_saved)

        keep_rate = (kept_sum / total_sum) if total_sum else 0.0
        summary_rows.append({
            "category": category,
            "lines_total": total_sum,
            "kept": kept_sum,
            "swapped": swapped_sum,
            "invalid": invalid_sum,
            "keep_rate": round(keep_rate, 3),
            "saved_pairs": cat_saved,
            "skipped_prompts": cat_skipped,
        })
        print(f"  kept {kept_sum}/{total_sum} ({keep_rate:.1%}); saved {cat_saved} pairs")

    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    merge_results_with_pairs(success_path, results_path, out_dir / "final_results.json")


def merge_results_with_pairs(
    success_path: Path, results_path: Path, final_path: Path
) -> None:
    """Combine validated pairs with their originating prompt/output for downstream stages."""
    pairs_by_key: Dict[tuple, List[Dict[str, str]]] = {}
    with success_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["category"], rec["provenance"]["rhs_hash"])
            pairs_by_key.setdefault(key, []).append({"lhs": rec["lhs"], "rhs": rec["rhs"]})

    final: List[dict] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cat = rec["category"]
            items = extract_items_from_prompt(rec["prompt"])
            key = (cat, short_hash(",".join(items)))
            if key in pairs_by_key:
                final.append({
                    "category": cat,
                    "prompt": rec["prompt"],
                    "output_raw": rec.get("output_raw", ""),
                    "pairs": pairs_by_key[key],
                })

    write_json(final_path, final)
    print(f"✅ Merged {len(final)} prompts → {final_path}")
