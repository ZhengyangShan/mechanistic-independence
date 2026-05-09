"""Stage 4: Inference-time ablation of SAE features and re-evaluation.

Greedy decoding step-by-step, applying SAE feature ablations on the residual
stream of the *last* token before each decode step. Failures inside the NNsight
trace are caught and a direct forward pass on the underlying transformer is
used as a logits fallback so a single problematic step does not abort the run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import torch as t
from tqdm import tqdm

from .config import DEMOGRAPHIC_VALUES
from .data_utils import append_jsonl, clear_cuda_cache
from .feature_aggregation import get_ablation_indices
from .prompts import (
    build_prompt,
    extract_items_from_prompt,
    load_prompt_blocks,
    parse_pairs,
    short_hash,
    validate_orientation,
)


MAX_NEW_TOKENS_DEFAULT = 180


def _make_name_index(submodules, sae_dicts):
    return {sm.name: (sm, sae_dicts.get(sm)) for sm in submodules}


def _step_with_ablation(current_ids, attn_mask, model, tokenizer, name2pair, ablation_indices):
    """One greedy decode step that ablates the configured SAE features at the last position.

    Tries to read logits via the NNsight trace; if that fails, falls back to a
    direct ``model._model(...)`` forward to recover logits.
    """
    cur_text = tokenizer.decode(current_ids[0], skip_special_tokens=False)
    logits_saved = None

    try:
        with model.trace(cur_text, scan=False, validate=False, compile=False):
            for sub_name, (sm, sae) in name2pair.items():
                idxs = ablation_indices.get(sub_name)
                if not sae or not idxs:
                    continue

                x_full = sm.get_activation()
                if x_full is None:
                    continue

                x_last = x_full[:, -1, :].detach().clone().contiguous()
                sae_dtype = getattr(sae, "dtype", None) or x_last.dtype
                x_for_sae = x_last.cpu().to(dtype=sae_dtype)

                f = sae.encode(x_for_sae)
                f_abl = f.clone()
                idxs_list = list(map(int, idxs))
                if f_abl.ndim == 3:
                    f_abl[:, -1, idxs_list] = 0.0
                else:
                    f_abl[:, idxs_list] = 0.0

                x_hat = sae.decode(f)
                x_hat_abl = sae.decode(f_abl)
                delta = (x_hat_abl - x_hat).to(dtype=x_full.dtype)

                try:
                    delta_device = delta.to(device=x_full.device, dtype=x_full.dtype)
                except Exception:
                    delta_device = delta.to(device=current_ids.device, dtype=x_full.dtype)

                try:
                    x_full[:, -1, :].add_(delta_device)
                    sm.set_activation(x_full)
                except Exception:
                    x_new = x_full.clone()
                    x_new[:, -1, :].add_(delta_device)
                    sm.set_activation(x_new)

            logits_saved = model.output.logits
            try:
                logits_saved = logits_saved.save()
            except Exception:
                pass
    except Exception as e:
        print(f"trace error in _step_with_ablation: {e!r}")
        clear_cuda_cache()
        logits_saved = None

    logits = None
    if logits_saved is not None:
        try:
            logits = getattr(logits_saved, "value", logits_saved)
            if not t.is_tensor(logits):
                logits = t.tensor(logits, device=current_ids.device)
        except Exception:
            logits = None

    if logits is None:
        with t.no_grad():
            inp = current_ids.to(next(model._model.parameters()).device)
            kwargs = {"input_ids": inp}
            if attn_mask is not None:
                kwargs["attention_mask"] = attn_mask.to(inp.device)
            out = model._model(**kwargs, return_dict=True)
            logits = out.logits

    next_id = int(logits[0, -1].argmax().item())
    del logits, logits_saved
    clear_cuda_cache()
    return next_id


def generate_with_ablation(
    prompt: str,
    model,
    tokenizer,
    submodules,
    sae_dicts,
    ablation_indices: Dict[str, set],
    *,
    max_new_tokens: int,
) -> str:
    """Greedy decode under SAE feature ablation."""
    name2pair = _make_name_index(submodules, sae_dicts)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)
    eos_id = tokenizer.eos_token_id

    with t.no_grad():
        for step in range(max_new_tokens):
            nid = _step_with_ablation(
                input_ids, attn_mask, model, tokenizer, name2pair, ablation_indices,
            )
            next_tok = t.tensor([[nid]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = t.cat([input_ids, next_tok], dim=1)
            if attn_mask is not None:
                attn_mask = t.cat([attn_mask, t.ones_like(next_tok)], dim=1)
            if eos_id is not None and nid == eos_id:
                break
            if (step + 1) % 8 == 0:
                clear_cuda_cache()

    new_ids = input_ids[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _resume_index(success_path: Path) -> Dict[tuple, int]:
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


def run_ablation_eval(
    *,
    model,
    tokenizer,
    submodules,
    sae_dicts,
    config: dict,
    prompts_dir: Path,
    out_dir: Path,
    prompt_format: str,
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
) -> None:
    """Run a single ``(ablate_task × ablation_type → eval_tasks)`` configuration.

    Required keys in ``config``:
      * ``ablate_task``  – source task whose features are ablated
      * ``type``         – ablation type (e.g. ``attribution``, ``intersection``)
      * ``eval_tasks``   – list of tasks to evaluate under ablation
      * ``top_attr``     – nested top-K attribution dict
      * ``top_corr``     – nested top-K correlation dict
    """
    label_on_left = (prompt_format == "demoL")

    subdir = out_dir / f"{config['ablate_task']}__{config['type']}"
    subdir.mkdir(parents=True, exist_ok=True)
    success_path = subdir / f"success_pairs_{config['type']}.jsonl"
    results_path = subdir / f"results_{config['type']}.jsonl"
    summary_csv = subdir / f"summary_{config['type']}.csv"

    done_counts = _resume_index(success_path)

    ablation_indices = get_ablation_indices(
        ablate_task=config["ablate_task"],
        ablation_type=config["type"],
        top_attr=config["top_attr"],
        top_corr=config["top_corr"],
    )
    if not any(len(v) for v in ablation_indices.values()):
        print(f"⚠️  No indices to ablate for {config['ablate_task']}/{config['type']}")

    summary_rows = []
    total_saved = sum(done_counts.values())

    for category in config["eval_tasks"]:
        prompt_file = prompts_dir / f"{category.lower().replace('-', '_')}_prompts.txt"
        blocks = load_prompt_blocks(prompt_file)
        if not blocks:
            print(f"⚠️  No prompts for {category} at {prompt_file}")
            continue

        labels = DEMOGRAPHIC_VALUES[category]

        kept_sum = swapped_sum = invalid_sum = total_sum = 0
        cat_saved = cat_skipped = 0

        print(f"\n=== ablate {config['ablate_task']} ({config['type']}) → eval {category} ===")
        pbar = tqdm(blocks, desc=category, dynamic_ncols=True)
        for raw_prompt in pbar:
            items = extract_items_from_prompt(raw_prompt)
            if not items:
                items = [x.strip() for x in raw_prompt.split(",") if x.strip()]
            item_hash = short_hash(",".join(items))

            if done_counts.get((category, item_hash), 0) >= len(items):
                cat_skipped += 1
                pbar.set_postfix(saved=cat_saved, skipped=cat_skipped)
                continue

            built = build_prompt(prompt_format, labels, items)
            output = generate_with_ablation(
                built, model, tokenizer, submodules, sae_dicts, ablation_indices,
                max_new_tokens=max_new_tokens,
            )

            raw_pairs = parse_pairs(output)
            kept, total, swapped, invalid = validate_orientation(
                category, raw_pairs, items, label_on_left=label_on_left
            )
            kept_sum += len(kept); total_sum += total
            swapped_sum += swapped; invalid_sum += invalid

            append_jsonl(results_path, [{
                "category": category,
                "prompt": built,
                "output_raw": output,
                "pairs_parsed": raw_pairs,
                "stats": {
                    "lines_total": total, "kept": len(kept),
                    "swapped": swapped, "invalid": invalid,
                },
                "ablation": config["type"],
                "ablate_task": config["ablate_task"],
            }])

            if kept:
                provenance = {
                    "category": category,
                    "rhs_hash": item_hash,
                    "method": "ablation",
                    "ablation": config["type"],
                    "ablate_task": config["ablate_task"],
                }
                append_jsonl(success_path, [
                    {"category": category, "lhs": p["lhs"], "rhs": p["rhs"], "provenance": provenance}
                    for p in kept
                ])
                cat_saved += len(kept)
                total_saved += len(kept)
                done_counts[(category, item_hash)] = done_counts.get((category, item_hash), 0) + len(kept)

            clear_cuda_cache()
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
        })
        print(f"  kept {kept_sum}/{total_sum} ({keep_rate:.1%}); saved {cat_saved} pairs")

    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def materialize_ablate_files(out_dir: Path, ablation_type: str) -> None:
    """Repack per-task results into one ``ABLATE__<category>.jsonl`` per evaluated task."""
    out_dir = Path(out_dir)
    for task_dir in out_dir.glob(f"*__{ablation_type}"):
        results_file = task_dir / f"results_{ablation_type}.jsonl"
        if not results_file.exists():
            continue
        ablate_task = task_dir.name.replace(f"__{ablation_type}", "")

        by_category: dict = {}
        with results_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                by_category.setdefault(rec["category"], []).append(rec)

        for category, recs in by_category.items():
            out_path = task_dir / f"ABLATE__{category}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for rec in recs:
                    stats = rec.get("stats", {})
                    f.write(json.dumps({
                        "category": category,
                        "ablation_from": ablate_task,
                        "ablation_type": ablation_type,
                        "prompt": rec["prompt"],
                        "output_raw": rec["output_raw"],
                        "pairs": rec["pairs_parsed"],
                        "swapped": stats.get("swapped", 0),
                        "coverage": (
                            stats["kept"] / stats["lines_total"]
                            if stats.get("lines_total") else 0.0
                        ),
                    }, ensure_ascii=False) + "\n")
            print(f"  wrote {len(recs)} records → {out_path}")
