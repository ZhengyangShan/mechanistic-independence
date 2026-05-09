"""Stage 4 entry point: run the full ablation grid.

For each ``ablate_task`` × ``ablation_type`` combination, generate predictions
under SAE feature ablation and write them to a per-config subdirectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch as t

from src.ablation import materialize_ablate_files, run_ablation_eval
from src.config import CATEGORY_ORDER, SUPPORTED_MODELS, get_paths
from src.data_utils import lists_to_sets, read_json
from src.sae_loader import load_resid_saes


DEFAULT_ABLATION_TYPES = ["attribution", "correlation", "intersection", "attr_minus_corr"]


def build_within_axis_configs(
    top_attr: dict,
    top_corr: dict,
    *,
    ablation_types,
    eval_self: bool = True,
):
    """``ablate=X, eval=X`` configurations for each task and ablation type."""
    return [
        {
            "ablate_task": task,
            "type": abl_type,
            "eval_tasks": [task] if eval_self else [],
            "top_attr": top_attr,
            "top_corr": top_corr,
        }
        for task in CATEGORY_ORDER
        for abl_type in ablation_types
    ]


def build_cross_axis_configs(top_attr, top_corr, *, ablation_types):
    """Cross-task transfers (e.g. ablate Education-Profession features, evaluate Race-Profession)."""
    cross = [
        ("Education-Profession", "Race-Profession",   "edu_to_rg"),
        ("Education-Profession", "Gender-Profession", "edu_to_rg"),
        ("Race-Name",            "Race-Profession",   "rn_to_rp"),
        ("Gender-Name",          "Gender-Profession", "gn_to_gp"),
    ]
    return [
        {
            "ablate_task": src,
            "type":        f"{abl_type}_{tag}",
            "eval_tasks":  [tgt],
            "top_attr":    top_attr,
            "top_corr":    top_corr,
        }
        for src, tgt, tag in cross
        for abl_type in ablation_types
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAE ablation experiments")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--end-layer", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--include-cross-axis", action="store_true",
                        help="Also run cross-task transfer configurations")
    parser.add_argument("--ablation-types", nargs="+", default=DEFAULT_ABLATION_TYPES)
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    run_dir = paths.run_dir(args.model, args.prompt_format)

    top_attr = lists_to_sets(read_json(run_dir / "att_category" / "top100_attr_indices.json"))
    top_corr = lists_to_sets(read_json(run_dir / "corr_category" / "top100_corr_indices.json"))

    from nnsight import LanguageModel
    spec = SUPPORTED_MODELS[args.model]
    print(f"Loading {spec['hf_id']} via NNsight ...")
    model = LanguageModel(
        spec["hf_id"],
        dispatch=True,
        attn_implementation="eager",
        device_map="cuda",
        torch_dtype=t.bfloat16,
    )
    tokenizer = model.tokenizer
    model._model.generation_config.pad_token_id = tokenizer.eos_token_id
    model._model.generation_config.eos_token_id = tokenizer.eos_token_id

    submodules, sae_dicts = load_resid_saes(
        model,
        args.model,
        end_layer=args.end_layer or spec["n_layers"],
        stride=args.stride,
        dtype=model.dtype,
        device=t.device("cpu"),
    )

    configs = build_within_axis_configs(top_attr, top_corr, ablation_types=args.ablation_types)
    if args.include_cross_axis:
        configs += build_cross_axis_configs(top_attr, top_corr, ablation_types=args.ablation_types)

    out_dir = run_dir / "ablation"
    for cfg in configs:
        if not cfg["eval_tasks"]:
            continue
        run_ablation_eval(
            model=model, tokenizer=tokenizer,
            submodules=submodules, sae_dicts=sae_dicts,
            config=cfg,
            prompts_dir=paths.prompts_dir,
            out_dir=out_dir,
            prompt_format=args.prompt_format,
            max_new_tokens=args.max_new_tokens,
        )

    for abl_type in args.ablation_types:
        materialize_ablate_files(out_dir, abl_type)


if __name__ == "__main__":
    main()
