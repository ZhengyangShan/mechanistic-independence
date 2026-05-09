"""WinoGender validation: baseline + per-ablation accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch as t

from src.config import SUPPORTED_MODELS, get_paths
from src.data_utils import lists_to_sets, read_json
from src.sae_loader import load_resid_saes
from src.winogender.ablation import run_ablation
from src.winogender.baseline import run_baseline


DEFAULT_ABLATION_SPECS = {
    "attribution":     "attribution",
    "correlation":     "correlation",
    "intersection":    "intersection",
    "attr_minus_corr": "attr_minus_corr",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WinoGender baseline and ablations")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--mode", choices=["baseline", "ablation", "both"], default="both")
    parser.add_argument("--ablate-task", default="Gender-Name")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    run_dir = paths.run_dir(args.model, args.prompt_format)
    out_dir = run_dir / "winogender"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = SUPPORTED_MODELS[args.model]

    if args.mode in ("baseline", "both"):
        run_baseline(model_name=spec["hf_id"], out_dir=out_dir / "baseline")

    if args.mode in ("ablation", "both"):
        from nnsight import LanguageModel

        top_attr = lists_to_sets(read_json(run_dir / "att_category" / "top100_attr_indices.json"))
        top_corr = lists_to_sets(read_json(run_dir / "corr_category" / "top100_corr_indices.json"))

        model = LanguageModel(
            spec["hf_id"], dispatch=True, attn_implementation="eager",
            device_map="cuda", torch_dtype=t.bfloat16,
        )
        tokenizer = model.tokenizer
        model._model.generation_config.pad_token_id = tokenizer.eos_token_id
        model._model.generation_config.eos_token_id = tokenizer.eos_token_id

        submodules, sae_dicts = load_resid_saes(
            model, args.model, end_layer=spec["n_layers"],
            stride=args.stride, dtype=model.dtype, device=t.device("cpu"),
        )

        run_ablation(
            model=model, tokenizer=tokenizer,
            submodules=submodules, sae_dicts=sae_dicts,
            out_dir=out_dir,
            ablation_specs=DEFAULT_ABLATION_SPECS,
            top_attr=top_attr, top_corr=top_corr,
            ablate_task=args.ablate_task,
        )


if __name__ == "__main__":
    main()
