"""Stage 2 entry point: extract attribution + correlation feature scores.

Loads the LM via NNsight together with Gemma-Scope SAEs on each residual layer,
then streams attribution and correlation records to JSONL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch as t

from src.attribution import run_attribution
from src.config import SUPPORTED_MODELS, get_paths
from src.correlation import run_correlation
from src.data_utils import read_json
from src.sae_loader import load_resid_saes


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract attribution and correlation features")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stride", type=int, default=1, help="SAE layer stride (>=1)")
    parser.add_argument("--end-layer", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2, help="Integrated-gradient steps")
    parser.add_argument("--save-topk", type=int, default=100)
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--skip-correlation", action="store_true")
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    run_dir = paths.run_dir(args.model, args.prompt_format)
    pairs_path = run_dir / "pairs" / "final_results.json"
    if not pairs_path.exists():
        raise SystemExit(f"Missing pairs file: {pairs_path}. Run scripts/01_generate_pairs.py first.")

    pairs_records = read_json(pairs_path)

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
    submodules, sae_dicts = load_resid_saes(
        model,
        args.model,
        end_layer=args.end_layer or spec["n_layers"],
        stride=args.stride,
        dtype=model.dtype,
        device=t.device("cpu"),
    )

    if not args.skip_attribution:
        run_attribution(
            model=model,
            submodules=submodules,
            sae_dicts=sae_dicts,
            pairs_records=pairs_records,
            out_path=run_dir / "att_category" / "attribution_records.jsonl",
            prompt_format=args.prompt_format,
            save_topk=args.save_topk,
            steps=args.steps,
        )

    if not args.skip_correlation:
        run_correlation(
            model=model,
            submodules=submodules,
            sae_dicts=sae_dicts,
            pairs_records=pairs_records,
            out_path=run_dir / "corr_category" / "correlation_records.jsonl",
            prompt_format=args.prompt_format,
        )


if __name__ == "__main__":
    main()
