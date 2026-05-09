"""Stage 1 entry point: generate (attribute, demographic) pairs with the target LM.

Example:
    python scripts/01_generate_pairs.py \\
        --model gemma-2-9b --prompt-format demoR \\
        --data-root ./data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SUPPORTED_MODELS, get_paths
from src.pair_generation import load_lm, run_pair_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demographic-attribute word pairs")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None,
                        help="Project data root (defaults to $MMI_DATA_ROOT or ./data)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    out_dir = paths.run_dir(args.model, args.prompt_format) / "pairs"

    model, tokenizer, device = load_lm(args.model, device=args.device)
    run_pair_generation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts_dir=paths.prompts_dir,
        out_dir=out_dir,
        prompt_format=args.prompt_format,
    )


if __name__ == "__main__":
    main()
