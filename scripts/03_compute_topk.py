"""Stage 3 entry point: aggregate per-pair scores into top-K feature sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SUPPORTED_MODELS, get_paths
from src.feature_aggregation import aggregate_attribution, aggregate_correlation, save_topk


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate top-K features")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("-k", "--topk", type=int, default=100)
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    run_dir = paths.run_dir(args.model, args.prompt_format)
    attr_jsonl = run_dir / "att_category" / "attribution_records.jsonl"
    corr_jsonl = run_dir / "corr_category" / "correlation_records.jsonl"

    print(f"Aggregating attribution from {attr_jsonl}")
    attr = aggregate_attribution(attr_jsonl, k=args.topk)
    print(f"Aggregating correlation from {corr_jsonl}")
    corr = aggregate_correlation(corr_jsonl, k=args.topk)

    attr_out = run_dir / "att_category" / f"top{args.topk}_attr_indices.json"
    corr_out = run_dir / "corr_category" / f"top{args.topk}_corr_indices.json"
    save_topk(attr, corr, attr_out, corr_out)
    print(f"✅ Saved top-{args.topk} indices → {attr_out}, {corr_out}")


if __name__ == "__main__":
    main()
