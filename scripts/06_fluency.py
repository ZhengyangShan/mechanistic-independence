"""Stage 6 entry point: structural validity + perplexity-based fluency analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SUPPORTED_MODELS, get_paths
from src.data_utils import read_jsonl
from src.fluency import aggregate_quality, evaluate_outputs, load_perplexity_model


def _gather_records(ablation_root: Path):
    records: list[dict] = []
    for run_subdir in sorted(ablation_root.iterdir()):
        if not run_subdir.is_dir():
            continue
        for results_file in run_subdir.glob("results_*.jsonl"):
            for rec in read_jsonl(results_file):
                rec.setdefault("ablation_from", run_subdir.name.split("__")[0])
                rec.setdefault("ablation_type", run_subdir.name.split("__", 1)[1] if "__" in run_subdir.name else "unknown")
                records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fluency / output-quality metrics")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-perplexity", action="store_true",
                        help="Skip perplexity scoring (faster, structural metrics only)")
    parser.add_argument("--ppl-model", default="gpt2-large")
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    run_dir = paths.run_dir(args.model, args.prompt_format)
    ablation_root = run_dir / "ablation"
    out_dir = run_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _gather_records(ablation_root)
    print(f"Collected {len(records)} ablated records")

    ppl_model = ppl_tokenizer = None
    if not args.no_perplexity:
        ppl_model, ppl_tokenizer = load_perplexity_model(args.ppl_model)

    df = evaluate_outputs(
        records,
        perplexity_model=ppl_model,
        perplexity_tokenizer=ppl_tokenizer,
    )
    df.to_csv(out_dir / "fluency_per_record.csv", index=False)

    agg = aggregate_quality(df)
    agg.to_csv(out_dir / "fluency_aggregated.csv", index=False)
    print(agg)
    print(f"\n✅ Wrote fluency outputs → {out_dir}")


if __name__ == "__main__":
    main()
