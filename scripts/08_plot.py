"""Generate the trade-off and Δ-heatmap figures used in the paper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SUPPORTED_MODELS, get_paths
from src.data_utils import read_json
from src.plotting import plot_diff_heatmap, plot_tradeoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper figures")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    run_dir = paths.run_dir(args.model, args.prompt_format)
    eval_dir = run_dir / "eval_results"
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_S_path = eval_dir / "_all_S.json"
    if not all_S_path.exists():
        raise SystemExit(f"Missing summaries: {all_S_path}. Run scripts/05_evaluate.py first.")
    all_S = read_json(all_S_path)

    plot_tradeoff(all_S, save_path=plots_dir / "tradeoff_acc_vs_kl.png")

    for run_name, run in all_S.items():
        for category in run:
            plot_diff_heatmap(category, run, save_dir=plots_dir / run_name)

    print(f"✅ Wrote plots → {plots_dir}")


if __name__ == "__main__":
    main()
