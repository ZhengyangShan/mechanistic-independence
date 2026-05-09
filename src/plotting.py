"""Plotting utilities for the trade-off and heatmap figures in the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluation import parse_run_name, relative_pct_delta


SOURCE_COLORS = {
    "Gender-Name":          "#ff7f0e",
    "Gender-Profession":    "#d62728",
    "Race-Name":            "#1f77b4",
    "Race-Profession":      "#2ca02c",
    "Education-Profession": "#9467bd",
}

TYPE_MARKERS = {
    "attribution":     "s",
    "correlation":     "o",
    "attr_minus_corr": "D",
    "intersection":    "P",
}


def plot_diff_heatmap(
    category: str,
    summary: dict,
    *,
    save_dir: Path | None = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Δ-heatmap of demographic predictions per profession (ablated − baseline)."""
    if category not in summary or "heatmap" not in summary[category]:
        return
    hm = summary[category]["heatmap"]
    labels = hm["labels"]
    order = hm["profession_order"]
    base = pd.DataFrame(hm["baseline"]).T.reindex(order)[labels].fillna(0.0)
    abl = pd.DataFrame(hm["ablated"]).T.reindex(order)[labels].fillna(0.0)
    diff = abl - base

    vmax = np.max(np.abs(diff.values))
    plt.figure(figsize=figsize)
    sns.heatmap(diff, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "Δ probability"})
    plt.title(f"Δ {category}: ablated − baseline")
    plt.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / f"diff_heatmap__{category}.png", dpi=150)
    plt.close()


def plot_tradeoff(
    all_S: Dict[str, dict],
    *,
    x_metric: str = "delta_acc_pct",
    y_metric: str = "delta_kl_pct",
    save_path: Path | None = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Trade-off scatter: bias change vs control-task change for each ablation run.

    Mirrors Figures 2–3 in the paper. The bottom-right green region is the ideal
    outcome (more accurate AND less biased); the top-left red region is the worst.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for run_name, run in all_S.items():
        src, abl_type = parse_run_name(run_name)
        for category, entry in run.items():
            x = entry.get(x_metric)
            y = entry.get(y_metric)
            if x is None or y is None:
                continue
            ax.scatter(
                x, y,
                color=SOURCE_COLORS.get(src, "gray"),
                marker=TYPE_MARKERS.get(abl_type, "x"),
                s=80,
                edgecolors="black",
                linewidths=0.5,
                label=f"{src} ({abl_type})",
            )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"Δ {x_metric}")
    ax.set_ylabel(f"Δ {y_metric}")
    ax.set_title("Bias vs control-task trade-off")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen, dedup_handles, dedup_labels = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); dedup_handles.append(h); dedup_labels.append(l)
        ax.legend(dedup_handles, dedup_labels, loc="best", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.close()
