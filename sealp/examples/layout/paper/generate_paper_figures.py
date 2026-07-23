"""Generate publication figures for main_robio_revised.tex."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "MLP": "#6C757D",
    "DeepSets": "#4C78A8",
    "DynEdge": "#E45756",
    "Global": "#72B7B2",
}


def fig_offline_comparison() -> None:
    models = ["MLP", "DeepSets", "DynEdge"]
    pr_auc = [0.471, 0.555, 0.578]
    p_at_10 = [0.600, 0.900, 0.800]
    composite = [0.318, 0.410, 0.390]

    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - width, pr_auc, width, label="PR-AUC", color="#4C78A8")
    ax.bar(x, p_at_10, width, label="Precision@10", color="#F58518")
    ax.bar(x + width, composite, width, label="Composite", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Validation metric")
    ax.set_title("Offline ranking quality (stratified split, seed 0)")
    ax.legend(loc="upper left", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.savefig(FIG_DIR / "fig_offline_comparison.pdf")
    plt.close(fig)


def fig_online_feasible_rate() -> None:
    methods = ["Global", "MLP", "DeepSets", "DynEdge"]
    feasible = [18, 33, 33, 31]
    rates = [f / 64.0 for f in feasible]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(methods, rates, color=[COLORS[m] for m in methods], width=0.62)
    ax.set_ylim(0.0, 0.7)
    ax.set_ylabel("L2-feasible hit rate")
    ax.set_title("Matched-budget online search (B=64 exact L2 evaluations)")
    for bar, val, cnt in zip(bars, rates, feasible):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{cnt}/64",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.savefig(FIG_DIR / "fig_online_feasible_rate.pdf")
    plt.close(fig)


def fig_best_at_eval() -> None:
    methods = ["DynEdge", "MLP", "Global"]
    best_at = [1, 9, 22]
    best_score = [0.5705, 0.5705, 0.5729]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bars = ax.barh(methods, best_at, color=[COLORS[m] for m in methods], height=0.55)
    ax.invert_xaxis()
    ax.set_xlabel("Evaluation index when best validated layout is found")
    ax.set_title("Search efficiency toward the optimal validated layout")
    ax.set_xlim(0, 30)
    for bar, score in zip(bars, best_score):
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"score={score:.4f}",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.savefig(FIG_DIR / "fig_best_at_eval.pdf")
    plt.close(fig)


def fig_training_pr_auc() -> None:
    history_path = (
        ROOT.parents[3]
        / "checkpoints"
        / "layout_models_repro"
        / "dynaseqrel_dynedge_edge_mlp_v2"
        / "stratified"
        / "seed0"
        / "training_history.csv"
    )
    if not history_path.exists():
        return
    df = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(df["epoch"], df["pr_auc"], color=COLORS["DynEdge"], linewidth=1.8, label="DynEdge PR-AUC")
    best_idx = int(df["pr_auc"].idxmax())
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val = float(df.loc[best_idx, "pr_auc"])
    ax.scatter([best_epoch], [best_val], color="#1D3557", s=36, zorder=3)
    ax.annotate(
        f"best={best_val:.3f} @ epoch {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 8, best_val - 0.04),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=9,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("DynEdge training curve (stratified validation)")
    ax.grid(alpha=0.25, linestyle="--")
    fig.savefig(FIG_DIR / "fig_training_pr_auc.pdf")
    plt.close(fig)


def fig_score_components() -> None:
    components = {
        "Grasp": 0.836,
        "Manipulability": 0.303,
        "Transport dist.": 0.542,
        "Rotation pref.": 0.721,
        "Spatial layout": 1.000,
    }
    labels = list(components.keys())
    values = list(components.values())

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.barh(labels, values, color="#457B9D", height=0.55)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Normalized component score")
    ax.set_title("Best validated layout quality decomposition (DynEdge search)")
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.savefig(FIG_DIR / "fig_score_components.pdf")
    plt.close(fig)


def main() -> None:
    fig_offline_comparison()
    fig_online_feasible_rate()
    fig_best_at_eval()
    fig_training_pr_auc()
    fig_score_components()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
