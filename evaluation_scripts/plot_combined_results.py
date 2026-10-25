"""
Generate professional grouped bar charts comparing
Base Flan-T5 vs LoRA Fine-tuned model on ROUGE-L and BLEU.
Generates 3 plots:
1. Combined (MBPP + SPIDER)
2. MBPP only
3. SPIDER only
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLOT_DIR = ROOT / "evaluation_plots"

# ── Load metrics ───────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)

mbpp_base = load(RESULTS / "base_model" / "mbpp_metrics.json")
mbpp_ft   = load(RESULTS / "finetuned"  / "mbpp_metrics.json")
spider_base = load(RESULTS / "base_model" / "spider_metrics.json")
spider_ft   = load(RESULTS / "finetuned"  / "spider_metrics.json")

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#1a1a1a",
})

# Colors: muted teal for base, rich purple for fine-tuned
c_base = "#5BA6A0"
c_ft   = "#8B5CF6"

def create_grouped_bar_chart(categories, base_vals, ft_vals, title, output_filename, fig_width=10):
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    x = np.arange(len(categories))
    width = 0.32

    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base (Flan-T5)",
                       color=c_base, edgecolor="white", linewidth=0.6, zorder=3)
    bars_ft   = ax.bar(x + width / 2, ft_vals,   width, label="Fine-tuned (LoRA)",
                       color=c_ft,   edgecolor="white", linewidth=0.6, zorder=3)

    # ── Value labels ───────────────────────────────────────────────────────
    def add_labels(bars, color):
        for bar in bars:
            h = bar.get_height()
            # Dynamic offset based on max value to keep text from overlapping bar edge
            offset = max(max(base_vals), max(ft_vals)) * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{h:.2f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="600", color=color)

    add_labels(bars_base, "#3D7A75")
    add_labels(bars_ft,   "#6D3FCC")

    # ── Axes ───────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="500")
    ax.set_ylabel("Score (%)", fontsize=11.5, fontweight="600", labelpad=10)
    ax.set_ylim(0, max(max(base_vals), max(ft_vals)) * 1.2)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))

    # Light grid behind bars
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=0)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Title & legend ─────────────────────────────────────────────────────
    ax.set_title(title, fontsize=14, fontweight="700", pad=16, color="#1a1a1a")

    legend = ax.legend(loc="upper left", frameon=True, fontsize=9.5,
                       edgecolor="#DDDDDD", fancybox=True, shadow=False,
                       framealpha=0.9)
    legend.get_frame().set_facecolor("#F5F5F5")

    plt.tight_layout(pad=1.5)
    out_path = PLOT_DIR / output_filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Saved to {out_path}")
    plt.close(fig)

# 1. Combined Plot
create_grouped_bar_chart(
    categories=["MBPP\nROUGE-L", "MBPP\nBLEU", "SPIDER\nROUGE-L", "SPIDER\nBLEU"],
    base_vals=[mbpp_base["rougeL"], mbpp_base["bleu"], spider_base["rougeL"], spider_base["bleu"]],
    ft_vals=[mbpp_ft["rougeL"], mbpp_ft["bleu"], spider_ft["rougeL"], spider_ft["bleu"]],
    title="Base vs Fine-tuned: ROUGE-L & BLEU Across Tasks",
    output_filename="combined_rougeL_bleu_comparison.png",
    fig_width=10
)

# 2. MBPP Only Plot
create_grouped_bar_chart(
    categories=["ROUGE-L", "BLEU"],
    base_vals=[mbpp_base["rougeL"], mbpp_base["bleu"]],
    ft_vals=[mbpp_ft["rougeL"], mbpp_ft["bleu"]],
    title="MBPP Dataset: Base vs Fine-tuned (ROUGE-L & BLEU)",
    output_filename="mbpp_rougeL_bleu_comparison.png",
    fig_width=7
)

# 3. SPIDER Only Plot
create_grouped_bar_chart(
    categories=["ROUGE-L", "BLEU"],
    base_vals=[spider_base["rougeL"], spider_base["bleu"]],
    ft_vals=[spider_ft["rougeL"], spider_ft["bleu"]],
    title="SPIDER Dataset: Base vs Fine-tuned (ROUGE-L & BLEU)",
    output_filename="spider_rougeL_bleu_comparison.png",
    fig_width=7
)
