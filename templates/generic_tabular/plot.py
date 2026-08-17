#!/usr/bin/env python3
"""Plot script for paper figures — reads run results and generates publication-quality charts."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_run_results(base_dir):
    """Load all run results from run_0, run_1, ... directories."""
    runs = {}
    for d in sorted(os.listdir(base_dir)):
        if d.startswith("run_"):
            info_path = os.path.join(base_dir, d, "final_info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    runs[d] = json.load(f)
    return runs

def plot_comparison(runs, out_path, metric="best_val_loss"):
    """Bar chart comparing key metric across runs."""
    labels = list(runs.keys())
    values = [runs[r][metric] for r in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4472C4" if i == 0 else "#ED7D31" for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Comparison of {metric.replace('_', ' ').title()}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "comparison.png"
    runs = load_run_results(base)
    if runs:
        plot_comparison(runs, out)
    else:
        print("No run results found!")
