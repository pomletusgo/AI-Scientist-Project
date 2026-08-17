#!/usr/bin/env python3
"""Plot script for paper figures — reads run results and generates publication-quality charts."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "."
OUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = {
    "run_0": "Baseline (default data)",
    "run_1": "Baseline (repeat)",
    "run_2": "Standard MLP (linear)",
    "run_3": "Causal MLP (linear)",
    "run_4": "Linear SCM (linear)",
    "run_5": "Causal MLP (nonlinear)",
}

def load_run_info(base_dir):
    runs_loss = {}
    runs_causal = {}
    for d in sorted(os.listdir(base_dir)):
        if not d.startswith("run_"):
            continue
        info_path = os.path.join(base_dir, d, "final_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                runs_loss[d] = json.load(f)
        eval_path = os.path.join(base_dir, d, "eval_metrics.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                runs_causal[d] = json.load(f)
    return runs_loss, runs_causal

def extract_best_val_loss(runs_loss):
    vals = {}
    for run, content in runs_loss.items():
        for dataset, info in content.items():
            if isinstance(info, dict) and "means" in info:
                vals[run] = info["means"]["best_val_loss_mean"]
    return vals

def extract_ate_bias(runs_causal):
    vals = {}
    for run, m in runs_causal.items():
        if "ate_bias" in m:
            vals[run] = m["ate_bias"]
    return vals

def extract_counterfactual_mae_do1(runs_causal):
    vals = {}
    for run, m in runs_causal.items():
        if "counterfactual_mae_do1" in m:
            vals[run] = m["counterfactual_mae_do1"]
    return vals

def extract_r2(runs_causal):
    vals = {}
    for run, m in runs_causal.items():
        if "r2_score" in m:
            vals[run] = m["r2_score"]
    return vals

def plot_bar(data, metric_name, save_path):
    if not data:
        return
    sorted_runs = sorted(data.keys(), key=lambda r: int(r.split("_")[-1]) if r.split("_")[-1].isdigit() else 0)
    labels = [LABELS.get(r, r) for r in sorted_runs]
    values = [data[r] for r in sorted_runs]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("Set2")
    colors = [cmap(i / max(1, len(labels))) for i in range(len(labels))]
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f"{val:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(metric_name, fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    runs_loss, runs_causal = load_run_info(base)

    # Best validation loss
    loss_vals = extract_best_val_loss(runs_loss)
    plot_bar(loss_vals, "Best Validation Loss", os.path.join(OUT_DIR, "best_val_loss.png"))

    # R² (on synthetic tests)
    r2_vals = extract_r2(runs_causal)
    plot_bar(r2_vals, "R² Score", os.path.join(OUT_DIR, "r2_score.png"))

    # ATE bias
    bias_vals = extract_ate_bias(runs_causal)
    plot_bar(bias_vals, "ATE Bias (|pred_ATE - true_ATE|)", os.path.join(OUT_DIR, "ate_bias.png"))

    # Counterfactual MAE do1
    mae_vals = extract_counterfactual_mae_do1(runs_causal)
    plot_bar(mae_vals, "Counterfactual MAE (do T=1)", os.path.join(OUT_DIR, "mae_do1.png"))

    print("All plots saved to", OUT_DIR)
