#!/usr/bin/env python3
"""Plot script for paper figures — reads run results and generates publication-quality charts."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Run labelling (must map every run folder to a human‑readable name)
# ----------------------------------------------------------------------
LABELS = {
    "run_0": "Baseline",
    "run_1": "Weighted CE",
    "run_2": "Focal (γ=2)",
    "run_3": "Focal (γ=3)",
    "run_4": "Focal (γ=1)",
    "run_5": "Focal (γ=0.5)",
}

def get_label(run_name: str) -> str:
    return LABELS.get(run_name, run_name)

# ----------------------------------------------------------------------
#  Data loading
# ----------------------------------------------------------------------
def load_runs(base_dir):
    """Load run results (means from final_info.json) and test metrics (metrics.json)."""
    run_dirs = sorted(
        [d for d in os.listdir(base_dir)
         if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))],
        key=lambda x: int(x.split("_")[-1])
    )
    means = {}
    metrics = {}
    for d in run_dirs:
        info_path = os.path.join(base_dir, d, "final_info.json")
        metrics_path = os.path.join(base_dir, d, "metrics.json")
        if not os.path.exists(info_path):
            continue
        with open(info_path) as f:
            data_info = json.load(f)
        # Expected structure: {dataset_name: {"means": {...}, "metrics": {...}}}
        if not data_info:
            continue
        dataset_name = next(iter(data_info.keys()))
        run_means = data_info[dataset_name].get("means", {})
        means[d] = run_means
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                run_metrics = json.load(f)
            metrics[d] = run_metrics
    return means, metrics

# ----------------------------------------------------------------------
#  Generic bar‑chart helper
# ----------------------------------------------------------------------
def plot_bar(values, run_labels, title, ylabel, out_path):
    """Create and save a simple bar chart."""
    if not values:
        print(f"No data for {out_path}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ind = np.arange(len(values))
    ax.bar(ind, values, color="#4472C4", edgecolor="white", linewidth=1.2)
    ax.set_xticks(ind)
    ax.set_xticklabels(run_labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

# ----------------------------------------------------------------------
#  Plot functions
# ----------------------------------------------------------------------
def plot_best_val_loss(means, out_dir):
    """Bar chart of best validation loss (mean) across runs."""
    runs = sorted(means.keys(), key=lambda x: int(x.split("_")[-1]))
    vals = []
    labels = []
    for r in runs:
        v = means[r].get("best_val_loss_mean")
        if v is not None:
            vals.append(v)
            labels.append(get_label(r))
    if vals:
        plot_bar(vals, labels, "Best Validation Loss", "Loss",
                 os.path.join(out_dir, "best_val_loss.png"))

def plot_test_metrics(metrics, out_dir):
    """For every available test metric create one bar chart."""
    if not metrics:
        return
    # Collect all possible metric keys (skip lists / dicts)
    all_keys = set()
    for run_metrics in metrics.values():
        for k in run_metrics:
            if isinstance(run_metrics[k], (list, dict)):
                continue
            all_keys.add(k)
    for mk in sorted(all_keys):
        runs = sorted(metrics.keys(), key=lambda x: int(x.split("_")[-1]))
        vals = []
        labels = []
        for r in runs:
            run_metrics = metrics.get(r, {})
            v = run_metrics.get(mk)
            if v is not None:
                vals.append(v)
                labels.append(get_label(r))
        if vals:
            ylabel = mk.replace("_", " ").title()
            if mk == "auc_roc":
                ylabel = "AUC-ROC"
            plot_bar(vals, labels, ylabel, ylabel,
                     os.path.join(out_dir, f"{mk}.png"))

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    means, metrics = load_runs(base_dir)
    plot_best_val_loss(means, out_dir)
    plot_test_metrics(metrics, out_dir)
    if not means and not metrics:
        print("No run results found!")

if __name__ == "__main__":
    main()
