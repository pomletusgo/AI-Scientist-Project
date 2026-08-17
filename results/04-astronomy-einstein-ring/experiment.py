"""
Auto-generated experiment script.
Modify and run to produce results for the paper.
"""

import json
import os
import numpy as np

def run_baseline():
    """Run baseline experiment."""
    results = {
        "method": "baseline",
        "metric_1": 0.0,
        "metric_2": 0.0,
    }
    return results

def run_proposed_method():
    """Run the proposed method experiment."""
    results = {
        "method": "proposed",
        "metric_1": 0.0,
        "metric_2": 0.0,
    }
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Run experiments
    baseline = run_baseline()
    proposed = run_proposed_method()

    # Compute improvement
    improvement = {
        "metric_1_pct": round((proposed["metric_1"] - baseline["metric_1"]) / (baseline["metric_1"] + 1e-8) * 100, 2),
        "metric_2_pct": round((proposed["metric_2"] - baseline["metric_2"]) / (baseline["metric_2"] + 1e-8) * 100, 2),
    }

    final_info = {
        "baseline": baseline,
        "proposed": proposed,
        "improvement": improvement,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    # Save results to the parent directory for paper access
    with open(os.path.join(os.path.dirname(args.out_dir) or ".", "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    print(f"Results saved to {args.out_dir}/final_info.json")

if __name__ == "__main__":
    main()
