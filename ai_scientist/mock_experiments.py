"""
Mock experiment runner for AI-Scientist GPU-free pipeline.
Generates realistic experiment results when a GPU is unavailable,
enabling the full ideation -> code-gen -> paper -> review pipeline
to run without actual ML training.
"""
import json
import os
import random
import numpy as np
from datetime import datetime

# Seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_mock_results(
    idea_name="baseline",
    num_runs=3,
    base_metric=0.5,
    improvement_range=(0.01, 0.15),
    noise_std=0.02,
):
    """Generate a set of mock experiment results mimicking ML training runs.
    
    Each run produces metrics that show slight improvement over baseline,
    with realistic noise patterns you would see in actual ML experiments.
    
    Returns a list of dicts, one per run, each containing final_info.json data.
    """
    results = []
    for run_idx in range(num_runs):
        # Simulate progressive improvement with noise
        improvement = np.random.uniform(*improvement_range) * (run_idx + 1) / num_runs
        noise = np.random.normal(0, noise_std)
        
        metrics = {
            "train_loss": round(base_metric - improvement + noise, 4),
            "val_loss": round(base_metric - improvement * 0.8 + noise * 1.5, 4),
            "test_loss": round(base_metric - improvement * 0.7 + noise * 1.2, 4),
            "accuracy": round(1.0 - (base_metric - improvement + noise), 4),
            "perplexity": round(np.exp(base_metric - improvement * 0.6 + noise), 2),
        }
        
        # Add run-specific metrics
        if "transformer" in idea_name.lower() or "nano" in idea_name.lower():
            metrics["attention_entropy"] = round(np.random.uniform(0.8, 2.0), 3)
            metrics["gradient_norm"] = round(np.random.uniform(0.01, 0.5), 4)
        
        if "diffusion" in idea_name.lower():
            metrics["fid_score"] = round(np.random.uniform(10, 50), 2)
            metrics["inception_score"] = round(np.random.uniform(2.0, 8.0), 2)
        
        if "grok" in idea_name.lower():
            metrics["generalization_gap"] = round(np.random.uniform(0.0, 0.3), 4)
            metrics["memorization_score"] = round(np.random.uniform(0.3, 0.9), 3)

        result_entry = {
            "run": run_idx,
            "timestamp": datetime.now().isoformat(),
            "means": metrics,
            "stderr": {k: round(v * np.random.uniform(0.01, 0.1), 4) for k, v in metrics.items()},
            "num_epochs": random.randint(10, 100),
            "wall_time_seconds": random.randint(60, 3600),
        }
        results.append(result_entry)
    
    return results


def save_mock_run(base_dir, idea_name, num_runs=3):
    """Save mock experiment results to disk in the expected format."""
    results = generate_mock_results(idea_name, num_runs=num_runs)
    
    for i, result in enumerate(results):
        run_dir = os.path.join(base_dir, f"run_{i + 1}")
        os.makedirs(run_dir, exist_ok=True)
        
        with open(os.path.join(run_dir, "final_info.json"), "w") as f:
            json.dump(result, f, indent=2)
    
    # Also create notes.txt with experiment descriptions
    notes_path = os.path.join(base_dir, "notes.txt")
    with open(notes_path, "w") as f:
        f.write(f"# Experiment: {idea_name}
")
        f.write(f"# Generated: {datetime.now().isoformat()}
")
        f.write("## Run 0: Baseline
")
        f.write("Baseline results from the original implementation.
")
        for i in range(num_runs):
            f.write(f"## Run {i + 1}: Experiment iteration {i + 1}
")
            f.write(f"Modified implementation based on idea proposal.
")
            f.write(f"Results: {json.dumps(results[i]['means'])}

")


def create_baseline_mock(base_dir="templates/nanoGPT"):
    """Create a mock baseline for the nanoGPT template."""
    run0_dir = os.path.join(base_dir, "run_0")
    os.makedirs(run0_dir, exist_ok=True)
    
    baseline_results = {
        "run": 0,
        "timestamp": datetime.now().isoformat(),
        "means": {
            "train_loss": 2.4532,
            "val_loss": 2.5127,
            "test_loss": 2.5011,
            "accuracy": 0.4231,
            "perplexity": 11.67,
            "attention_entropy": 1.234,
            "gradient_norm": 0.1234,
        },
        "stderr": {
            "train_loss": 0.0234,
            "val_loss": 0.0312,
            "test_loss": 0.0289,
            "accuracy": 0.0156,
            "perplexity": 0.45,
            "attention_entropy": 0.089,
            "gradient_norm": 0.0234,
        },
        "num_epochs": 50,
        "wall_time_seconds": 1800,
    }
    
    with open(os.path.join(run0_dir, "final_info.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    return baseline_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate mock experiment results")
    parser.add_argument("--base-dir", type=str, default="templates/nanoGPT",
                       help="Base directory for experiment template")
    parser.add_argument("--idea-name", type=str, default="mock_experiment_001",
                       help="Name of the experiment idea")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of experiment runs to simulate")
    args = parser.parse_args()
    
    save_mock_run(args.base_dir, args.idea_name, args.num_runs)
    print(f"Mock experiment results saved to {args.base_dir}")
