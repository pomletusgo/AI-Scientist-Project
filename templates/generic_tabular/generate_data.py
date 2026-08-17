#!/usr/bin/env python3
"""Generate or load a CSV dataset for the generic_tabular template.
Usage: python generate_data.py --topic "software defect prediction" --output data.csv [--rows 1000]
"""
import argparse, os, numpy as np, pandas as pd

TOPIC_PRESETS = {
    "default": lambda n: pd.DataFrame({
        **{f"feature_{i}": np.random.randn(n) * 2 + 5 for i in range(1, 9)},
        "feature_a": np.random.rand(n) * 0.8 + 0.1,
        "feature_b": np.random.exponential(3, n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3])
    }),
    "software defect prediction": lambda n: pd.DataFrame({
        "loc": np.random.lognormal(5, 1.5, n).astype(int),
        "complexity": np.random.randint(1, 30, n),
        "code_churn": np.random.randint(0, 500, n),
        "num_developers": np.random.randint(1, 15, n),
        "code_review_hours": np.random.exponential(4, n),
        "previous_bugs": np.random.poisson(2, n),
        "test_coverage": np.random.beta(2, 3, n) * 100,
        "dependency_count": np.random.randint(0, 50, n),
        "has_bug": np.random.choice([0, 1], n, p=[0.75, 0.25])
    }),
    "customer churn": lambda n: pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_charges": np.random.normal(70, 30, n),
        "total_charges": np.random.normal(3000, 2000, n),
        "contract_type": np.random.choice([0, 1, 2], n),
        "payment_method": np.random.choice([0, 1, 2, 3], n),
        "online_security": np.random.choice([0, 1, 2], n),
        "tech_support_tickets": np.random.poisson(3, n),
        "satisfaction_score": np.random.randint(1, 6, n),
        "churned": np.random.choice([0, 1], n, p=[0.7, 0.3])
    }),
    "medical diagnosis": lambda n: pd.DataFrame({
        "age": np.random.randint(18, 90, n),
        "bmi": np.random.normal(27, 6, n),
        "blood_pressure_sys": np.random.normal(130, 20, n),
        "blood_pressure_dia": np.random.normal(85, 12, n),
        "glucose": np.random.normal(100, 30, n),
        "cholesterol": np.random.normal(200, 40, n),
        "smoking_years": np.random.exponential(10, n),
        "exercise_hours_week": np.random.exponential(3, n),
        "has_disease": np.random.choice([0, 1], n, p=[0.8, 0.2])
    }),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="default")
    p.add_argument("--output", default="data.csv")
    p.add_argument("--rows", type=int, default=1000)
    args = p.parse_args()

    # Find best matching preset
    key = args.topic.lower()
    best = "default"
    for preset in TOPIC_PRESETS:
        if preset in key or key in preset:
            best = preset
            break

    df = TOPIC_PRESETS[best](args.rows)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} rows × {len(df.columns)} cols → {args.output}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]} ({df[df.columns[-1]].nunique()} classes)")
