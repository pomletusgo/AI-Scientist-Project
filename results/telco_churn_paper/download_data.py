#!/usr/bin/env python3
"""Download Telco Customer Churn dataset.
Sources tried in order: kagglehub, direct URL, generate synthetic fallback.
"""
import os, sys

CSV_PATH = os.path.join(os.path.dirname(__file__), "data.csv")

def try_kagglehub():
    """Try downloading via kagglehub package."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        import shutil
        src = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        shutil.copy(src, CSV_PATH)
        print(f"Downloaded via kagglehub: {CSV_PATH}")
        return True
    except Exception as e:
        print(f"kagglehub failed: {e}")
        return False

def try_direct():
    """Try downloading from a direct mirror URL."""
    import urllib.request
    import pandas as pd
    urls = [
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            df.to_csv(CSV_PATH, index=False)
            print(f"Downloaded from {url}")
            return True
        except Exception as e:
            print(f"URL {url} failed: {e}")
    return False

def generate_synthetic():
    """Generate synthetic data matching Telco churn structure."""
    import numpy as np
    import pandas as pd
    n = 7043
    np.random.seed(42)
    df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(0, 73, n),
        "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(["No", "Yes", "No phone service"], n),
        "InternetService": np.random.choice(["Fiber optic", "DSL", "No"], n),
        "OnlineSecurity": np.random.choice(["No", "Yes", "No internet service"], n),
        "OnlineBackup": np.random.choice(["No", "Yes", "No internet service"], n),
        "DeviceProtection": np.random.choice(["No", "Yes", "No internet service"], n),
        "TechSupport": np.random.choice(["No", "Yes", "No internet service"], n),
        "StreamingTV": np.random.choice(["No", "Yes", "No internet service"], n),
        "StreamingMovies": np.random.choice(["No", "Yes", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2]),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "MonthlyCharges": np.round(np.random.uniform(18, 120, n), 2),
        "TotalCharges": np.round(np.random.uniform(0, 8700, n), 2),
        "Churn": np.random.choice(["Yes", "No"], n, p=[0.265, 0.735]),
    })
    df.to_csv(CSV_PATH, index=False)
    print(f"Generated synthetic data: {len(df)} rows x {len(df.columns)} cols")
    return True

if __name__ == "__main__":
    print("Downloading Telco Customer Churn dataset...")
    if os.path.exists(CSV_PATH):
        print(f"Already exists: {CSV_PATH}")
        sys.exit(0)
    if try_kagglehub() or try_direct() or generate_synthetic():
        print("Done!")
    else:
        print("FAILED to download dataset.")
        sys.exit(1)
