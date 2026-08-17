#!/usr/bin/env python3
"""Generic Tabular ML Experiment — GPU Training Script for AI-Scientist.

Usage:
    python experiment.py --data my_dataset.csv --out_dir run_0
    python experiment.py --data my_dataset.csv --out_dir run_1

The script auto-detects classification vs regression, trains an MLP,
and saves results + charts.
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
HIDDEN_SIZES = [256, 128, 64]
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev = in_features
        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h_size
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Load & Preprocess
# ============================================================
def load_data(csv_path, target_col=None):
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Drop ID-like columns (single unique value per row or named 'customerID', 'id')
    for drop_col in ['customerID', 'customerid', 'id', 'ID']:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
            print(f"  Dropped ID column: {drop_col}")

    # Separate numeric vs non-numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Encode non-numeric columns
    for col in non_numeric:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Target detection: prefer --target arg, then non-numeric binary cols, then last col
    if target_col is None:
        # Prefer binary non-numeric columns (most likely the label)
        binary_candidates = [c for c in non_numeric if df[c].nunique() == 2]
        if binary_candidates:
            target_col = binary_candidates[-1]  # last binary categorical column
        else:
            # Prefer last non-numeric column, else last column overall
            target_col = non_numeric[-1] if non_numeric else df.columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values

    # Determine task type
    unique_y = len(np.unique(y_raw))
    if unique_y <= 10 and unique_y >= 2:
        task = "classification"
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        out_features = len(le.classes_)
    else:
        task = "regression"
        y = y_raw.astype(np.float32)
        out_features = 1

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (
        torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test),
        torch.LongTensor(y_train) if task == "classification" else torch.FloatTensor(y_train).unsqueeze(1),
        torch.LongTensor(y_val) if task == "classification" else torch.FloatTensor(y_val).unsqueeze(1),
        torch.LongTensor(y_test) if task == "classification" else torch.FloatTensor(y_test).unsqueeze(1),
        task, out_features, len(feature_cols)
    )


# ============================================================
# Training
# ============================================================
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience):
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if task_type == "classification":
                pred = out.argmax(1)
                train_correct += (pred == batch_y).sum().item()
                train_total += batch_y.size(0)

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        if task_type == "classification":
            history["train_acc"].append(train_correct / train_total)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                val_loss += loss.item()
                if task_type == "classification":
                    pred = out.argmax(1)
                    val_correct += (pred == batch_y).sum().item()
                    val_total += batch_y.size(0)

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        if task_type == "classification":
            history["val_acc"].append(val_correct / val_total)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history


# ============================================================
# Evaluate
# ============================================================
def evaluate(model, test_loader):
    model.eval()
    model.load_state_dict(torch.load("best_model.pt"))
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            out = model(batch_x)
            if task_type == "classification":
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
            else:
                all_preds.extend(out.cpu().numpy().flatten())
                all_labels.extend(batch_y.numpy().flatten())

    if task_type == "classification":
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        return {"accuracy": round(float(acc), 4), "confusion_matrix": cm.tolist()}
    else:
        r2 = r2_score(all_labels, all_preds)
        return {"r2_score": round(float(r2), 4)}


# ============================================================
# Plot
# ============================================================
def plot_results(history, out_dir, task_type):
    os.makedirs(out_dir, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy curve (classification only)
    if task_type == "classification" and "train_acc" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
        ax.plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name (auto-detect if omitted)")
    parser.add_argument("--out_dir", type=str, default="run_0")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading: {args.data}")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, task_type, out_features, n_features = load_data(args.data)
    print(f"Task: {task_type} | Features: {n_features} | Output: {out_features}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Dataloaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = MLP(n_features, out_features, HIDDEN_SIZES, DROPOUT).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Train
    start_time = time.time()
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, PATIENCE)
    train_time = time.time() - start_time

    # Evaluate
    metrics = evaluate(model, test_loader)
    print(f"Metrics: {metrics}")
    print(f"Train time: {train_time:.1f}s")

    # Plot
    plot_results(history, args.out_dir, task_type)

    # Save results in AI-Scientist expected format (only {dataset: {means: {...}}})
    dataset_name = os.path.basename(args.data).replace(".csv", "")
    result = {
        dataset_name: {
            "means": {
                "best_val_loss_mean": round(min(history["val_loss"]), 4),
                "final_train_loss_mean": round(history["train_loss"][-1], 4),
                "total_train_time_mean": round(train_time, 1),
                "avg_inference_tokens_per_second_mean": 0.0,
            }
        }
    }
    # Store extra metadata in a separate file
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"_task": task_type, "_accuracy": metrics.get("accuracy", metrics.get("r2_score", 0))}, f)
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.out_dir}/final_info.json")
