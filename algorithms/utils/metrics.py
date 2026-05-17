"""Metric helpers shared by training and evaluation scripts."""
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
METRICS_COLUMNS = ["dataset","model","feature_type","precision_macro","recall_macro","f1_macro","precision_weighted","recall_weighted","f1_weighted","accuracy","train_time_sec","inference_time_sec","random_seed"]

def compute_classification_metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, float]:
    """Compute macro/weighted precision, recall, F1, and accuracy."""
    return {"precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0), "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0), "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0), "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0), "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0), "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0), "accuracy": accuracy_score(y_true, y_pred)}

def build_metrics_row(dataset: str, model: str, feature_type: str, metrics: dict[str, float], train_time_sec: float, inference_time_sec: float, random_seed: int) -> dict[str, Any]:
    """Build one metrics row using the standard coursework column order."""
    row = {"dataset": dataset, "model": model, "feature_type": feature_type, "train_time_sec": train_time_sec, "inference_time_sec": inference_time_sec, "random_seed": random_seed}
    row.update(metrics)
    return {column: row.get(column) for column in METRICS_COLUMNS}

def save_metrics_row(metrics_row: dict[str, Any], output_path: str | Path) -> None:
    """Save a single metrics row to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_row], columns=METRICS_COLUMNS).to_csv(path, index=False)
