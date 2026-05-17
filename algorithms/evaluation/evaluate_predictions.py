"""Evaluate saved prediction CSV files with standard classification metrics."""

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from algorithms.utils.data_io import load_csv_dataset, validate_required_columns
from algorithms.utils.metrics import (
    build_metrics_row,
    compute_classification_metrics,
    save_metrics_row,
)

REQUIRED_COLUMNS = ["true_label", "predicted_label"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a prediction CSV file.")
    parser.add_argument("--prediction_path", required=True, help="Path to prediction CSV.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--model_name", required=True, help="Model identifier.")
    parser.add_argument("--feature_type", required=True, help="Feature representation name.")
    parser.add_argument("--output_path", required=True, help="Output metrics CSV path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """Load predictions, compute metrics, and save one metrics row."""
    args = parse_args()
    prediction_dataframe = load_csv_dataset(args.prediction_path)
    validate_required_columns(prediction_dataframe, REQUIRED_COLUMNS)

    metrics = compute_classification_metrics(
        prediction_dataframe["true_label"].tolist(),
        prediction_dataframe["predicted_label"].tolist(),
    )
    metrics_row = build_metrics_row(
        dataset=args.dataset_name,
        model=args.model_name,
        feature_type=args.feature_type,
        metrics=metrics,
        train_time_sec=0.0,
        inference_time_sec=0.0,
        random_seed=args.seed,
    )
    save_metrics_row(metrics_row, args.output_path)

    print(f"Saved metrics to: {args.output_path}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
