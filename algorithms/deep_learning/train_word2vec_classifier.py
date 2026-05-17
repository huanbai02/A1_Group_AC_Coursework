"""Command-line interface template for Word2Vec-based classifier.

This initialization-stage file defines the required interface only. It must not
train a model or save fake predictions/metrics until the assigned group member
implements the algorithm.
"""
import argparse
from pathlib import Path
MODEL_NAME = "word2vec"
PREDICTION_COLUMNS = ["id", "text", "true_label", "predicted_label"]
METRICS_COLUMNS = ["dataset", "model", "feature_type", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted", "accuracy", "train_time_sec", "inference_time_sec", "random_seed"]

def parse_args() -> argparse.Namespace:
    """Parse the standard model training arguments."""
    parser = argparse.ArgumentParser(description="Interface template for Word2Vec-based classifier.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--train_path", required=True, help="Path to train.csv.")
    parser.add_argument("--val_path", required=True, help="Path to val.csv.")
    parser.add_argument("--test_path", required=True, help="Path to test.csv.")
    parser.add_argument("--output_dir", required=True, help="Base output directory, usually data/results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def validate_input_paths(args: argparse.Namespace) -> None:
    """Validate that train, validation, and test input paths exist."""
    for attr in ["train_path", "val_path", "test_path"]:
        path = Path(getattr(args, attr))
        if not path.exists():
            raise FileNotFoundError(f"Required input path does not exist: {path}")

def describe_expected_outputs(args: argparse.Namespace) -> None:
    """Print the standard output paths for the future implementation."""
    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions" / f"{args.dataset_name}_{MODEL_NAME}_predictions.csv"
    metrics_path = output_dir / "metrics" / f"{args.dataset_name}_{MODEL_NAME}_metrics.csv"
    print("Interface only: no model is trained in this initialization-stage script.")
    print(f"Future prediction CSV: {prediction_path}")
    print(f"Prediction columns: {PREDICTION_COLUMNS}")
    print(f"Future metrics CSV: {metrics_path}")
    print(f"Metrics columns: {METRICS_COLUMNS}")

def main() -> None:
    """Validate the interface and stop before any model training."""
    args = parse_args()
    validate_input_paths(args)
    describe_expected_outputs(args)
    # TODO: Group member B should implement Word2Vec embeddings + document vector pooling + classifier here.
    raise NotImplementedError("Word2Vec-based classifier training is not implemented in the initialization stage.")
if __name__ == "__main__": main()
