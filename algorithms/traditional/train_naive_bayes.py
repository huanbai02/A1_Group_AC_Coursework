"""Naive Bayes training and prediction pipeline.

This script implements the coursework baseline: TF-IDF unigram/bigram features
with Multinomial Naive Bayes. The vectorizer is fitted only on the training set,
and final metrics are computed only on the test set.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from algorithms.utils.data_io import (
    load_csv_dataset,
    save_dataframe,
    validate_required_columns,
)
from algorithms.utils.metrics import (
    build_metrics_row,
    compute_classification_metrics,
    save_metrics_row,
)
from algorithms.utils.seed import set_random_seed

MODEL_NAME = "naive_bayes"
FEATURE_TYPE = "tfidf_unigram_bigram"
REQUIRED_COLUMNS = ["id", "text", "label"]
PREDICTION_COLUMNS = ["id", "text", "true_label", "predicted_label"]


def parse_args() -> argparse.Namespace:
    """Parse the standard model training arguments."""
    parser = argparse.ArgumentParser(description="Train TF-IDF + MultinomialNB model.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--train_path", required=True, help="Path to train.csv.")
    parser.add_argument("--val_path", required=True, help="Path to val.csv.")
    parser.add_argument("--test_path", required=True, help="Path to test.csv.")
    parser.add_argument("--output_dir", required=True, help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def validate_input_paths(args: argparse.Namespace) -> None:
    """Validate that train, validation, and test input paths exist."""
    for attr in ["train_path", "val_path", "test_path"]:
        path = Path(getattr(args, attr))
        if not path.exists():
            raise FileNotFoundError(f"Required input path does not exist: {path}")


def load_model_dataset(path: str | Path) -> pd.DataFrame:
    """Load a model dataset split and validate the required columns."""
    dataframe = load_csv_dataset(path)
    validate_required_columns(dataframe, REQUIRED_COLUMNS)
    return dataframe


def build_predictions_dataframe(test_dataframe: pd.DataFrame, predictions: list[str]) -> pd.DataFrame:
    """Build the standard prediction CSV DataFrame."""
    prediction_dataframe = test_dataframe[["id", "text", "label"]].copy()
    prediction_dataframe.rename(columns={"label": "true_label"}, inplace=True)
    prediction_dataframe["predicted_label"] = predictions
    return prediction_dataframe[PREDICTION_COLUMNS]


def main() -> None:
    """Train Naive Bayes on the training split and evaluate on the test split."""
    args = parse_args()
    validate_input_paths(args)
    set_random_seed(args.seed)

    train_dataframe = load_model_dataset(args.train_path)
    load_model_dataset(args.val_path)
    test_dataframe = load_model_dataset(args.test_path)

    train_texts = train_dataframe["text"].fillna("").astype(str)
    test_texts = test_dataframe["text"].fillna("").astype(str)
    y_train = train_dataframe["label"].astype(str)
    y_true = test_dataframe["label"].astype(str).tolist()

    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2))
    model = MultinomialNB()

    start_train_time = time.time()
    x_train = vectorizer.fit_transform(train_texts)
    model.fit(x_train, y_train)
    train_time_sec = time.time() - start_train_time

    start_inference_time = time.time()
    x_test = vectorizer.transform(test_texts)
    predictions = model.predict(x_test).astype(str).tolist()
    inference_time_sec = time.time() - start_inference_time

    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions" / f"{args.dataset_name}_{MODEL_NAME}_predictions.csv"
    metrics_path = output_dir / "metrics" / f"{args.dataset_name}_{MODEL_NAME}_metrics.csv"

    save_dataframe(build_predictions_dataframe(test_dataframe, predictions), prediction_path)

    metrics = compute_classification_metrics(y_true, predictions)
    metrics_row = build_metrics_row(
        dataset=args.dataset_name,
        model=MODEL_NAME,
        feature_type=FEATURE_TYPE,
        metrics=metrics,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
        random_seed=args.seed,
    )
    save_metrics_row(metrics_row, metrics_path)

    print(f"Saved predictions to {prediction_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
