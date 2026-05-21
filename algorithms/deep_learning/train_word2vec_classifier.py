"""Word2Vec-based document classifier.

The pipeline trains Word2Vec embeddings only on the training split, averages
in-vocabulary token vectors into document vectors, and trains Logistic
Regression for final classification.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

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
from algorithms.utils.text_processing import simple_tokenize

MODEL_NAME = "word2vec"
FEATURE_TYPE = "word2vec_avg_logreg"
REQUIRED_COLUMNS = ["id", "text", "label"]
PREDICTION_COLUMNS = ["id", "text", "true_label", "predicted_label"]


def parse_args() -> argparse.Namespace:
    """Parse standard model training arguments and Word2Vec options."""
    parser = argparse.ArgumentParser(description="Train Word2Vec average-vector classifier.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--train_path", required=True, help="Path to train.csv.")
    parser.add_argument("--val_path", required=True, help="Path to val.csv.")
    parser.add_argument("--test_path", required=True, help="Path to test.csv.")
    parser.add_argument("--output_dir", required=True, help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--vector_size", type=int, default=100, help="Word embedding size.")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec context window size.")
    parser.add_argument("--min_count", type=int, default=2, help="Minimum token frequency.")
    parser.add_argument("--epochs", type=int, default=5, help="Word2Vec training epochs.")
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


def tokenize_texts(texts: pd.Series) -> list[list[str]]:
    """Tokenize a series of texts using the project lightweight tokenizer."""
    return [simple_tokenize(text) for text in texts.fillna("").astype(str)]


def train_word2vec_model(
    tokenized_train_texts: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
) -> Word2Vec:
    """Train a reproducible Word2Vec model on training tokens only."""
    return Word2Vec(
        sentences=tokenized_train_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        seed=seed,
        sg=1,
        epochs=epochs,
    )


def build_document_vector(tokens: list[str], word2vec_model: Word2Vec, vector_size: int) -> np.ndarray:
    """Average in-vocabulary token vectors for one document.

    Documents with no in-vocabulary tokens receive a zero vector so the matrix
    shape remains stable without borrowing information from validation or test.
    """
    vectors = [
        word2vec_model.wv[token]
        for token in tokens
        if token in word2vec_model.wv
    ]
    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def build_document_matrix(
    tokenized_texts: list[list[str]],
    word2vec_model: Word2Vec,
    vector_size: int,
) -> np.ndarray:
    """Build a 2D document-vector matrix from tokenized documents."""
    vectors = [
        build_document_vector(tokens, word2vec_model, vector_size)
        for tokens in tokenized_texts
    ]
    return np.vstack(vectors)


def build_predictions_dataframe(test_dataframe: pd.DataFrame, predictions: list[str]) -> pd.DataFrame:
    """Build the standard prediction CSV DataFrame."""
    prediction_dataframe = test_dataframe[["id", "text", "label"]].copy()
    prediction_dataframe.rename(columns={"label": "true_label"}, inplace=True)
    prediction_dataframe["predicted_label"] = predictions
    return prediction_dataframe[PREDICTION_COLUMNS]


def main() -> None:
    """Train Word2Vec and Logistic Regression, then evaluate on the test split."""
    args = parse_args()
    validate_input_paths(args)
    set_random_seed(args.seed)

    train_dataframe = load_model_dataset(args.train_path)
    load_model_dataset(args.val_path)
    test_dataframe = load_model_dataset(args.test_path)

    tokenized_train_texts = tokenize_texts(train_dataframe["text"])
    tokenized_test_texts = tokenize_texts(test_dataframe["text"])
    y_train = train_dataframe["label"].astype(str)
    y_true = test_dataframe["label"].astype(str).tolist()

    start_train_time = time.time()
    word2vec_model = train_word2vec_model(
        tokenized_train_texts=tokenized_train_texts,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        seed=args.seed,
    )
    x_train = build_document_matrix(tokenized_train_texts, word2vec_model, args.vector_size)
    classifier = LogisticRegression(max_iter=1000, random_state=args.seed)
    classifier.fit(x_train, y_train)
    train_time_sec = time.time() - start_train_time

    start_inference_time = time.time()
    x_test = build_document_matrix(tokenized_test_texts, word2vec_model, args.vector_size)
    predictions = classifier.predict(x_test).astype(str).tolist()
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
