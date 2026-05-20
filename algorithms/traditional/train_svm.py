"""SVM training and prediction pipeline.

Implementation of TF-IDF + LinearSVC using the unified project structure.
"""
import argparse
import time
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from algorithms.utils.data_io import load_csv_dataset, save_dataframe
from algorithms.utils.seed import set_random_seed
from algorithms.utils.metrics import compute_classification_metrics, build_metrics_row, save_metrics_row

MODEL_NAME = "svm"
PREDICTION_COLUMNS = ["id", "text", "true_label", "predicted_label"]

def parse_args() -> argparse.Namespace:
    """Parse standard model training arguments."""
    parser = argparse.ArgumentParser(description="Train LinearSVC model.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--train_path", required=True, help="Path to train.csv.")
    parser.add_argument("--val_path", required=True, help="Path to val.csv.")
    parser.add_argument("--test_path", required=True, help="Path to test.csv.")
    parser.add_argument("--output_dir", required=True, help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def validate_input_paths(args: argparse.Namespace) -> None:
    """Validate input CSV paths."""
    for attr in ["train_path", "val_path", "test_path"]:
        path = Path(getattr(args, attr))
        if not path.exists():
            raise FileNotFoundError(f"Required input path does not exist: {path}")

def main() -> None:
    args = parse_args()
    validate_input_paths(args)
    set_random_seed(args.seed)

    print(f"Loading data from {args.train_path} and {args.test_path}...")
    train_df = load_csv_dataset(args.train_path)
    test_df = load_csv_dataset(args.test_path)

    # 1. Vectorizing Text using TF-IDF (fitting vectorizer only on train data)
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2))
    
    start_train_time = time.time()
    
    # Clean text or just handle NaNs
    train_texts = train_df["text"].fillna("").astype(str)
    test_texts = test_df["text"].fillna("").astype(str)
    
    X_train = vectorizer.fit_transform(train_texts)
    y_train = train_df["label"].astype(str)

    # 2. Training LinearSVC
    print("Training LinearSVC...")
    model = LinearSVC(random_state=args.seed, max_iter=2000)
    model.fit(X_train, y_train)
    
    train_time_sec = time.time() - start_train_time
    print(f"Training completed in {train_time_sec:.4f} seconds.")

    # 3. Evaluating on Test Set
    print("Evaluating on test set...")
    start_inference_time = time.time()
    X_test = vectorizer.transform(test_texts)
    y_pred = model.predict(X_test)
    inference_time_sec = time.time() - start_inference_time
    print(f"Inference completed in {inference_time_sec:.4f} seconds.")

    # 4. Save predictions
    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions" / f"{args.dataset_name}_{MODEL_NAME}_predictions.csv"
    
    pred_df = test_df[["id", "text", "label"]].copy()
    pred_df.rename(columns={"label": "true_label"}, inplace=True)
    pred_df["predicted_label"] = y_pred
    
    save_dataframe(pred_df[PREDICTION_COLUMNS], prediction_path)
    print(f"Saved predictions to {prediction_path}")

    # 5. Compute metrics & Save
    metrics_path = output_dir / "metrics" / f"{args.dataset_name}_{MODEL_NAME}_metrics.csv"
    y_true = test_df["label"].astype(str).tolist()
    y_pred_list = list(y_pred)
    
    metrics = compute_classification_metrics(y_true, y_pred_list)
    metrics_row = build_metrics_row(
        dataset=args.dataset_name,
        model=MODEL_NAME,
        feature_type="tfidf",
        metrics=metrics,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
        random_seed=args.seed
    )
    save_metrics_row(metrics_row, metrics_path)
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()
