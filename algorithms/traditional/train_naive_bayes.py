"""
Train Naïve Bayes Classifier for Document Topic Classification

This script trains a Naïve Bayes classifier using TF-IDF features.
It supports command-line arguments for dataset paths and output configuration.

Note: Validation set is loaded for interface consistency but not used for tuning
in this baseline implementation.
"""

import os
import argparse
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from algorithms.utils.data_io import load_dataset, ensure_output_dir
from algorithms.utils.metrics import compute_all_metrics, build_metrics_row, save_metrics_row
from algorithms.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Naïve Bayes classifier for document topic classification')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset identifier')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def load_data(train_path, val_path, test_path):
    """Load train, validation, and test datasets.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)
    test_df = load_dataset(test_path)
    
    return train_df, val_df, test_df


def train_model(X_train, y_train, seed):
    """Train Naïve Bayes classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        seed: Random seed (currently not used, kept for interface consistency)
        
    Returns:
        Trained MultinomialNB model
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics.
    
    Args:
        model: Trained classifier
        X: Features to evaluate
        y: Ground truth labels
        dataset_name: Dataset identifier (for logging)
        
    Returns:
        Tuple of (metrics dict, predictions array)
    """
    y_pred = model.predict(X)
    
    metrics = compute_all_metrics(y, y_pred)
    
    return metrics, y_pred


def save_results(dataset_name, model_name, metrics, predictions, output_dir, feature_type, train_time, inference_time, seed):
    """Save predictions and metrics to CSV files.
    
    Args:
        dataset_name: Dataset identifier
        model_name: Model name
        metrics: Dictionary with computed metrics
        predictions: Dictionary with predictions
        output_dir: Output directory
        feature_type: Feature type (e.g., 'TF-IDF')
        train_time: Training time in seconds
        inference_time: Inference time in seconds
        seed: Random seed
        
    Returns:
        Metrics dictionary
    """
    # Create output directories
    predictions_dir = ensure_output_dir(os.path.join(output_dir, 'predictions'))
    metrics_dir = ensure_output_dir(os.path.join(output_dir, 'metrics'))
    
    # Save predictions
    pred_df = pd.DataFrame({
        'id': predictions['id'],
        'text': predictions['text'],
        'true_label': predictions['true_label'],
        'predicted_label': predictions['predicted_label']
    })
    pred_df.to_csv(os.path.join(predictions_dir, f'{dataset_name}_{model_name}_predictions.csv'), index=False)
    
    # Build and save metrics row
    metrics_row = build_metrics_row(
        dataset=dataset_name,
        model=model_name,
        feature_type=feature_type,
        metrics=metrics,
        train_time=train_time,
        inference_time=inference_time,
        seed=seed
    )
    save_metrics_row(metrics_row, os.path.join(metrics_dir, f'{dataset_name}_{model_name}_metrics.csv'))
    
    return metrics


def main():
    args = parse_args()
    
    print("="*60)
    print("TRAINING NAÏVE BAYES CLASSIFIER")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_df, val_df, test_df = load_data(args.train_path, args.val_path, args.test_path)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val: {len(val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    
    # Encode labels
    print("\n[2/5] Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['label'])
    
    y_train = label_encoder.transform(train_df['label'])
    y_val = label_encoder.transform(val_df['label'])
    y_test = label_encoder.transform(test_df['label'])
    
    print(f"  Labels: {list(label_encoder.classes_)}")
    
    # Create TF-IDF vectorizer
    print("\n[3/5] Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Fit on training data only
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature shape: {X_train.shape}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Train model
    print("\n[4/5] Training model...")
    start_time = time.time()
    model = train_model(X_train, y_train, args.seed)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    start_time = time.time()
    metrics, y_pred_test = evaluate_model(model, X_test, y_test, 'test')
    inference_time = time.time() - start_time
    print(f"  Inference time: {inference_time:.2f} seconds")
    
    # Print metrics
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"  F1-score (weighted): {metrics['f1_weighted']:.4f}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    predictions = {
        'id': test_df['id'].values,
        'text': test_df['text'].values,
        'true_label': label_encoder.inverse_transform(y_test),
        'predicted_label': label_encoder.inverse_transform(y_pred_test)
    }
    
    metrics = save_results(
        dataset_name=args.dataset_name,
        model_name='naive_bayes',
        metrics=metrics,
        predictions=predictions,
        output_dir=args.output_dir,
        feature_type='TF-IDF',
        train_time=train_time,
        inference_time=inference_time,
        seed=args.seed
    )
    
    print(f"  Predictions saved to: {args.output_dir}/predictions/{args.dataset_name}_naive_bayes_predictions.csv")
    print(f"  Metrics saved to: {args.output_dir}/metrics/{args.dataset_name}_naive_bayes_metrics.csv")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
