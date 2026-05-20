"""
Train Word2Vec-based Classifier for Document Topic Classification

This script trains a classifier using a simplified skip-gram-style embedding baseline with average pooling.
Note: This is a simplified implementation for coursework purposes. It does not implement full negative sampling
or full softmax Word2Vec training. For a standard Word2Vec implementation, consider using gensim.models.Word2Vec.

It supports command-line arguments for dataset paths and output configuration.
"""

import os
import argparse
import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from algorithms.utils.data_io import load_dataset, ensure_output_dir
from algorithms.utils.metrics import compute_all_metrics, build_metrics_row, save_metrics_row
from algorithms.utils.seed import set_seed
from algorithms.utils.text_processing import simple_tokenize, remove_stopwords


def parse_args():
    parser = argparse.ArgumentParser(description='Train Word2Vec-based classifier for document topic classification')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset identifier')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Word2Vec embedding dimension')
    parser.add_argument('--window_size', type=int, default=5, help='Word2Vec window size')
    parser.add_argument('--min_count', type=int, default=2, help='Word2Vec minimum word count')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')
    return parser.parse_args()


def preprocess_text(text):
    """Clean and tokenize text using the text_processing module.
    
    Args:
        text: Input text string
        
    Returns:
        List of preprocessed tokens
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Use simple_tokenize from text_processing module
    tokens = simple_tokenize(text)
    
    # Remove stopwords using the utility function
    tokens = remove_stopwords(tokens)
    
    return tokens


def softmax(x):
    """Compute softmax values for each set of scores."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def build_word2vec_model(texts, embedding_dim, window_size, min_count, epochs, learning_rate, seed):
    """
    Build simplified Word2Vec model using skip-gram-style training.
    
    This implements a simplified Word2Vec algorithm:
    - Skip-gram: predict context words from target word
    - Simplified training: without full negative sampling or full softmax
    
    Note: This is a simplified implementation for coursework purposes. It does not implement
    full negative sampling or full softmax Word2Vec training.
    """
    # Build vocabulary using Counter
    vocab = Counter()
    for text in texts:
        tokens = preprocess_text(text)
        vocab.update(tokens)
    
    # Filter by min_count
    vocab = {word: count for word, count in vocab.items() if count >= min_count}
    vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
    
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab.keys())}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    vocab_size = len(vocab)
    
    print(f"  Vocabulary size: {vocab_size}")
    
    # Initialize embeddings (simplified Word2Vec initialization)
    set_seed(seed)
    # Input embeddings (for target words)
    input_embeddings = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
    # Output embeddings (for context words)
    output_embeddings = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
    
    # Training loop
    print(f"  Training for {epochs} epochs...")
    print("  Note: This is a simplified skip-gram-style embedding baseline.")
    print("        It does not implement full negative sampling or full softmax Word2Vec training.")
    
    for epoch in range(epochs):
        total_loss = 0
        num_samples = 0
        
        for text in texts:
            tokens = preprocess_text(text)
            if len(tokens) < 3:
                continue
            
            for i, target_word in enumerate(tokens):
                if target_word not in word_to_idx:
                    continue
                
                target_idx = word_to_idx[target_word]
                
                # Get context words
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i == j:
                        continue
                    
                    context_word = tokens[j]
                    if context_word not in word_to_idx:
                        continue
                    
                    context_idx = word_to_idx[context_word]
                    
                    # Simplified skip-gram training
                    # Positive sample: (target, context)
                    # Note: This is a simplified implementation without full negative sampling
                    
                    # Forward pass for positive sample
                    input_vec = input_embeddings[target_idx]
                    output_vec = output_embeddings[context_idx]
                    
                    # Compute score
                    score = np.dot(input_vec, output_vec)
                    
                    # Compute loss (binary cross-entropy)
                    # Positive sample should have high score (close to 1)
                    # We use sigmoid to convert score to probability
                    prob = 1.0 / (1.0 + np.exp(-score))
                    
                    # Binary cross-entropy loss
                    loss = -np.log(prob + 1e-10)
                    total_loss += loss
                    num_samples += 1
                    
                    # Backward pass
                    # Gradient for positive sample
                    error = prob - 1  # -1 because we want prob to be close to 1
                    
                    # Update embeddings
                    input_grad = error * output_vec
                    output_grad = error * input_vec
                    
                    input_embeddings[target_idx] -= learning_rate * input_grad
                    output_embeddings[context_idx] -= learning_rate * output_grad
        
        avg_loss = total_loss / max(num_samples, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Combine input and output embeddings (simplified Word2Vec practice)
    final_embeddings = input_embeddings + output_embeddings
    
    return final_embeddings, word_to_idx, idx_to_word, vocab_size


def get_document_vector(tokens, embeddings, word_to_idx, embedding_dim):
    """Get document vector by averaging word embeddings."""
    valid_tokens = [token for token in tokens if token in word_to_idx]
    
    if len(valid_tokens) == 0:
        return np.zeros(embedding_dim)
    
    doc_vector = np.mean([embeddings[word_to_idx[token]] for token in valid_tokens], axis=0)
    return doc_vector


def main():
    args = parse_args()
    
    print("="*60)
    print("TRAINING WORD2VEC-BASED CLASSIFIER (Simplified Skip-gram-style Embedding Baseline)")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = load_dataset(args.train_path)
    val_df = load_dataset(args.val_path)
    test_df = load_dataset(args.test_path)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val: {len(val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    
    # Encode labels
    print("\n[2/6] Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['label'])
    
    y_train = label_encoder.transform(train_df['label'])
    y_val = label_encoder.transform(val_df['label'])
    y_test = label_encoder.transform(test_df['label'])
    
    print(f"  Labels: {list(label_encoder.classes_)}")
    
    # Build Word2Vec model on training data only
    print("\n[3/6] Building Word2Vec model (Simplified Skip-gram-style Embedding Baseline)...")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Window size: {args.window_size}")
    print(f"  Min count: {args.min_count}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print("  Note: This is a simplified skip-gram-style embedding baseline. It does not implement")
    print("        full negative sampling or full softmax Word2Vec training.")
    
    start_time = time.time()
    embeddings, word_to_idx, idx_to_word, vocab_size = build_word2vec_model(
        train_df['text'].tolist(),
        args.embedding_dim,
        args.window_size,
        args.min_count,
        args.epochs,
        args.learning_rate,
        args.seed
    )
    build_time = time.time() - start_time
    print(f"  Build time: {build_time:.2f} seconds")
    
    # Convert documents to vectors
    print("\n[4/6] Converting documents to vectors...")
    
    def get_all_vectors(df):
        vectors = []
        for text in df['text']:
            tokens = preprocess_text(text)
            doc_vector = get_document_vector(tokens, embeddings, word_to_idx, args.embedding_dim)
            vectors.append(doc_vector)
        return np.array(vectors)
    
    X_train = get_all_vectors(train_df)
    X_val = get_all_vectors(val_df)
    X_test = get_all_vectors(test_df)
    
    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape: {X_val.shape}")
    print(f"  Test shape: {X_test.shape}")
    
    # Train classifier
    print("\n[5/6] Training classifier (Logistic Regression)...")
    start_time = time.time()
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=args.seed,
        solver='lbfgs',
        multi_class='multinomial'
    )
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    start_time = time.time()
    y_pred_test = classifier.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate metrics using the utility function
    metrics = compute_all_metrics(y_test, y_pred_test)
    
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
    
    # Create output directories
    predictions_dir = ensure_output_dir(os.path.join(args.output_dir, 'predictions'))
    metrics_dir = ensure_output_dir(os.path.join(args.output_dir, 'metrics'))
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(os.path.join(predictions_dir, f'{args.dataset_name}_word2vec_predictions.csv'), index=False)
    print(f"  Predictions saved to: {predictions_dir}/{args.dataset_name}_word2vec_predictions.csv")
    
    # Build and save metrics row
    metrics_row = build_metrics_row(
        dataset=args.dataset_name,
        model='word2vec',
        feature_type=f'Simplified-Word2Vec-{args.embedding_dim}d',
        metrics=metrics,
        train_time=train_time,
        inference_time=inference_time,
        seed=args.seed
    )
    save_metrics_row(metrics_row, os.path.join(metrics_dir, f'{args.dataset_name}_word2vec_metrics.csv'))
    print(f"  Metrics saved to: {metrics_dir}/{args.dataset_name}_word2vec_metrics.csv")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNote: This Word2Vec implementation is a simplified skip-gram-style embedding baseline.")
    print("      It does not implement full negative sampling or full softmax Word2Vec training.")


if __name__ == '__main__':
    main()
