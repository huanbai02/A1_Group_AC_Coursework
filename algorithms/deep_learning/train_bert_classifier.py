"""BERT-based classifier implementation.

Implementation: DistilBERT fine-tuning using Hugging Face Transformers.
"""
import argparse
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from algorithms.utils.data_io import load_csv_dataset, save_dataframe
from algorithms.utils.seed import set_random_seed
from algorithms.utils.metrics import compute_classification_metrics, build_metrics_row, save_metrics_row

MODEL_NAME = "bert"
PREDICTION_COLUMNS = ["id", "text", "true_label", "predicted_label"]

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT classifier.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--train_path", required=True, help="Path to train.csv.")
    parser.add_argument("--val_path", required=True, help="Path to val.csv.")
    parser.add_argument("--test_path", required=True, help="Path to test.csv.")
    parser.add_argument("--output_dir", required=True, help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading data...")
    train_df = load_csv_dataset(args.train_path)
    val_df = load_csv_dataset(args.val_path)
    test_df = load_csv_dataset(args.test_path)
    
    # Encode labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df["label"].astype(str))
    val_labels = le.transform(val_df["label"].astype(str))
    test_labels = le.transform(test_df["label"].astype(str))
    num_labels = len(le.classes_)
    
    pretrained_model = "distilbert-base-uncased"
    print(f"Loading tokenizer {pretrained_model}...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    train_dataset = TextDataset(train_df['text'].tolist(), train_labels, tokenizer)
    val_dataset = TextDataset(val_df['text'].tolist(), val_labels, tokenizer)
    test_dataset = TextDataset(test_df['text'].tolist(), test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Loading model {pretrained_model}...")
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    print("Training started...")
    start_train = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
                loss = outputs.loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
        print(f"Epoch {epoch+1}/{args.epochs} completed.")
        
    train_time = time.time() - start_train
    
    print("Evaluating on test set...")
    start_inference = time.time()
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            
    inference_time = time.time() - start_inference
    
    # Decode predictions back to original label strings
    y_pred_labels = le.inverse_transform(predictions)
    
    # Save predictions
    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions" / f"{args.dataset_name}_{MODEL_NAME}_predictions.csv"
    
    pred_df = test_df[["id", "text", "label"]].copy()
    pred_df.rename(columns={"label": "true_label"}, inplace=True)
    pred_df["predicted_label"] = y_pred_labels
    
    save_dataframe(pred_df[PREDICTION_COLUMNS], prediction_path)
    print(f"Saved predictions to {prediction_path}")
    
    # Compute metrics & Save
    metrics_path = output_dir / "metrics" / f"{args.dataset_name}_{MODEL_NAME}_metrics.csv"
    y_true = test_df["label"].astype(str).tolist()
    y_pred_list = list(y_pred_labels)
    
    metrics = compute_classification_metrics(y_true, y_pred_list)
    metrics_row = build_metrics_row(
        dataset=args.dataset_name,
        model=MODEL_NAME,
        feature_type="distilbert",
        metrics=metrics,
        train_time_sec=train_time,
        inference_time_sec=inference_time,
        random_seed=args.seed
    )
    save_metrics_row(metrics_row, metrics_path)
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()
