# Document Topic Classification - DTS406TC Coursework 1

This repository contains the implementation for the DTS406TC Natural Language Processing Coursework 1: Document Topic Classification.

## Overview

This project implements a complete document topic classification pipeline with:

- **Dataset Collection and Preprocessing**: Two datasets (Sports vs Business news articles)
- **Four Classification Methods**: Naïve Bayes, SVM, Word2Vec, and BERT
- **Evaluation**: Consistent metrics across all models
- **Documentation**: Complete README in English and Chinese

## Project Structure

```
.
├── algorithms/
│   ├── preprocessing/          # Dataset preprocessing
│   │   ├── preprocess_dataset.py
│   │   ├── split_dataset.py
│   │   ├── dataset_statistics.py
│   │   └── stats_analysis_dataset1.py
│   │
│   ├── traditional/            # Traditional ML algorithms
│   │   ├── train_naive_bayes.py
│   │   └── train_svm.py
│   │
│   ├── deep_learning/          # Deep learning algorithms
│   │   ├── train_word2vec_classifier.py
│   │   └── train_bert_classifier.py
│   │
│   ├── evaluation/             # Evaluation utilities
│   │   ├── evaluate_predictions.py
│   │   ├── aggregate_results.py
│   │   └── plot_results.py
│   │
│   └── utils/                  # Utility functions
│       ├── data_io.py
│       ├── metrics.py
│       ├── text_processing.py
│       └── seed.py
│
├── data/
│   ├── raw/                    # Raw datasets
│   │   ├── dataset_1/          # Dataset 1 (if applicable)
│   │   └── dataset_2/          # Dataset 2: AG News
│   ├── processed/              # Preprocessed data
│   │   ├── dataset_1/          # Dataset 1 processed data
│   │   └── dataset_2/          # Dataset 2 processed data
│   └── results/                # Model results
│       ├── predictions/        # Prediction files
│       ├── metrics/            # Metrics files
│       └── figures/            # Generated plots
│           └── dataset_2/      # Dataset 2 figures
│
├── dataset_2/                  # Dataset 2 files
│   ├── raw_data.csv
│   ├── sample_100.csv
│   ├── dataset_info.md
│   ├── label_mapping.csv
│   └── initial_label_distribution.csv
│
├── requirements.txt
├── README.md                   # English documentation
└── README_cn.md                # Chinese documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Dataset

```bash
python algorithms/preprocessing/preprocess_dataset.py \
    --dataset_name dataset_2 \
    --input_path dataset_2/raw_data.csv \
    --output_dir data/processed \
    --seed 42
```

### 2. Train Naïve Bayes

```bash
python algorithms/traditional/train_naive_bayes.py \
    --dataset_name dataset_2 \
    --train_path data/processed/dataset_2/train.csv \
    --val_path data/processed/dataset_2/val.csv \
    --test_path data/processed/dataset_2/test.csv \
    --output_dir data/results \
    --seed 42
```

### 3. Train Word2Vec Classifier

```bash
python algorithms/deep_learning/train_word2vec_classifier.py \
    --dataset_name dataset_2 \
    --train_path data/processed/dataset_2/train.csv \
    --val_path data/processed/dataset_2/val.csv \
    --test_path data/processed/dataset_2/test.csv \
    --output_dir data/results \
    --seed 42 \
    --embedding_dim 100 \
    --window_size 5 \
    --min_count 2 \
    --epochs 10
```

### 4. Evaluate Results

```bash
python algorithms/evaluation/evaluate_predictions.py \
    --dataset_name dataset_2 \
    --predictions_dir data/results/predictions \
    --metrics_dir data/results/metrics \
    --output_dir data/results/evaluation
```

### 5. Aggregate Results

```bash
python algorithms/evaluation/aggregate_results.py \
    --metrics_dir data/results/metrics \
    --output_dir data/results
```

### 6. Generate Plots

```bash
python algorithms/evaluation/plot_results.py \
    --dataset_name dataset_2 \
    --metrics_dir data/results/metrics \
    --output_dir data/results/figures
```

## Datasets

### Dataset 2: AG News (Sports vs Business)

- **Path**: `data/raw/dataset_2/`
- **Scenario**: News topic classification
- **Labels**: sports, business
- **Number of Raw Instances**: 60,000
- **Split**:
  - train: 42,000
  - validation: 9,000
  - test: 9,000

## Implemented Algorithms

### Dataset 2

- **Naïve Bayes**: Implemented
- **Word2Vec-based classifier**: Implemented (simplified skip-gram-style embedding baseline)

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn

## Authors

- Group Member 1: [Your Name]
- Group Member 2: [Your Name]

## License

MIT License
