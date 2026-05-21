# DTS406TC Coursework 1: Document Topic Classification

## 1. Project Overview

This repository is for DTS406TC Natural Language Processing Coursework 1. The topic is document topic classification. The project will build a reproducible Python pipeline for collecting datasets, preprocessing text, training topic classifiers, evaluating predictions, and preparing report-ready results.

Current stage: dataset preparation and algorithm integration. `dataset_1` has been prepared as a balanced Yahoo Answers Topics subset, and `dataset_2` has been prepared as an AG News Sports vs Business binary news topic classification dataset. Model predictions and metrics are generated only when real model scripts are run.

## 2. Coursework Requirements Summary

The final coursework project must support:

- two document topic classification datasets;
- at least 3000 instances in each dataset;
- two different classification scenarios;
- four algorithms applied to both datasets:
  - Naive Bayes;
  - SVM;
  - Word2Vec-based classifier;
  - BERT-based classifier;
- evaluation with at least precision, recall, and F1-score;
- CSV outputs for predictions, metrics, summaries, and figures;
- a final group report and individual literature reviews.

All implementation code must be written in Python.

## 3. Current Project Status

Implemented:

- standard project directory structure;
- English README and Chinese README;
- raw dataset delivery validation script;
- shared data I/O, seed, text processing, and metrics utilities;
- basic preprocessing, split, and statistics scripts;
- `dataset_1` and `dataset_2` raw and processed CSV files;
- Naive Bayes, SVM, Word2Vec-based classifier, and BERT script interfaces with common command-line arguments;
- Naive Bayes implementation using TF-IDF + MultinomialNB;
- Word2Vec implementation using self-trained Word2Vec average vectors + Logistic Regression;
- prediction evaluation, metrics aggregation, and result plotting scripts;
- lightweight requirements file.

Current group ownership:

- Junhao Feng: `dataset_1`, SVM, BERT-based classifier;
- Xinyu Ren: `dataset_2`;
- Jiacheng Gui: overall framework, Naive Bayes, Word2Vec-based classifier, integration and final review.

Current datasets:

| Dataset | Source / Name | Classification scenario | Raw size | Labels | Split |
| --- | --- | --- | ---: | ---: | --- |
| `dataset_1` | Yahoo Answers Topics | Community Q&A topic classification | 6000 | 10 | 4200 / 900 / 900 |
| `dataset_2` | AG News - Sports vs Business Classification | News topic classification | 60000 | 2 | 41966 / 8993 / 8993 |

`dataset_1` and `dataset_2` represent different classification scenarios. No fake datasets, predictions, or metrics are generated. Current real metrics exist for all four required models on both datasets.

## 4. Expected Project Structure

```text
.
├── AGENTS.md
├── README.md
├── README_cn.md
├── requirements.txt
├── docs/
├── algorithms/
│   ├── preprocessing/
│   │   ├── validate_raw_dataset.py
│   │   ├── preprocess_dataset.py
│   │   ├── split_dataset.py
│   │   └── dataset_statistics.py
│   ├── traditional/
│   │   ├── train_naive_bayes.py
│   │   └── train_svm.py
│   ├── deep_learning/
│   │   ├── train_word2vec_classifier.py
│   │   └── train_bert_classifier.py
│   ├── evaluation/
│   │   ├── evaluate_predictions.py
│   │   ├── aggregate_results.py
│   │   └── plot_results.py
│   └── utils/
│       ├── data_io.py
│       ├── metrics.py
│       ├── text_processing.py
│       └── seed.py
├── data/
│   ├── raw/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   ├── processed/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   └── results/
│       ├── predictions/
│       ├── metrics/
│       ├── tables/
│       └── figures/
├── report/
│   ├── main.tex
│   ├── references.bib
│   ├── sections/
│   ├── tables/
│   ├── figures/
│   ├── group_report/
│   └── individual_literature_reviews/
└── notes/
```

## 5. Dataset Collection Format

Each raw dataset should be submitted as one folder under `data/raw/`, for example `data/raw/dataset_1/`.

Required files:

```text
raw_data.csv
sample_100.csv
dataset_info.md
label_mapping.csv
initial_label_distribution.csv
```

`raw_data.csv` must contain at least:

```csv
id,text,label
```

Rules:

- `id` must be non-empty and unique;
- `text` must be non-empty;
- `label` must be non-empty;
- each dataset must contain at least 3000 rows;
- `sample_100.csv` must contain exactly 100 rows and keep the same required columns;
- `label_mapping.csv` must contain `original_label,unified_label,description`;
- `initial_label_distribution.csv` must contain `label,count,percentage`.

## 6. Raw Dataset Validation Command

Validate a submitted raw dataset folder:

```bash
python algorithms/preprocessing/validate_raw_dataset.py \
  --dataset_dir data/raw/dataset_1
```

The script prints clear validation results and writes a validation report to `notes/`. If serious errors are found, it returns a non-zero exit code.

## 7. Processed Data Format

After preprocessing and splitting, each processed dataset should contain:

```text
data/processed/dataset_1/
├── cleaned.csv
├── train.csv
├── val.csv
├── test.csv
├── dataset_card.md
├── preprocessing_log.md
├── statistics.csv
├── label_distribution.csv
└── word_frequency.csv
```

The minimum required columns for split files are:

```csv
id,text,label
```

The recommended split ratio is 70% train, 15% validation, and 15% test, using a fixed random seed such as 42.

## 8. Preprocessing, Split, and Statistics Workflow

`dataset_1` was prepared from Yahoo Answers Topics. The final standardized
files are stored under `data/raw/dataset_1/` and `data/processed/dataset_1/`.
The original downloaded source files are not kept in the repository. Kaggle API
access is not required, and `kaggle` is not a project dependency.

For generic datasets, clean a raw dataset:

```bash
python algorithms/preprocessing/preprocess_dataset.py \
  --input_path data/raw/dataset_2/raw_data.csv \
  --output_path data/processed/dataset_2/cleaned.csv
```

Create train/validation/test splits:

```bash
python algorithms/preprocessing/split_dataset.py \
  --input_path data/processed/dataset_2/cleaned.csv \
  --dataset_name dataset_2 \
  --output_dir data/processed/dataset_2 \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

Generate statistics for the full processed dataset. This is the recommended mode for formal report statistics because it combines `train.csv`, `val.csv`, and `test.csv` before writing `statistics.csv`, `label_distribution.csv`, and `word_frequency.csv`:

```bash
python algorithms/preprocessing/dataset_statistics.py \
  --input_dir data/processed/dataset_2 \
  --dataset_name dataset_2 \
  --output_dir data/processed/dataset_2
```

If train-only or another split-specific statistic is needed, use the single-file `--input_path` mode and save it to a clearly named output directory:

```bash
python algorithms/preprocessing/dataset_statistics.py \
  --input_path data/processed/dataset_1/train.csv \
  --dataset_name dataset_1 \
  --output_dir data/processed/dataset_1/train_statistics \
  --split_name train
```

The statistics script outputs:

- `statistics.csv`;
- `label_distribution.csv`;
- `word_frequency.csv`.

For the prepared datasets, the standard statistics files under `data/processed/dataset_x/` are generated from the full processed dataset (`train.csv + val.csv + test.csv`). Formal report tables should use these full processed dataset statistics, not train-only statistics.

## 9. Algorithm Script Interface

All model scripts share the same command-line interface:

```bash
python algorithms/traditional/train_naive_bayes.py \
  --dataset_name dataset_1 \
  --train_path data/processed/dataset_1/train.csv \
  --val_path data/processed/dataset_1/val.csv \
  --test_path data/processed/dataset_1/test.csv \
  --output_dir data/results \
  --seed 42
```

Available model scripts:

- `algorithms/traditional/train_naive_bayes.py`
- `algorithms/traditional/train_svm.py`
- `algorithms/deep_learning/train_word2vec_classifier.py`
- `algorithms/deep_learning/train_bert_classifier.py`

All model scripts must keep this interface. Scripts that are still templates must raise clearly instead of saving fake outputs; implemented scripts must save real prediction and metrics CSV files in the standard result paths.

## 10. Result CSV Format

Future model implementations must save prediction files to:

```text
data/results/predictions/{dataset_name}_{model_name}_predictions.csv
```

Required prediction columns:

```csv
id,text,true_label,predicted_label
```

Future model implementations must save metrics files to:

```text
data/results/metrics/{dataset_name}_{model_name}_metrics.csv
```

Required metrics columns:

```csv
dataset,model,feature_type,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted,accuracy,train_time_sec,inference_time_sec,random_seed
```

Current result status: real prediction and metrics CSV files have been generated for:

- `dataset_1_naive_bayes`;
- `dataset_1_svm`;
- `dataset_1_word2vec`;
- `dataset_1_bert`;
- `dataset_2_naive_bayes`;
- `dataset_2_svm`;
- `dataset_2_word2vec`;
- `dataset_2_bert`.

The aggregate summary is saved at `data/results/tables/all_metrics_summary.csv`, and the macro-F1 comparison figure is saved at `data/results/figures/f1_macro_comparison.png`.

Current real test-set metric summary:

| Dataset | Model | precision_macro | recall_macro | f1_macro | accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `dataset_1` | BERT | 0.6764 | 0.6822 | 0.6750 | 0.6822 |
| `dataset_1` | Naive Bayes | 0.5936 | 0.5500 | 0.5383 | 0.5500 |
| `dataset_1` | SVM | 0.5740 | 0.5778 | 0.5736 | 0.5778 |
| `dataset_1` | Word2Vec | 0.4336 | 0.4433 | 0.4291 | 0.4433 |
| `dataset_2` | BERT | 0.9945 | 0.9944 | 0.9944 | 0.9944 |
| `dataset_2` | Naive Bayes | 0.9845 | 0.9843 | 0.9843 | 0.9843 |
| `dataset_2` | SVM | 0.9896 | 0.9895 | 0.9895 | 0.9895 |
| `dataset_2` | Word2Vec | 0.9859 | 0.9859 | 0.9859 | 0.9859 |

## 11. Evaluation and Aggregation Workflow

Evaluate an existing prediction CSV:

```bash
python algorithms/evaluation/evaluate_predictions.py \
  --prediction_path data/results/predictions/dataset_1_naive_bayes_predictions.csv \
  --dataset_name dataset_1 \
  --model_name naive_bayes \
  --feature_type tfidf_unigram_bigram \
  --output_path data/results/metrics/dataset_1_naive_bayes_metrics.csv \
  --seed 42
```

Aggregate all metrics CSV files:

```bash
python algorithms/evaluation/aggregate_results.py \
  --metrics_dir data/results/metrics \
  --output_dir data/results/tables
```

Plot macro F1 comparison from the aggregated summary:

```bash
python algorithms/evaluation/plot_results.py \
  --summary_path data/results/tables/all_metrics_summary.csv \
  --output_dir data/results/figures
```

The plotting script only generates figures from an existing real summary CSV.

## 12. What Has Been Implemented

The following model scripts are fully implemented:

- Naive Bayes: TF-IDF unigram/bigram features + MultinomialNB implemented by Jiacheng Gui;
- SVM: TF-IDF + LinearSVC classification implemented by Junhao Feng;
- Word2Vec-based classifier: self-trained Word2Vec + average document vectors + Logistic Regression implemented by Jiacheng Gui;
- BERT-based classifier: DistilBERT sequence classification fine-tuning implemented by Junhao Feng.

Naive Bayes uses `feature_type=tfidf_unigram_bigram`. Word2Vec uses `feature_type=word2vec_avg_logreg`, `vector_size=100`, `window=5`, `min_count=2`, `workers=1`, `sg=1`, and Logistic Regression with `max_iter=1000`.

## 13. How Group Members Should Add Algorithm Implementations Later

When implementing a model, group members should:

1. keep the existing command-line arguments unchanged;
2. load `train.csv`, `val.csv`, and `test.csv` from the given paths;
3. fit vectorizers, embeddings, and models only on training data;
4. use validation data only for tuning or early stopping;
5. evaluate final performance on the test set;
6. save prediction and metrics CSV files using the standard paths and columns;
7. update both `README.md` and `README_cn.md` if dependencies, commands, or outputs change;
8. avoid hard-coded absolute paths and avoid generating unsupported report claims.

## 14. Final Report Structure

The integrated final report is built from `report/main.tex`. Section files are stored under `report/sections/`, generated LaTeX tables under `report/tables/`, and report-local figure copies under `report/figures/`. Existing individual literature review source folders remain under `report/individual_literature_reviews/`; Junhao Feng's literature review is currently represented by a placeholder in the integrated report until his source text is provided.

The current final-report table and figure inputs are derived from real project outputs:

- metrics table: `data/results/tables/all_metrics_summary.csv`;
- F1 figure source: `data/results/figures/f1_macro_comparison.png`;
- dataset statistics: `data/processed/dataset_1/statistics.csv` and `data/processed/dataset_2/statistics.csv`.

## 15. Dependencies

Dependencies are listed in `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
torch
transformers
gensim
```

Install them with:

```bash
pip install -r requirements.txt
```
