# AGENTS.md

## 1. Project Role and Scope

This repository is for **DTS406TC Natural Language Processing Coursework 1: Document Topic Classification**.

The project goal is to build a complete and reproducible document topic classification pipeline, including:

- dataset collection and standardisation;
- data preprocessing and statistical analysis;
- implementation of four classification methods;
- evaluation with consistent metrics;
- result aggregation for the final report;
- English `README.md` and Chinese `README_cn.md` maintenance.

This file defines the development rules for AI Agents and human contributors.  
All Agents working on this repository must follow this document.

---

## 2. Core Coursework Requirements

The implementation must support:

1. **Two datasets**
   - Each dataset must be used for document topic classification.
   - Each dataset must contain at least 3000 instances.
   - The two datasets should represent different classification scenarios, such as:
     - news topic classification;
     - social media topic classification;
     - forum discussion classification;
     - academic paper topic classification.

2. **Four algorithms**
   - Naïve Bayes
   - SVM
   - Word2Vec-based classifier
   - BERT-based classifier

3. **All algorithms must run on both datasets**
   - Do not implement an algorithm for only one dataset.
   - Do not use different train/test splits for different models on the same dataset.

4. **Evaluation metrics**
   - At least precision, recall, and F1-score must be computed.
   - Accuracy can be reported as an additional metric.
   - Macro and weighted averages are recommended.

5. **Python only**
   - All implementation code must be written in Python.

6. **Documentation**
   - `README.md` must be maintained in English.
   - `README_cn.md` must be maintained in Chinese.
   - After every development change, both README files must be checked and updated if needed.

---

## 3. Development Priorities

Agents should follow this order:

1. Preserve and validate the data format.
2. Build reusable preprocessing and evaluation utilities.
3. Implement models with a unified input/output interface.
4. Save all experimental outputs as CSV files.
5. Generate result tables and figures only from saved CSV results.
6. Keep README documentation synchronised.
7. Avoid unnecessary complexity that makes the project hard to run or reproduce.

The project should prioritise **reproducibility, clear structure, and fair comparison** over overly complex modelling.

---

## 4. Expected Repository Structure

Use the following structure unless the user explicitly asks for changes:

```text
.
├── AGENTS.md
├── README.md
├── README_cn.md
├── requirements.txt
│
├── docs/
├── algorithms/
│   ├── preprocessing/
│   │   ├── preprocess_dataset.py
│   │   ├── split_dataset.py
│   │   └── dataset_statistics.py
│   │
│   ├── traditional/
│   │   ├── train_naive_bayes.py
│   │   └── train_svm.py
│   │
│   ├── deep_learning/
│   │   ├── train_word2vec_classifier.py
│   │   └── train_bert_classifier.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_predictions.py
│   │   ├── aggregate_results.py
│   │   └── plot_results.py
│   │
│   └── utils/
│       ├── data_io.py
│       ├── metrics.py
│       ├── text_processing.py
│       └── seed.py
│
├── data/
│   ├── raw/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   │
│   ├── processed/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   │
│   └── results/
│       ├── predictions/
│       ├── metrics/
│       ├── tables/
│       └── figures/
│
├── report/
│   ├── group_report/
│   └── individual_literature_reviews/
│
└── notes/
```

Do not create unrelated top-level folders without a clear reason.

---

## 5. Dataset Collection Format

Group members will collect datasets using the following required structure:

```text
dataset_name/
├── raw_data.csv
├── sample_100.csv
├── dataset_info.md
├── label_mapping.csv
└── initial_label_distribution.csv
```

### 5.1 `raw_data.csv`

`raw_data.csv` must contain at least these three columns:

```csv
id,text,label
```

Column rules:

| Column | Requirement |
|---|---|
| `id` | Unique sample identifier. It must not be empty. |
| `text` | Text content used for topic classification. It must not be empty. |
| `label` | Topic label. It must not be empty. |

Agents must not rename these required columns unless the user explicitly asks.

### 5.2 `sample_100.csv`

`sample_100.csv` must contain 100 randomly sampled rows from `raw_data.csv`.

Purpose:

- quick manual data quality inspection;
- label quality checking;
- text length and noise checking;
- early detection of unsuitable datasets.

The file must keep the same columns as `raw_data.csv`.

### 5.3 `dataset_info.md`

`dataset_info.md` must describe:

```markdown
# Dataset Information

## Dataset Name

## Source

## Collection Method

## Classification Scenario

## Number of Instances

## Number of Labels

## Text Field Description

## Label Field Description

## Label List

## Known Data Quality Issues

## Reason for Selection
```

The source must be clear. If a public dataset is used, include the dataset name and access link if available.

### 5.4 `label_mapping.csv`

`label_mapping.csv` must describe how original labels are mapped to unified labels.

Required columns:

```csv
original_label,unified_label,description
```

Rules:

- If the original labels are already clean, `original_label` and `unified_label` can be the same.
- Do not silently merge labels without documenting the reason.
- Do not remove labels without documenting the reason.

### 5.5 `initial_label_distribution.csv`

Required columns:

```csv
label,count,percentage
```

Rules:

- `count` must be the number of samples for each label.
- `percentage` must be calculated against the total number of samples.
- The sum of percentages should be close to 100%.

---

## 6. Processed Data Format

After preprocessing, each dataset should be saved as:

```text
data/processed/dataset_name/
├── train.csv
├── val.csv
├── test.csv
├── dataset_card.md
├── preprocessing_log.md
├── statistics.csv
├── label_distribution.csv
└── word_frequency.csv
```

### 6.1 Required Split Files

Each split file must contain:

```csv
id,text,label
```

Recommended split ratio:

```text
train: 70%
validation: 15%
test: 15%
```

or:

```text
train: 80%
validation: 10%
test: 10%
```

Rules:

- Use a fixed random seed, preferably `42`.
- Use stratified splitting when possible.
- Do not allow the same `id` or duplicate text to appear across train, validation, and test splits.
- Do not use the test set for vectorizer fitting, model training, embedding training, hyperparameter tuning, or early stopping.

---

## 7. Preprocessing Rules

The preprocessing pipeline should include:

- lowercasing;
- tokenization;
- stopword removal;
- optional stemming or lemmatization;
- empty text removal;
- duplicate removal when necessary;
- label normalisation.

Traditional models and Word2Vec can use cleaned text.

BERT-based models should use the original or lightly cleaned text because BERT has its own tokenizer and depends on contextual information.

If both cleaned text and lightly cleaned text are needed, prefer the following format:

```csv
id,text,bert_text,label
```

However, the minimum required format remains:

```csv
id,text,label
```

Any additional columns must be documented in `dataset_card.md` and `README.md` / `README_cn.md`.

---

## 8. Statistical Analysis Requirements

For each dataset, generate:

1. word frequency distribution;
2. vocabulary size;
3. average document length;
4. label distribution.

### 8.1 Main Statistics Scope

The standard statistics files under each processed dataset directory are:

```text
data/processed/dataset_x/statistics.csv
data/processed/dataset_x/label_distribution.csv
data/processed/dataset_x/word_frequency.csv
```

These files must be computed from the complete processed dataset, meaning:

```text
train.csv + val.csv + test.csv
```

The default `split` value for these report-ready statistics is `full`, which means the complete processed dataset. The default `text_version` value is `text`, which means the statistics are computed from the `text` column saved in the split CSV files.

Formal report tables should preferentially cite these full processed dataset statistics. If train-only, validation-only, or test-only statistics are needed later, they must either use different file names or explicitly mark the scope in the `split` column. Split-specific statistics must not overwrite the standard `full` statistics files.

### 8.2 `statistics.csv`

Required columns:

```csv
dataset,split,text_version,num_instances,num_labels,vocabulary_size,avg_doc_length,min_doc_length,max_doc_length
```

### 8.3 `label_distribution.csv`

Required columns:

```csv
dataset,split,label,count,percentage
```

### 8.4 `word_frequency.csv`

Required columns:

```csv
dataset,split,word,count
```

---

## 9. Model Implementation Rules

Each model script should support command-line execution.

Example:

```bash
python algorithms/traditional/train_naive_bayes.py   --dataset_name dataset_1   --train_path data/processed/dataset_1/train.csv   --val_path data/processed/dataset_1/val.csv   --test_path data/processed/dataset_1/test.csv   --output_dir data/results   --seed 42
```

All model scripts should support:

| Argument | Meaning |
|---|---|
| `--dataset_name` | Dataset identifier. |
| `--train_path` | Path to training CSV. |
| `--val_path` | Path to validation CSV. |
| `--test_path` | Path to test CSV. |
| `--output_dir` | Output directory for results. |
| `--seed` | Random seed. |

Agents must not hard-code dataset paths, local absolute paths, or personal environment paths.

---

## 10. Model-Specific Guidance

### 10.1 Naïve Bayes

Recommended implementation:

```text
TF-IDF or Bag-of-Words + MultinomialNB
```

Required notes:

- feature type;
- vectorizer settings;
- smoothing parameter;
- reason for using it as a baseline.

### 10.2 SVM

Recommended implementation:

```text
TF-IDF + LinearSVC
```

Required notes:

- feature type;
- kernel or linear classifier choice;
- regularisation parameter;
- reason why SVM works well with high-dimensional sparse text features.

### 10.3 Word2Vec-based Classifier

Recommended implementation:

```text
Word2Vec embeddings + average pooling + Logistic Regression or MLP classifier
```

Required notes:

- embedding dimension;
- window size;
- minimum word count;
- whether embeddings are self-trained or pretrained;
- how document vectors are generated;
- how out-of-vocabulary words are handled.

If self-trained Word2Vec is used, it must only be trained on the training set.

### 10.4 BERT-based Classifier

Recommended implementation options:

```text
Option 1: BERT or DistilBERT fine-tuning
Option 2: BERT feature extraction + classifier
```

Required notes:

- pretrained model name;
- tokenizer;
- maximum sequence length;
- batch size;
- learning rate;
- number of epochs;
- device used;
- whether the model is fine-tuned or used as a feature extractor.

---

## 11. Result Output Format

Each model must produce two CSV files for each dataset.

### 11.1 Prediction File

Path format:

```text
data/results/predictions/{dataset_name}_{model_name}_predictions.csv
```

Required columns:

```csv
id,text,true_label,predicted_label
```

### 11.2 Metrics File

Path format:

```text
data/results/metrics/{dataset_name}_{model_name}_metrics.csv
```

Required columns:

```csv
dataset,model,feature_type,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted,accuracy,train_time_sec,inference_time_sec,random_seed
```

Rules:

- Save one metrics row per dataset-model pair.
- Do not only print results to the terminal.
- Do not save screenshots as the only evidence of results.
- CSV files are the source of truth for report tables and figures.

---

## 12. Evaluation Rules

At minimum, compute:

- precision;
- recall;
- F1-score.

Recommended additional outputs:

- accuracy;
- macro average;
- weighted average;
- confusion matrix;
- classification report.

Evaluation must be performed on the test set.

Do not evaluate on the training set and report it as final performance.

---

## 13. Coding and Comment Rules

All Python code must follow the rules below. The goal is to keep the coursework implementation clear, reproducible, easy to test, and easy to integrate across group members.

### 13.1 Basic Coding Principles

- Follow simple, readable, and maintainable Python style.
- Prefer clear implementation over clever or overly abstract implementation.
- Keep each script, function, and class focused on one clear responsibility.
- Keep high cohesion and low coupling between preprocessing, model training, evaluation, plotting, and utility modules.
- Do not place logic in the wrong layer only for convenience.
- Do not perform unrelated refactoring, renaming, formatting, or directory changes.
- Do not introduce framework-style abstractions that make the coursework harder to run or explain.
- Make code runnable from the project root unless the user explicitly asks for another execution style.

### 13.2 File Encoding and Formatting

- All source files must use UTF-8 encoding.
- Keep one newline at the end of each file.
- Do not mix tabs and spaces.
- Follow common Python formatting conventions, including PEP 8 where practical.
- Keep formatting consistent with the existing codebase.
- Do not reformat unrelated files or make large formatting-only changes.
- Use stable relative paths from the project root. Do not hard-code local absolute paths or personal environment paths.

### 13.3 Naming Rules

- Use clear and meaningful names for files, functions, variables, classes, and constants.
- Names should describe the real role of the object, not just its type.
- Avoid vague names such as `data`, `temp`, `info`, `result`, or `output` when the meaning is not obvious.
- Use consistent names for the same concept across the project.
- Keep dataset names, model names, metric names, and output file names consistent with the required CSV formats.
- Do not rename required columns such as `id`, `text`, and `label` unless the user explicitly asks.

### 13.4 Function and Module Design

- A function should do one main thing and expose clear inputs and outputs.
- Avoid long functions that mix loading, preprocessing, training, evaluation, and saving.
- Keep data loading, preprocessing, model training, evaluation, result saving, and plotting as separable steps.
- Reusable logic should be placed in `algorithms/utils/` or an appropriate shared module.
- Model scripts should keep a consistent command-line interface and output format.
- Avoid unnecessary global variables. Use function arguments, configuration values, or command-line arguments instead.
- Fix random seeds where applicable to support reproducibility.
- Create output directories automatically before saving files.

### 13.5 Data, Type, and Interface Rules

- Validate required columns before using a dataset.
- Validate train/validation/test paths before training or evaluation.
- Keep model input and output formats stable across all four algorithms.
- Use type hints for important functions, especially utility functions shared by multiple scripts.
- Use explicit data structures when returning multiple values. Avoid returning unclear tuples or mixed dictionaries.
- Do not silently change label mappings, split ratios, random seeds, or output schemas.
- Do not use the test set for training, vectorizer fitting, embedding training, hyperparameter tuning, or early stopping.

### 13.6 Error Handling Rules

- Handle errors explicitly and provide useful error messages.
- Do not silently swallow exceptions.
- Avoid broad `except Exception` blocks unless there is a clear reason and the error is re-raised or reported.
- If a required file, column, directory, or dependency is missing, fail clearly with a message that helps locate the problem.
- Do not hide data quality problems by automatically dropping large amounts of data without reporting the reason.
- Preserve the original exception context when re-raising errors.

### 13.7 Dependency and Configuration Rules

- Do not add new dependencies unless they are necessary for the coursework implementation.
- Prefer the Python standard library and existing dependencies when they are enough.
- If dependencies change, update `requirements.txt` and both README files when needed.
- Do not hard-code API keys, local machine paths, private dataset locations, or personal environment settings.
- Do not create a new virtual environment or change the expected runtime environment unless the user explicitly asks.

### 13.8 Comment and Docstring Rules

- Important modules must include a module-level docstring explaining the module purpose and responsibility.
- Important public functions and complex internal helper functions must include docstrings.
- Docstrings should explain intent, parameters, return values, important assumptions, and possible errors when relevant.
- Comments should explain why the logic exists, not repeat what the code already says.
- Add short comments for non-obvious logic, such as stratified splitting, duplicate removal, label mapping, random seed control, and metric aggregation.
- Keep comments accurate and update them when code behavior changes.
- Do not add outdated, vague, or misleading comments.
- Do not use comments to describe unsupported results or claims that are not backed by saved CSV outputs.

Example docstring:

```python
def load_dataset(file_path: str) -> pandas.DataFrame:
    """Load and validate a processed dataset CSV file.

    Args:
        file_path: Path to a CSV file containing at least `id`, `text`, and `label` columns.

    Returns:
        A DataFrame with the required columns validated.

    """
```

### 13.9 Testing and Verification Rules

- New utility functions should be easy to test independently.
- After changing preprocessing, splitting, evaluation, aggregation, or plotting logic, run the related script or test command.
- After changing a model script, verify that it can run from the project root with the documented command-line arguments.
- Do not claim that a model, metric, table, or figure was generated unless the corresponding output file exists.
- If verification cannot be run, explain the reason in the development report.

---

## 14. Documentation Synchronisation Rule

This project must maintain two README files:

```text
README.md       English version
README_cn.md    Chinese version
```

After every completed development task, Agents must check whether the change affects:

- project structure;
- installation steps;
- dependencies;
- dataset format;
- preprocessing workflow;
- model training commands;
- evaluation commands;
- output file paths;
- result interpretation;
- known limitations.

If yes, update both files.

Do not update only one README unless the user explicitly asks.

At the end of each task, report:

```text
README.md updated: yes/no
README_cn.md updated: yes/no
Reason:
```

---

## 15. Report Material Rule

The final LaTeX group report will depend on the saved outputs.

Agents should prepare reusable report materials:

- dataset statistics tables;
- label distribution tables;
- model comparison tables;
- figures generated from CSV results;
- short notes explaining major observations.

Do not write final report claims that are not supported by saved CSV files.

---

## 16. Prohibited Actions

Agents must not:

1. Change the required dataset columns `id,text,label` without permission.
2. Use only one dataset.
3. Implement only part of the four required algorithms.
4. Evaluate different models on different train/test splits.
5. Use the test set during training or hyperparameter tuning.
6. Report only accuracy.
7. Save results only as terminal output.
8. Hard-code local absolute paths.
9. Delete raw data files without permission.
10. Overwrite existing results without clear naming or user approval.
11. Add large generated files or caches into the final submission structure unless needed.
12. Modify the final report with unsupported claims.
13. Update only English README while ignoring Chinese README, or vice versa.
14. Create unrelated framework code that makes the coursework harder to run.

---

## 17. Completion Report Format

After each development task, Agents must report:

```markdown
# Development Report

## Completed Work
- ...

## Modified Files
- ...

## Created Files
- ...

## How to Run
Use bash commands here.

## Output Files
- ...

## Validation Performed
- ...

## README Sync
- README.md updated: yes/no
- README_cn.md updated: yes/no
- Reason:

## Remaining Issues
- ...

## Next Suggested Step
- ...
```

---

## 18. Final Submission Awareness

The final submission should be packaged as:

```text
TeamID_Coursework.zip
```

It should contain:

- cover letter with group member information;
- final PDF reports;
- individual literature review PDFs;
- group report PDF;
- `algorithms/` folder;
- `data/` folder;
- README file;
- dependency information.

Agents should keep the project clean so that the final zip does not contain unnecessary cache files, temporary outputs, development logs, or local environment folders.
