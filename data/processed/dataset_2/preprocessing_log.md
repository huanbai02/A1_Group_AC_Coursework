# Preprocessing Log: dataset_2

## Input
- Source file: `data/raw/dataset_2/raw_data.csv`
- Raw rows: 60,000
- Required columns: `id,text,label`

## Raw Validation
- Command: `python algorithms/preprocessing/validate_raw_dataset.py --dataset_dir data/raw/dataset_2`
- Result: passed
- Report: `notes/dataset_2_raw_validation_report.md`

## Cleaning
- Command: `python algorithms/preprocessing/preprocess_dataset.py --input_path data/raw/dataset_2/raw_data.csv --output_path data/processed/dataset_2/cleaned.csv`
- Basic cleaning: lowercase, URL removal, punctuation normalization, whitespace normalization.
- Removed empty rows: 0
- Removed duplicate ID rows: 0
- Removed duplicate text rows: 48
- Total removed rows: 48
- Cleaned rows: 59,952

## Splitting
- Command: `python algorithms/preprocessing/split_dataset.py --input_path data/processed/dataset_2/cleaned.csv --dataset_name dataset_2 --output_dir data/processed/dataset_2 --seed 42 --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15`
- Split ratio: train 70%, validation 15%, test 15%
- Random seed: 42
- Stratification: label-based stratified split

| Split | Rows | sports | business |
| --- | ---: | ---: | ---: |
| train | 41,966 | 20,989 | 20,977 |
| val | 8,993 | 4,498 | 4,495 |
| test | 8,993 | 4,498 | 4,495 |

## Overlap Checks
- train/val ID overlap: 0
- train/test ID overlap: 0
- val/test ID overlap: 0
- train/val text overlap: 0
- train/test text overlap: 0
- val/test text overlap: 0

## Full Statistics
- Command: `python algorithms/preprocessing/dataset_statistics.py --input_dir data/processed/dataset_2 --dataset_name dataset_2 --output_dir data/processed/dataset_2`
- Scope: full processed dataset (`train.csv + val.csv + test.csv`)
- `split`: `full`
- `text_version`: `text`

| num_instances | num_labels | vocabulary_size | avg_doc_length | min_doc_length | max_doc_length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 59,952 | 2 | 51,655 | 54.8935 | 18 | 177 |
