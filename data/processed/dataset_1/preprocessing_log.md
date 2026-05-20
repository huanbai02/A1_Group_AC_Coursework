# Preprocessing Log: Yahoo Answers Topics

## Source Files
The standardized subset was prepared from the Yahoo Answers Topics `train.csv`,
`test.csv`, and `classes.txt` files. The original downloaded source directory is
not kept in the repository.

## Output File Paths
- Raw output directory: `data/raw/dataset_1`
- Processed output directory: `data/processed/dataset_1`
- Cleaned file: `data/processed/dataset_1/cleaned.csv`
- Train file: `data/processed/dataset_1/train.csv`
- Validation file: `data/processed/dataset_1/val.csv`
- Test file: `data/processed/dataset_1/test.csv`

## Text Construction Rule
`text = question_title + ' ' + question_content + ' ' + best_answer`.

## Preprocessing Steps
- Parsed local Yahoo Answers train/test CSV files.
- Mapped numeric class indices to unified snake_case labels from classes.txt.
- Constructed text from title, question content, and best answer.
- Removed records with empty constructed or cleaned text.
- Removed duplicate cleaned text globally before sampling.
- Sampled 600 cleaned unique records per label with a fixed random seed.
- Applied basic text cleaning for `cleaned.csv` and split files.
- Created stratified train/validation/test splits.

## Removal Counts
- Source rows read: 1459998
- Invalid column rows: 0
- Invalid label rows: 0
- Removed empty text count: 5
- Removed duplicate text count: 309

## Final Counts
- Raw sample count: 6000
- Cleaned sample count: 6000
- Train rows: 4200
- Validation rows: 900
- Test rows: 900

## Split Configuration
- Split ratio: 70 / 15 / 15
- Random seed: 42
- Stratified split: yes

## Cross-Split Overlap
- Train/validation ID overlap: 0
- Train/test ID overlap: 0
- Validation/test ID overlap: 0
- Train/validation text overlap: 0
- Train/test text overlap: 0
- Validation/test text overlap: 0
- Cross-split overlap exists: no
