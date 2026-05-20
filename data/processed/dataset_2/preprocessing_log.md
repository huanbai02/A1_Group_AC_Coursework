# Preprocessing Log

## Dataset: dataset_2
## Timestamp: 2026-05-19 16:53:35

## Raw Data
- Total samples: 60,000
- Empty texts: 0
- Empty labels: 0

## Preprocessing Steps
1. Lowercase conversion
2. URL removal
3. HTML tag removal
4. Punctuation and number removal
5. Tokenization
6. Stopword removal
7. Porter stemming

## Results
- Samples after cleaning: 60,000
- Vocabulary size: 45,372
- Average document length: 34.1 words

## Split Statistics
- Train: 42,000 samples (70.0%)
- Validation: 9,000 samples (15.0%)
- Test: 9,000 samples (15.0%)

## Statistics Computation Notes
Statistics are computed on the full processed dataset by combining train.csv, val.csv, and test.csv. The text field used for statistics is the processed text column.
