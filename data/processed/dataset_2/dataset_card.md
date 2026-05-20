# Dataset Card: dataset_2

## Overview
This dataset contains 60,000 documents for document topic classification.

## Statistics
- **Total Samples**: 60,000
- **Training Samples**: 42,000
- **Validation Samples**: 9,000
- **Test Samples**: 9,000
- **Vocabulary Size**: 45,372
- **Average Document Length**: 34.1 words

## Label Distribution
| Label | Count | Percentage |
|-------|-------|------------|
| sports | 30000 | 50.0% |
| business | 30000 | 50.0% |

## Preprocessing
- Lowercase conversion
- URL removal
- HTML tag removal
- Punctuation and number removal
- Tokenization
- Stopword removal
- Porter stemming

## Train/Val/Test Split
- Train: 70%
- Validation: 15%
- Test: 15%
- Random seed: 42
- Stratified: Yes

## Statistics Computation Notes
Statistics are computed on the full processed dataset by combining train.csv, val.csv, and test.csv. The text field used for statistics is the processed text column.
