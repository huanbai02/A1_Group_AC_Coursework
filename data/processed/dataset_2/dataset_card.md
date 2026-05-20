# Dataset Card: dataset_2

## Dataset Name
AG News - Sports vs Business Classification

## Source
- Original dataset: AG News Corpus
- Hugging Face URL: https://huggingface.co/datasets/ag_news
- Original task adapted to binary news topic classification by selecting Sports and Business samples.

## Classification Scenario
News Topic Classification

## Number of Instances
- Raw rows: 60,000
- Cleaned rows: 59,952
- Train rows: 41,966
- Validation rows: 8,993
- Test rows: 8,993

## Number of Labels
2

## Label List
- `sports`
- `business`

## Text Field
The `text` field contains cleaned English news title and description content.
Text was lowercased, URLs were removed, selected punctuation was removed, and
whitespace was normalized by the shared `basic_clean_text` utility.

## Preprocessing Steps
1. Validated the raw dataset delivery files and required columns.
2. Removed rows with empty `id`, `text`, or `label` values.
3. Removed duplicate IDs.
4. Removed duplicate cleaned text.
5. Saved the cleaned dataset with the required `id,text,label` columns.
6. Created stratified train/validation/test splits using seed 42.
7. Generated full processed dataset statistics from `train.csv + val.csv + test.csv`.

## Removal Summary
- Removed empty rows: 0
- Removed duplicate ID rows: 0
- Removed duplicate text rows: 48
- Total removed rows: 48

## Split Configuration
- Split ratio: 70% train, 15% validation, 15% test
- Random seed: 42
- Stratification: label-based stratified split
- Cross-split ID overlap: 0
- Cross-split text overlap: 0

## Split Label Distribution
| Split | Sports | Business | Total |
| --- | ---: | ---: | ---: |
| train | 20,989 | 20,977 | 41,966 |
| val | 4,498 | 4,495 | 8,993 |
| test | 4,498 | 4,495 | 8,993 |

## Full Statistics Summary
| Statistic | Value |
| --- | ---: |
| num_instances | 59,952 |
| num_labels | 2 |
| vocabulary_size | 51,655 |
| avg_doc_length | 54.8935 |
| min_doc_length | 18 |
| max_doc_length | 177 |

## Known Limitations
- The task is a binary subset of AG News rather than the original four-class benchmark.
- Some articles may remain near-duplicates after exact duplicate cleaned text removal.
- The cleaned text is suitable for traditional models and Word2Vec; BERT can still use this column unless a lighter-cleaned `bert_text` column is added later.
