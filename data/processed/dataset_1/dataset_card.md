# Dataset Card: Yahoo Answers Topics

## Dataset Name
Yahoo Answers Topics

## Dataset ID
dataset_1

## Source
Yahoo Answers Topics. The original downloaded source files were used to prepare this standardized subset and are not kept in the repository.

## Classification Scenario
community Q&A topic classification

## Raw Sample Count
6000

## Cleaned Sample Count
6000

## Number of Labels
10

## Label List
- society_culture
- science_mathematics
- health
- education_reference
- computers_internet
- sports
- business_finance
- entertainment_music
- family_relationships
- politics_government

## Train / Validation / Test Split
- Train: 4200
- Validation: 900
- Test: 900
- Split ratio: 70 / 15 / 15
- Random seed: 42
- Stratified split: yes

## Statistical Scope
Statistics are computed over the combined processed train, validation, and test splits.
Word frequency is saved for all tokens in the combined processed dataset.

## Known Limitations
The source is user-generated Q&A text and may contain informal language, spelling mistakes, short questions, noisy answers, URLs, HTML-like marks, and special characters.
The dataset is a balanced coursework subset and should not be interpreted as the original full Yahoo Answers class distribution.
