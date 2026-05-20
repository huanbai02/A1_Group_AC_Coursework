# Dataset Information

## Dataset Name
IMDB Dataset of 50K Movie Reviews

## Source
Maas et al. (2011), Stanford Large Movie Review Dataset.Sourced via the Kaggle dataset equivalent: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Collection Method
Formally collected from the Hugging Face `imdb` dataset to merge original train and test sets into a unified 50,000-review dataset, maintaining the identical content and format as the Kaggle version.

## Classification Scenario
Document-level sentiment classification (binary classification: positive vs. negative review).

## Number of Instances
50,000

## Number of Labels
2

## Text Field Description
`text`: Contains the raw English text of the movie review.

## Label Field Description
`label`: Mapped binary sentiment. `0` indicates negative sentiment, and `1` indicates positive sentiment.

## Label List
- `0` (negative)
- `1` (positive)

## Known Data Quality Issues
Contains HTML formatting tags (e.g., `<br />`) and standard online review informalities (abbreviations, slang, punctuation noise) which require preprocessing/cleaning.

## Reason for Selection
Highly standard NLP benchmark dataset with balanced labels, sufficient instance count (>3,000 as required by coursework rules), and clear evaluation criteria for comparing traditional (SVM) and deep learning (BERT) algorithms.
