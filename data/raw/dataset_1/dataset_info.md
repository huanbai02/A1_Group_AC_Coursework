# IMDB Dataset of 50K Movie Reviews

## Data Source
The data originates from the Stanford Large Movie Review Dataset, provided by Maas et al. (2011). It contains 50,000 movie reviews for natural language processing or Text analytics.
This package was specifically formatted to mirror the Kaggle dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
To ensure accessibility and avoid API authentication issues, the raw data was sourced from the equivalent `imdb` dataset on Hugging Face Datasets, which contains the exact same 50,000 reviews (25,000 for training, 25,000 for testing).

## Files Included
- `raw_data.csv`: The complete dataset of 50,000 reviews, merging the original training and testing sets. Contains `id`, `text`, and `label` columns.
- `sample_100.csv`: A random sample of 100 records from the `raw_data.csv` for quick data quality inspection.
- `label_mapping.csv`: The mapping between the original sentiment text labels (`positive`, `negative`) and the unified integer labels (`1`, `0`).
- `initial_label_distribution.csv`: Statistics on the label distribution across the entire dataset.

## Label Information
The unified labels are represented as integers:
- `0`: Negative sentiment
- `1`: Positive sentiment
