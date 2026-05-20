# Dataset Information

## Dataset Name
AG News - Sports vs Business Classification

## Source
- **Original Dataset**: AG News Corpus
- **Paper**: Xiang Zhang, Junbo Zhao, Yann LeCun (2015). "Character-level Convolutional Networks for Text Classification"
- **Hugging Face URL**: https://huggingface.co/datasets/ag_news
- **Original GitHub**: https://github.com/mhjabreel/CharCnn_Keras

## Collection Method
The dataset was collected from AG News, a large-scale news corpus containing news articles from various sources. The articles were originally categorized into 4 classes: World, Sports, Business, and Science/Technology. For this binary classification task, we selected only the Sports and Business categories.

## Classification Scenario
**Document Topic Classification** - News Topic Classification

This is a binary text classification task where the goal is to classify news articles into two categories:
- **Sports**: News articles related to sports events, teams, athletes, and sporting activities
- **Business**: News articles related to business, finance, economics, and corporate activities

## Number of Instances
- **Total Samples**: 60,000
- **Sports Class**: 30,000 samples
- **Business Class**: 30,000 samples

## Number of Labels
2 labels (binary classification)

## Text Field Description
The `text` field contains the concatenated news title and description. Each article typically includes:
- News title (headline)
- Brief description or summary of the article
- Source information (e.g., "Reuters", "AP")

The text is in English and represents authentic journalistic writing styles.

## Label Field Description
The `label` field indicates the category of the news article:
- **sports**: News articles about sports events, teams, and athletes
- **business**: News articles about business, finance, and corporate activities

## Label List
1. `sports` - Sports news articles
2. `business` - Business news articles

## Known Data Quality Issues
1. **Duplicate Content**: Some articles may have similar or duplicate content due to news aggregation from multiple sources.
2. **HTML Tags**: Some descriptions may contain HTML tags that were not fully removed during preprocessing.
3. **URLs**: Some articles may contain URLs in the text that do not contribute to topic classification.
4. **Special Characters**: Some articles may contain special characters or encoding issues from the original source.

## Reason for Selection
1. **Sample Size**: The dataset contains 60,000 samples, well exceeding the minimum requirement of 3,000 samples.
2. **Balanced Distribution**: Each class has exactly 30,000 samples, ensuring balanced classification.
3. **Real-World Application**: News articles represent authentic journalistic writing styles.
4. **Standard Benchmark**: AG News is a widely used benchmark for text classification research.
5. **Clear Topic Distinction**: Sports and Business categories have distinct vocabulary and content, making them suitable for binary classification.
6. **English Language**: The dataset is in English, which is the target language for this coursework.
