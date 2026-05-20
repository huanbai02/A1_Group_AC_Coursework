# Dataset Information

## Dataset Name
News Category Dataset (Modified Subset)

## Source
Rishabh Misra (2012-2022). HuffPost news headlines dataset. Sourced via Hugging Face `heegyu/news-category-dataset`.

## Collection Method
Filtered from the original News Category Dataset to extract exactly six categories: POLITICS, WELLNESS, ENTERTAINMENT, TRAVEL, STYLE & BEAUTY, and PARENTING. For each category, exactly 8,000 instances were randomly sampled (random seed 42) to form a balanced subset of 48,000 instances in total. The text field was constructed by concatenating the headline and the short description.

## Classification Scenario
Document category classification (six-class single-label text classification).

## Number of Instances
48,000

## Number of Labels
6

## Text Field Description
`text`: The combined text of the news headline and its short description (separated by a space).

## Label Field Description
`label`: The news category string (one of: POLITICS, WELLNESS, ENTERTAINMENT, TRAVEL, STYLE & BEAUTY, PARENTING).

## Label List
- POLITICS
- WELLNESS
- ENTERTAINMENT
- TRAVEL
- STYLE & BEAUTY
- PARENTING

## Known Data Quality Issues
Some headlines or short descriptions may contain HTML entity characters or contractions. The short descriptions for some old news articles might be empty, in which case the text field relies solely on the headline.

## Reason for Selection
Replaces the previous IMDB movie reviews dataset to represent a news topic classification scenario with 6 balanced classes and 48,000 total instances, satisfying the coursework requirement (>3000 instances per dataset).
