# Dataset Information

## Dataset Name
Yahoo Answers Topics

## Source
Yahoo Answers 10 categories for NLP CSV public dataset. The original downloaded source files were used to prepare this standardized subset and are not kept in the repository.

## Collection Method
The original public dataset was manually downloaded. A balanced subset was sampled with a fixed random seed.

## Classification Scenario
Community Q&A topic classification.

## Number of Instances
6000

## Number of Labels
10

## Text Field Description
The text field is constructed by concatenating the question title, question content, and best answer when available.

## Label Field Description
The label field represents the main Yahoo Answers topic category of each Q&A document.

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

## Known Data Quality Issues
User-generated text may contain informal expressions, spelling mistakes, short questions, noisy answers, repeated content, URLs, HTML-like marks, or special characters.

## Reason for Selection
This dataset represents community Q&A topic classification, which is different from dataset_2 news topic classification.
