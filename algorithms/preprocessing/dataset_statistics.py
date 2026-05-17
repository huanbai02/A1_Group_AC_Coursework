"""Generate basic statistics for a processed dataset CSV.

The statistics are derived from the input CSV only. In the recommended workflow,
this input is the training split so downstream report materials are explicit
about what has been measured.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from algorithms.utils.data_io import (
    ensure_output_dir,
    load_csv_dataset,
    save_dataframe,
    validate_required_columns,
)
from algorithms.utils.text_processing import simple_tokenize

REQUIRED_COLUMNS = ["id", "text", "label"]
STATISTICS_COLUMNS = [
    "dataset",
    "num_instances",
    "num_labels",
    "vocabulary_size",
    "avg_doc_length",
    "min_doc_length",
    "max_doc_length",
]
LABEL_DISTRIBUTION_COLUMNS = ["dataset", "label", "count", "percentage"]
WORD_FREQUENCY_COLUMNS = ["dataset", "word", "count"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate dataset statistics CSV files.")
    parser.add_argument(
        "--input_path",
        required=True,
        help="Input processed CSV path, usually train.csv.",
    )
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--output_dir", required=True, help="Output directory for statistics files.")
    return parser.parse_args()


def build_statistics(dataframe: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build statistics, label distribution, and word-frequency DataFrames.

    Parameters:
        dataframe: Processed dataset with id, text, and label columns.
        dataset_name: Dataset identifier written to output CSV files.

    Returns:
        statistics, label distribution, and word-frequency DataFrames.
    """
    validate_required_columns(dataframe, REQUIRED_COLUMNS)

    tokenized_documents = [simple_tokenize(text) for text in dataframe["text"]]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    vocabulary = {token for tokens in tokenized_documents for token in tokens}
    word_counts = Counter(token for tokens in tokenized_documents for token in tokens)

    statistics_dataframe = build_statistics_dataframe(
        dataframe,
        dataset_name,
        vocabulary_size=len(vocabulary),
        document_lengths=document_lengths,
    )
    label_distribution_dataframe = build_label_distribution_dataframe(dataframe, dataset_name)
    word_frequency_dataframe = build_word_frequency_dataframe(word_counts, dataset_name)
    return statistics_dataframe, label_distribution_dataframe, word_frequency_dataframe


def build_statistics_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    vocabulary_size: int,
    document_lengths: list[int],
) -> pd.DataFrame:
    """Create the one-row dataset statistics DataFrame."""
    average_document_length = 0.0
    min_document_length = 0
    max_document_length = 0
    if document_lengths:
        average_document_length = round(sum(document_lengths) / len(document_lengths), 4)
        min_document_length = min(document_lengths)
        max_document_length = max(document_lengths)

    row = {
        "dataset": dataset_name,
        "num_instances": len(dataframe),
        "num_labels": dataframe["label"].nunique(),
        "vocabulary_size": vocabulary_size,
        "avg_doc_length": average_document_length,
        "min_doc_length": min_document_length,
        "max_doc_length": max_document_length,
    }
    return pd.DataFrame([row], columns=STATISTICS_COLUMNS)


def build_label_distribution_dataframe(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Create label count and percentage statistics."""
    label_distribution = dataframe["label"].value_counts().reset_index()
    label_distribution.columns = ["label", "count"]
    label_distribution.insert(0, "dataset", dataset_name)
    if len(dataframe) > 0:
        label_distribution["percentage"] = (
            label_distribution["count"] / len(dataframe) * 100
        ).round(4)
    else:
        label_distribution["percentage"] = 0.0
    return label_distribution[LABEL_DISTRIBUTION_COLUMNS]


def build_word_frequency_dataframe(word_counts: Counter[str], dataset_name: str) -> pd.DataFrame:
    """Create a word-frequency DataFrame sorted by descending count."""
    rows = [
        {"dataset": dataset_name, "word": word, "count": count}
        for word, count in word_counts.most_common()
    ]
    return pd.DataFrame(rows, columns=WORD_FREQUENCY_COLUMNS)


def main() -> None:
    """Run statistics generation from the command line."""
    args = parse_args()
    processed_dataframe = load_csv_dataset(args.input_path)
    statistics_dataframe, label_distribution_dataframe, word_frequency_dataframe = build_statistics(
        processed_dataframe,
        args.dataset_name,
    )

    output_dir = ensure_output_dir(args.output_dir)
    save_dataframe(statistics_dataframe, output_dir / "statistics.csv")
    save_dataframe(label_distribution_dataframe, output_dir / "label_distribution.csv")
    save_dataframe(word_frequency_dataframe, output_dir / "word_frequency.csv")

    print(f"Saved statistics to: {output_dir}")
    print(statistics_dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
