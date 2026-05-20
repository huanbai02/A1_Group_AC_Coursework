"""Generate basic statistics for processed dataset CSV files.

The script supports two explicit scopes:

- ``--input_dir`` (recommended): read and combine ``train.csv``, ``val.csv``,
  and ``test.csv`` from a processed dataset directory. This is the standard
  report-ready scope for ``statistics.csv``, ``label_distribution.csv``, and
  ``word_frequency.csv``.
- ``--input_path``: read a single processed CSV file. Use this for split-only
  diagnostics such as train-only statistics.
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
PROCESSED_SPLIT_FILES = ["train.csv", "val.csv", "test.csv"]
STATISTICS_COLUMNS = [
    "dataset",
    "split",
    "text_version",
    "num_instances",
    "num_labels",
    "vocabulary_size",
    "avg_doc_length",
    "min_doc_length",
    "max_doc_length",
]
LABEL_DISTRIBUTION_COLUMNS = ["dataset", "split", "label", "count", "percentage"]
WORD_FREQUENCY_COLUMNS = ["dataset", "split", "word", "count"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate dataset statistics CSV files.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_path",
        help=(
            "Single processed CSV path for split-specific statistics, "
            "for example data/processed/dataset_1/train.csv."
        ),
    )
    input_group.add_argument(
        "--input_dir",
        help=(
            "Processed dataset directory containing train.csv, val.csv, and test.csv. "
            "Recommended for report-ready full processed dataset statistics."
        ),
    )
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--output_dir", required=True, help="Output directory for statistics files.")
    parser.add_argument(
        "--text_column",
        default="text",
        help="Text column to analyse. Defaults to the processed text column.",
    )
    parser.add_argument(
        "--split_name",
        default=None,
        help=(
            "Optional split label for --input_path output. If omitted, the input "
            "file stem is used, such as train, val, or test."
        ),
    )
    return parser.parse_args()


def load_processed_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Load either a single CSV or the full processed train/val/test directory."""
    if args.input_dir:
        return load_processed_directory(args.input_dir), "full"

    input_path = Path(args.input_path)
    split_name = args.split_name or input_path.stem
    return load_csv_dataset(input_path), split_name


def load_processed_directory(input_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate train.csv, val.csv, and test.csv from a directory."""
    processed_dir = Path(input_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory does not exist: {processed_dir}")
    if not processed_dir.is_dir():
        raise ValueError(f"Path is not a directory: {processed_dir}")

    split_dataframes = []
    for split_file in PROCESSED_SPLIT_FILES:
        split_path = processed_dir / split_file
        split_dataframe = load_csv_dataset(split_path)
        validate_required_columns(split_dataframe, REQUIRED_COLUMNS)
        split_dataframes.append(split_dataframe)

    return pd.concat(split_dataframes, ignore_index=True)


def build_statistics(
    dataframe: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    text_column: str = "text",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build statistics, label distribution, and word-frequency DataFrames.

    Parameters:
        dataframe: Processed dataset with id, text, and label columns.
        dataset_name: Dataset identifier written to output CSV files.
        split_name: Statistical scope label, such as full or train.
        text_column: Text column used for tokenisation and text statistics.

    Returns:
        statistics, label distribution, and word-frequency DataFrames.
    """
    validate_required_columns(dataframe, REQUIRED_COLUMNS)
    if text_column not in dataframe.columns:
        raise ValueError(
            f"Text column does not exist: {text_column}. "
            f"Existing columns: {list(dataframe.columns)}"
        )

    text_series = dataframe[text_column].fillna("").astype(str)
    tokenized_documents = [simple_tokenize(text) for text in text_series]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    vocabulary = {token for tokens in tokenized_documents for token in tokens}
    word_counts = Counter(token for tokens in tokenized_documents for token in tokens)

    statistics_dataframe = build_statistics_dataframe(
        dataframe,
        dataset_name,
        split_name,
        text_version=text_column,
        vocabulary_size=len(vocabulary),
        document_lengths=document_lengths,
    )
    label_distribution_dataframe = build_label_distribution_dataframe(dataframe, dataset_name, split_name)
    word_frequency_dataframe = build_word_frequency_dataframe(word_counts, dataset_name, split_name)
    return statistics_dataframe, label_distribution_dataframe, word_frequency_dataframe


def build_statistics_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    text_version: str,
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
        "split": split_name,
        "text_version": text_version,
        "num_instances": len(dataframe),
        "num_labels": dataframe["label"].nunique(),
        "vocabulary_size": vocabulary_size,
        "avg_doc_length": average_document_length,
        "min_doc_length": min_document_length,
        "max_doc_length": max_document_length,
    }
    return pd.DataFrame([row], columns=STATISTICS_COLUMNS)


def build_label_distribution_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Create label count and percentage statistics."""
    label_distribution = dataframe["label"].value_counts().reset_index()
    label_distribution.columns = ["label", "count"]
    label_distribution.insert(0, "split", split_name)
    label_distribution.insert(0, "dataset", dataset_name)
    if len(dataframe) > 0:
        label_distribution["percentage"] = (
            label_distribution["count"] / len(dataframe) * 100
        ).round(4)
    else:
        label_distribution["percentage"] = 0.0
    return label_distribution.loc[:, LABEL_DISTRIBUTION_COLUMNS]


def build_word_frequency_dataframe(
    word_counts: Counter[str],
    dataset_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Create a word-frequency DataFrame sorted by descending count."""
    rows = [
        {"dataset": dataset_name, "split": split_name, "word": word, "count": count}
        for word, count in word_counts.most_common()
    ]
    return pd.DataFrame(rows, columns=WORD_FREQUENCY_COLUMNS)


def main() -> None:
    """Run statistics generation from the command line."""
    args = parse_args()
    processed_dataframe, split_name = load_processed_input(args)
    statistics_dataframe, label_distribution_dataframe, word_frequency_dataframe = build_statistics(
        processed_dataframe,
        args.dataset_name,
        split_name,
        text_column=args.text_column,
    )

    output_dir = ensure_output_dir(args.output_dir)
    save_dataframe(statistics_dataframe, output_dir / "statistics.csv")
    save_dataframe(label_distribution_dataframe, output_dir / "label_distribution.csv")
    save_dataframe(word_frequency_dataframe, output_dir / "word_frequency.csv")

    print(f"Saved statistics to: {output_dir}")
    print(statistics_dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
