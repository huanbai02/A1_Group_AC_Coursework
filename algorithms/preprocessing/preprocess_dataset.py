"""Preprocess a raw document topic classification dataset.

The script performs only dataset-level cleaning required in the initialization
stage: required-column validation, basic text normalization, empty-row removal,
and duplicate removal. It does not perform model-specific feature engineering.
"""

import argparse
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from algorithms.utils.data_io import load_csv_dataset, save_dataframe, validate_required_columns
from algorithms.utils.text_processing import basic_clean_text

REQUIRED_COLUMNS = ["id", "text", "label"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean raw dataset text and remove invalid rows."
    )
    parser.add_argument("--input_path", required=True, help="Input raw_data.csv path.")
    parser.add_argument("--output_path", required=True, help="Output cleaned CSV path.")
    return parser.parse_args()


def preprocess_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Clean text, remove invalid rows, and remove duplicates.

    Parameters:
        dataframe: Raw DataFrame containing at least id, text, and label.

    Returns:
        A tuple of cleaned DataFrame and removal summary counts.
    """
    validate_required_columns(dataframe, REQUIRED_COLUMNS)

    cleaned_dataframe = dataframe.copy()
    cleaned_dataframe["id"] = cleaned_dataframe["id"].astype(str).str.strip()
    cleaned_dataframe["label"] = cleaned_dataframe["label"].astype(str).str.strip()
    cleaned_dataframe["text"] = cleaned_dataframe["text"].apply(basic_clean_text)

    initial_rows = len(cleaned_dataframe)
    valid_mask = (
        (cleaned_dataframe["id"] != "")
        & (cleaned_dataframe["text"] != "")
        & (cleaned_dataframe["label"] != "")
    )
    cleaned_dataframe = cleaned_dataframe[valid_mask]
    removed_empty_rows = initial_rows - len(cleaned_dataframe)

    before_id_deduplication = len(cleaned_dataframe)
    cleaned_dataframe = cleaned_dataframe.drop_duplicates(subset=["id"])
    removed_duplicate_ids = before_id_deduplication - len(cleaned_dataframe)

    before_text_label_deduplication = len(cleaned_dataframe)
    cleaned_dataframe = cleaned_dataframe.drop_duplicates(subset=["text", "label"])
    removed_duplicate_text_label = before_text_label_deduplication - len(cleaned_dataframe)

    removal_summary = {
        "removed_empty_rows": removed_empty_rows,
        "removed_duplicate_ids": removed_duplicate_ids,
        "removed_duplicate_text_label_rows": removed_duplicate_text_label,
        "total_removed_rows": initial_rows - len(cleaned_dataframe),
    }
    return cleaned_dataframe[REQUIRED_COLUMNS], removal_summary


def main() -> None:
    """Run preprocessing from the command line."""
    args = parse_args()
    raw_dataframe = load_csv_dataset(args.input_path)
    cleaned_dataframe, removal_summary = preprocess_dataframe(raw_dataframe)
    save_dataframe(cleaned_dataframe, args.output_path)

    print(f"Input rows: {len(raw_dataframe)}")
    print(f"Output rows: {len(cleaned_dataframe)}")
    for summary_name, count in removal_summary.items():
        print(f"{summary_name}: {count}")
    print(f"Saved cleaned dataset to: {args.output_path}")


if __name__ == "__main__":
    main()
