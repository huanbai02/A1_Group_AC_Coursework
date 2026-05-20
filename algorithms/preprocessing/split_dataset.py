"""Split a cleaned dataset into train, validation, and test CSV files.

The script validates the required dataset schema, removes invalid text/label rows,
and uses stratified splitting when label counts make it safe.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from algorithms.utils.data_io import (
    ensure_output_dir,
    load_csv_dataset,
    save_dataframe,
    validate_required_columns,
)
from algorithms.utils.seed import set_random_seed

REQUIRED_COLUMNS = ["id", "text", "label"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for one dataset."
    )
    parser.add_argument("--input_path", required=True, help="Input cleaned CSV path.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    parser.add_argument("--output_dir", required=True, help="Output directory for split CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio.")
    return parser.parse_args()


def get_stratification_labels(labels: pd.Series) -> pd.Series | None:
    """
    Return labels for stratification when every class has enough samples.

    Stratified splitting can fail if a class has too few samples. Returning None
    lets scikit-learn perform a normal random split in that edge case.
    """
    label_counts = labels.value_counts()
    if len(label_counts) > 1 and label_counts.min() >= 2:
        return labels
    return None


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate train/validation/test ratios before splitting."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be positive")


def prepare_dataframe_for_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, remove invalid rows, and deduplicate IDs before splitting."""
    validate_required_columns(dataframe, REQUIRED_COLUMNS)
    prepared_dataframe = dataframe.copy()
    prepared_dataframe["id"] = prepared_dataframe["id"].astype(str).str.strip()
    prepared_dataframe["text"] = prepared_dataframe["text"].astype(str).str.strip()
    prepared_dataframe["label"] = prepared_dataframe["label"].astype(str).str.strip()

    valid_rows = (
        (prepared_dataframe["id"] != "")
        & (prepared_dataframe["text"] != "")
        & (prepared_dataframe["label"] != "")
    )
    prepared_dataframe = prepared_dataframe[valid_rows].drop_duplicates(subset=["id"])
    if prepared_dataframe.empty:
        raise ValueError("No valid rows remain after removing empty id/text/label rows")
    return prepared_dataframe[REQUIRED_COLUMNS]


def split_dataset(
    dataframe: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train, validation, and test sets.

    Returns:
        Train, validation, and test DataFrames with stable required columns.
    """
    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    prepared_dataframe = prepare_dataframe_for_split(dataframe)

    train_dataframe, temporary_dataframe = train_test_split(
        prepared_dataframe,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=get_stratification_labels(prepared_dataframe["label"]),
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=get_stratification_labels(temporary_dataframe["label"]),
    )

    validate_no_id_overlap(train_dataframe, validation_dataframe, test_dataframe)
    validate_no_text_overlap(train_dataframe, validation_dataframe, test_dataframe)
    return train_dataframe, validation_dataframe, test_dataframe


def validate_no_id_overlap(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """Ensure the same ID does not appear in more than one split."""
    train_ids = set(train_dataframe["id"].astype(str))
    validation_ids = set(validation_dataframe["id"].astype(str))
    test_ids = set(test_dataframe["id"].astype(str))

    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Duplicate IDs found across train/validation/test splits")


def validate_no_text_overlap(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """Ensure duplicate text does not appear in more than one split."""
    train_texts = set(train_dataframe["text"].astype(str))
    validation_texts = set(validation_dataframe["text"].astype(str))
    test_texts = set(test_dataframe["text"].astype(str))

    if train_texts & validation_texts or train_texts & test_texts or validation_texts & test_texts:
        raise ValueError("Duplicate text found across train/validation/test splits")


def main() -> None:
    """Run dataset splitting from the command line."""
    args = parse_args()
    set_random_seed(args.seed)
    cleaned_dataframe = load_csv_dataset(args.input_path)
    train_dataframe, validation_dataframe, test_dataframe = split_dataset(
        cleaned_dataframe,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    output_dir = ensure_output_dir(args.output_dir)
    save_dataframe(train_dataframe, output_dir / "train.csv")
    save_dataframe(validation_dataframe, output_dir / "val.csv")
    save_dataframe(test_dataframe, output_dir / "test.csv")

    print(f"Dataset: {args.dataset_name}")
    print(f"Train rows: {len(train_dataframe)}")
    print(f"Validation rows: {len(validation_dataframe)}")
    print(f"Test rows: {len(test_dataframe)}")
    print(f"Saved splits to: {output_dir}")


if __name__ == "__main__":
    main()
