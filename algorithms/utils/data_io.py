"""Shared CSV input/output helpers for the coursework pipeline.

This module centralises small file-system and CSV operations so preprocessing,
model, and evaluation scripts can keep their own responsibilities focused.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import ParserError


def load_csv_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset from disk and raise clear errors for invalid paths.

    Parameters:
        path: Path to a CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or the CSV cannot be parsed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        return pd.read_csv(file_path)
    except (ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Failed to read CSV file {file_path}: {exc}") from exc


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Parameters:
        dataframe: DataFrame to validate.
        required_columns: Required column names.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Existing columns: {list(dataframe.columns)}"
        )


def ensure_output_dir(path: str | Path) -> Path:
    """
    Create an output directory if needed and return it as a Path.

    Parameters:
        path: Directory path to create.

    Returns:
        Created or existing directory path.
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataframe(dataframe: pd.DataFrame, path: str | Path) -> None:
    """
    Save a DataFrame as CSV, creating the parent directory automatically.

    Parameters:
        dataframe: DataFrame to save.
        path: Destination CSV path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
