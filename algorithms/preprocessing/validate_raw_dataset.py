"""Validate the raw dataset delivery format required by the coursework."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

REQUIRED_FILES = [
    "raw_data.csv",
    "sample_100.csv",
    "dataset_info.md",
    "label_mapping.csv",
    "initial_label_distribution.csv",
]
RAW_COLUMNS = ["id", "text", "label"]
LABEL_MAPPING_COLUMNS = ["original_label", "unified_label", "description"]
DISTRIBUTION_COLUMNS = ["label", "count", "percentage"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate a raw dataset folder.")
    parser.add_argument("--dataset_dir", required=True, help="Path to a raw dataset directory.")
    parser.add_argument("--report_dir", default="notes", help="Directory for validation reports.")
    return parser.parse_args()


def check_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    file_name: str,
    errors: list[str],
) -> None:
    """Append an error if the DataFrame is missing required columns."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        errors.append(f"{file_name}: missing required columns {missing_columns}")


def contains_empty_values(series: pd.Series) -> bool:
    """Return True if a CSV column contains null or blank values."""
    return series.isna().any() or (series.astype(str).str.strip() == "").any()


def validate_raw_data(raw_dataframe: pd.DataFrame, errors: list[str]) -> None:
    """Validate raw_data.csv schema, size, ID, text, and label rules."""
    check_columns(raw_dataframe, RAW_COLUMNS, "raw_data.csv", errors)
    if len(raw_dataframe) < 3000:
        errors.append(f"raw_data.csv: expected at least 3000 rows, found {len(raw_dataframe)}")

    if not set(RAW_COLUMNS).issubset(raw_dataframe.columns):
        return

    if contains_empty_values(raw_dataframe["id"]):
        errors.append("raw_data.csv: id contains empty values")
    duplicate_id_count = int(raw_dataframe["id"].duplicated().sum())
    if duplicate_id_count:
        errors.append(f"raw_data.csv: id contains {duplicate_id_count} duplicates")
    if contains_empty_values(raw_dataframe["text"]):
        errors.append("raw_data.csv: text contains empty values")
    if contains_empty_values(raw_dataframe["label"]):
        errors.append("raw_data.csv: label contains empty values")


def validate_initial_distribution(
    distribution_dataframe: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate initial_label_distribution.csv columns and percentage total."""
    check_columns(
        distribution_dataframe,
        DISTRIBUTION_COLUMNS,
        "initial_label_distribution.csv",
        errors,
    )
    if "percentage" not in distribution_dataframe.columns:
        return

    total_percentage = pd.to_numeric(
        distribution_dataframe["percentage"],
        errors="coerce",
    ).sum()
    if not 99.0 <= total_percentage <= 101.0:
        warnings.append(
            "initial_label_distribution.csv: percentage sum is "
            f"{total_percentage:.2f}, expected close to 100"
        )


def validate_dataset(dataset_dir: Path) -> tuple[list[str], list[str]]:
    """Validate one dataset directory and return errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return [f"Dataset directory does not exist or is not a directory: {dataset_dir}"], warnings

    for file_name in REQUIRED_FILES:
        if not (dataset_dir / file_name).exists():
            errors.append(f"Missing required file: {file_name}")

    raw_path = dataset_dir / "raw_data.csv"
    if raw_path.exists():
        validate_raw_data(pd.read_csv(raw_path), errors)

    sample_path = dataset_dir / "sample_100.csv"
    if sample_path.exists():
        sample_dataframe = pd.read_csv(sample_path)
        check_columns(sample_dataframe, RAW_COLUMNS, "sample_100.csv", errors)
        if len(sample_dataframe) != 100:
            errors.append(f"sample_100.csv: expected exactly 100 rows, found {len(sample_dataframe)}")

    mapping_path = dataset_dir / "label_mapping.csv"
    if mapping_path.exists():
        check_columns(
            pd.read_csv(mapping_path),
            LABEL_MAPPING_COLUMNS,
            "label_mapping.csv",
            errors,
        )

    distribution_path = dataset_dir / "initial_label_distribution.csv"
    if distribution_path.exists():
        validate_initial_distribution(pd.read_csv(distribution_path), errors, warnings)

    return errors, warnings


def write_report(
    dataset_dir: Path,
    errors: list[str],
    warnings: list[str],
    report_dir: Path,
) -> Path:
    """Write a Markdown validation report under notes/."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{dataset_dir.name or 'dataset'}_raw_validation_report.md"
    status = "FAILED" if errors else "PASSED"
    lines = [
        "# Raw Dataset Validation Report",
        "",
        f"Dataset directory: `{dataset_dir}`",
        f"Validation time: {datetime.now().isoformat(timespec='seconds')}",
        f"Status: **{status}**",
        "",
        "## Errors",
    ]
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    """Run raw dataset validation from the command line."""
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    errors, warnings = validate_dataset(dataset_dir)
    report_path = write_report(dataset_dir, errors, warnings, Path(args.report_dir))

    print(f"Dataset directory: {dataset_dir}")
    print(f"Validation report: {report_path}")
    if errors:
        print("Status: FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        for warning in warnings:
            print(f"WARNING: {warning}")
        raise SystemExit(1)

    print("Status: PASSED")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
