"""Aggregate existing metrics CSV files into one summary table."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate metrics CSV files.")
    parser.add_argument(
        "--metrics_dir",
        required=True,
        help="Directory containing metrics CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for all_metrics_summary.csv.",
    )
    return parser.parse_args()


def load_metrics_files(metrics_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate metrics CSV files in a stable order."""
    metrics_frames = [pd.read_csv(metrics_path) for metrics_path in metrics_paths]
    return pd.concat(metrics_frames, ignore_index=True)


def main() -> None:
    """Aggregate metrics files if any exist."""
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_dir.exists():
        print(f"Metrics directory does not exist: {metrics_dir}")
        return

    metrics_paths = sorted(metrics_dir.glob("*_metrics.csv"))
    if not metrics_paths:
        print(f"No metrics CSV files found in: {metrics_dir}")
        return

    summary_dataframe = load_metrics_files(metrics_paths)
    output_path = output_dir / "all_metrics_summary.csv"
    summary_dataframe.to_csv(output_path, index=False)

    print(f"Aggregated {len(metrics_paths)} metrics files.")
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
