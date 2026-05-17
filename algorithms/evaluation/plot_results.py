"""Plot model comparison figures from an existing metrics summary CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_SUMMARY_COLUMNS = {"dataset", "model", "f1_macro"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot F1-score comparison from summary metrics."
    )
    parser.add_argument("--summary_path", required=True, help="Path to all_metrics_summary.csv.")
    parser.add_argument("--output_dir", required=True, help="Output directory for figures.")
    return parser.parse_args()


def validate_summary_dataframe(summary_dataframe: pd.DataFrame, summary_path: Path) -> bool:
    """
    Validate summary CSV content before plotting.

    Returns:
        True when plotting can proceed, False when there is no data to plot.
    """
    missing_columns = REQUIRED_SUMMARY_COLUMNS - set(summary_dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Summary CSV is missing required columns: {sorted(missing_columns)}"
        )
    if summary_dataframe.empty:
        print(f"Summary CSV has no rows: {summary_path}")
        return False
    return True


def plot_macro_f1_comparison(summary_dataframe: pd.DataFrame, output_dir: Path) -> Path:
    """Generate and save a macro F1 comparison bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pivot_dataframe = summary_dataframe.pivot_table(
        index="model",
        columns="dataset",
        values="f1_macro",
        aggfunc="mean",
    )

    axes = pivot_dataframe.plot(kind="bar", figsize=(9, 5))
    axes.set_title("Macro F1-score Comparison")
    axes.set_ylabel("Macro F1-score")
    axes.set_xlabel("Model")
    axes.set_ylim(0, 1)
    axes.legend(title="Dataset")
    plt.tight_layout()

    output_path = output_dir / "f1_macro_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    """Generate a macro F1 comparison figure from real summary data."""
    args = parse_args()
    summary_path = Path(args.summary_path)
    if not summary_path.exists():
        print(f"Summary CSV does not exist: {summary_path}")
        return

    summary_dataframe = pd.read_csv(summary_path)
    if not validate_summary_dataframe(summary_dataframe, summary_path):
        return

    output_path = plot_macro_f1_comparison(summary_dataframe, Path(args.output_dir))
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
