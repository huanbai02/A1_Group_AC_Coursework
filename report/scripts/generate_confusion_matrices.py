"""Generate report confusion matrix figures from saved prediction CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "report" / "figures"

LABEL_ORDER = [
    "business_finance",
    "computers_internet",
    "education_reference",
    "entertainment_music",
    "family_relationships",
    "health",
    "politics_government",
    "science_mathematics",
    "society_culture",
    "sports",
]

LABEL_DISPLAY = [
    "Bus.",
    "Comp.",
    "Edu.",
    "Enter.",
    "Family",
    "Health",
    "Pol.",
    "Sci.",
    "Soc.",
    "Sports",
]


def plot_normalized_confusion_matrix(
    prediction_path: Path,
    output_path: Path,
    title: str,
) -> None:
    """Plot a row-normalized confusion matrix from a prediction CSV."""
    predictions = pd.read_csv(prediction_path)
    required_columns = {"true_label", "predicted_label"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"{prediction_path} missing columns: {sorted(missing_columns)}")

    matrix = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=LABEL_ORDER,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=240)
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion by true label", rotation=90, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xticks(np.arange(len(LABEL_DISPLAY)))
    ax.set_yticks(np.arange(len(LABEL_DISPLAY)))
    ax.set_xticklabels(LABEL_DISPLAY, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(LABEL_DISPLAY, fontsize=10)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if value >= 0.20:
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.50 else "black",
                    fontsize=8,
                )

    ax.set_ylim(len(LABEL_DISPLAY) - 0.5, -0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    figures = [
        (
            PROJECT_ROOT / "data" / "results" / "predictions" / "dataset_1_bert_predictions.csv",
            FIGURE_DIR / "dataset_1_bert_confusion_matrix.png",
            "DistilBERT on Yahoo Answers Topics",
        ),
        (
            PROJECT_ROOT
            / "data"
            / "results"
            / "predictions"
            / "dataset_1_word2vec_predictions.csv",
            FIGURE_DIR / "dataset_1_word2vec_confusion_matrix.png",
            "Word2Vec-based classifier on Yahoo Answers Topics",
        ),
    ]
    for prediction_path, output_path, title in figures:
        plot_normalized_confusion_matrix(prediction_path, output_path, title)
        print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
