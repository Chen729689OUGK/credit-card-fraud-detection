#!/usr/bin/env python
"""Credit card fraud detection with a leakage-free SMOTE pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Class"
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a credit card fraud classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("creditcard.csv"),
        help="Path to the Kaggle creditcard.csv file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to classify a transaction as fraud.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display the class distribution and confusion matrix plots.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the transaction dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Download creditcard.csv from Kaggle and place it in the project root."
        )

    dataframe = pd.read_csv(path)
    required_columns = {TARGET_COLUMN, AMOUNT_COLUMN, TIME_COLUMN}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"Column {TARGET_COLUMN!r} contains missing values.")

    return dataframe


def prepare_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare features and make a stratified train/test split."""
    features = dataframe.drop(columns=[TARGET_COLUMN, TIME_COLUMN])
    target = dataframe[TARGET_COLUMN].astype("int64")

    return train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_model() -> Pipeline:
    """Build a pipeline that fits scaling and SMOTE only on training data."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("amount_scaler", StandardScaler(), [AMOUNT_COLUMN]),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Print classification metrics and return the main scores."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype("int64")

    print("\nClassification report")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix")
    print(matrix)

    scores = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }
    print(f"ROC-AUC: {scores['roc_auc']:.5f}")
    print(f"PR-AUC:  {scores['pr_auc']:.5f}")
    print(f"Threshold: {threshold:.2f}")
    return scores


def plot_results(dataframe: pd.DataFrame, matrix: object) -> None:
    """Display class distribution and confusion matrix plots."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    dataframe[TARGET_COLUMN].value_counts().sort_index().plot(
        kind="bar",
        ax=axes[0],
        color=["steelblue", "darkred"],
    )
    axes[0].set_title("Class distribution")
    axes[0].set_xlabel("Class (0 = legitimate, 1 = fraud)")
    axes[0].set_ylabel("Transactions")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1].set_title("Confusion matrix")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].set_xticks([0, 1], ["Legitimate", "Fraud"])
    axes[1].set_yticks([0, 1], ["Legitimate", "Fraud"])

    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                str(matrix[row][column]),
                ha="center",
                va="center",
                color="white" if matrix[row][column] > matrix.max() / 2 else "black",
            )

    figure.tight_layout()
    plt.show()


def main() -> None:
    """Load data, train the model, and report evaluation metrics."""
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be strictly between 0 and 1.")

    dataframe = load_data(args.data)
    print(f"Loaded {len(dataframe):,} transactions.")
    print(f"Fraud rate: {dataframe[TARGET_COLUMN].mean() * 100:.4f}%")

    x_train, x_test, y_train, y_test = prepare_data(dataframe)
    print(f"Training rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")

    model = build_model()
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= args.threshold).astype("int64")
    scores = evaluate_model(model, x_test, y_test, args.threshold)

    if args.plot:
        plot_results(dataframe, confusion_matrix(y_test, predictions))

    print(
        f"\nEvaluation complete: PR-AUC={scores['pr_auc']:.5f}, "
        f"ROC-AUC={scores['roc_auc']:.5f}"
    )


if __name__ == "__main__":
    main()
