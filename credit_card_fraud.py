#!/usr/bin/env python
"""Credit card fraud detection with a leakage-free SMOTE pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
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


def build_model(strategy: str) -> Pipeline:
    """Build a leakage-free pipeline for the requested imbalance strategy."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("amount_scaler", StandardScaler(), [AMOUNT_COLUMN]),
        ],
        remainder="passthrough",
    )

    if strategy not in {"standard", "balanced", "smote"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    steps = [("preprocessor", preprocessor)]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced" if strategy == "balanced" else None,
            ),
        )
    )
    return Pipeline(steps=steps)


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict[str, float], object]:
    """Print classification metrics and return scores and the confusion matrix."""
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
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }
    print(f"Precision: {scores['precision']:.5f}")
    print(f"Recall:    {scores['recall']:.5f}")
    print(f"F1-score:  {scores['f1']:.5f}")
    print(f"ROC-AUC: {scores['roc_auc']:.5f}")
    print(f"PR-AUC:  {scores['pr_auc']:.5f}")
    print(f"Threshold: {threshold:.2f}")
    return scores, matrix


def cross_validate_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
) -> dict[str, float]:
    """Evaluate a model with stratified cross-validation on training data."""
    scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=splitter,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    return {
        metric: scores[f"test_{metric}"].mean()
        for metric in scoring
    }


def plot_results(dataframe: pd.DataFrame, matrix: object) -> None:
    """Display class distribution and confusion matrix plots."""
    import matplotlib.pyplot as plt

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
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    dataframe = load_data(args.data)
    print(f"Loaded {len(dataframe):,} transactions.")
    print(f"Fraud rate: {dataframe[TARGET_COLUMN].mean() * 100:.4f}%")

    x_train, x_test, y_train, y_test = prepare_data(dataframe)
    print(f"Training rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")

    strategies = {
        "standard": "Random Forest standard",
        "balanced": "Random Forest class_weight=balanced",
        "smote": "Random Forest avec SMOTE",
    }
    results: list[dict[str, float | str]] = []
    cv_results: list[dict[str, float | str]] = []
    matrices: dict[str, object] = {}

    for strategy, name in strategies.items():
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        model = build_model(strategy)
        validation_scores = cross_validate_model(
            model,
            x_train,
            y_train,
            args.cv_folds,
        )
        cv_results.append({"strategy": name, **validation_scores})
        print("Cross-validation means")
        for metric, score in validation_scores.items():
            print(f"{metric}: {score:.5f}")

        model.fit(x_train, y_train)
        scores, matrix = evaluate_model(
            model,
            x_test,
            y_test,
            args.threshold,
        )
        results.append({"strategy": name, **scores})
        matrices[strategy] = matrix

    cv_comparison = pd.DataFrame(cv_results).set_index("strategy")
    print("\nCross-validation comparison")
    print(
        cv_comparison[
            ["precision", "recall", "f1", "pr_auc", "roc_auc"]
        ].round(5)
    )

    comparison = pd.DataFrame(results).set_index("strategy")
    print("\nModel comparison")
    print(comparison[["precision", "recall", "f1", "pr_auc", "roc_auc"]].round(5))

    best_strategy = comparison["pr_auc"].idxmax()
    print(f"\nBest model according to PR-AUC: {best_strategy}")

    if args.plot:
        plot_results(dataframe, matrices["smote"])


if __name__ == "__main__":
    main()
