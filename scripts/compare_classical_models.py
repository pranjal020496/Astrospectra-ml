import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path(
    "data/processed/medium_dataset.npz"
)

OUTPUT_DIRECTORY = Path(
    "outputs"
)

COMPARISON_CSV_FILE = (
    OUTPUT_DIRECTORY / "classical_model_comparison.csv"
)

COMPARISON_PLOT_FILE = (
    OUTPUT_DIRECTORY / "classical_model_comparison.png"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


def build_models() -> dict[str, object]:
    """Create the classical models to compare."""

    baseline_logistic = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=10,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    tuned_logistic = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=50,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    return {
        "Baseline Logistic Regression": baseline_logistic,
        "Tuned Logistic Regression": tuned_logistic,
        "Random Forest": random_forest,
    }


def evaluate_model(
    model_name: str,
    model: object,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, object]:
    """Train one model and return its evaluation metrics."""

    print(f"\nTraining: {model_name}")
    print("-" * (10 + len(model_name)))

    model.fit(
        train_features,
        train_targets,
    )

    predictions = model.predict(
        test_features
    )

    accuracy = accuracy_score(
        test_targets,
        predictions,
    )

    report = classification_report(
        test_targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    mistakes = int(
        np.sum(
            predictions != test_targets
        )
    )

    correct = int(
        len(test_targets) - mistakes
    )

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Correct: {correct}/{len(test_targets)}")
    print(f"Mistakes: {mistakes}")
    print(f"Macro precision: {macro_precision:.3f}")
    print(f"Macro recall: {macro_recall:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_targets),
        "mistakes": mistakes,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def save_results_csv(
    results: list[dict[str, object]],
) -> None:
    """Save comparison results as a CSV file."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with COMPARISON_CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "model",
            "accuracy",
            "correct",
            "total",
            "mistakes",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


def save_accuracy_plot(
    results: list[dict[str, object]],
) -> None:
    """Save a simple accuracy comparison plot."""

    model_names = [
        str(row["model"])
        for row in results
    ]

    accuracies = [
        float(row["accuracy"])
        for row in results
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        model_names,
        accuracies,
    )

    plt.ylim(
        0,
        1,
    )

    plt.ylabel(
        "Test accuracy"
    )

    plt.title(
        "Classical Model Comparison"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    for index, accuracy in enumerate(
        accuracies
    ):
        plt.text(
            index,
            accuracy + 0.02,
            f"{accuracy:.3f}",
            ha="center",
        )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_PLOT_FILE,
        dpi=150,
    )

    plt.close()


def main() -> None:
    """Compare classical machine-learning models."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    dataset = np.load(
        DATASET_FILE
    )

    features = dataset["X"]
    targets = dataset["y"]
    class_names = dataset["class_names"]

    print("Dataset")
    print("-------")
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Classes: {class_names}")

    (
        train_features,
        test_features,
        train_targets,
        test_targets,
    ) = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    print("\nDataset split")
    print("-------------")
    print(f"Training spectra: {len(train_targets)}")
    print(f"Testing spectra: {len(test_targets)}")

    models = build_models()

    results = []

    for model_name, model in models.items():
        result = evaluate_model(
            model_name=model_name,
            model=model,
            train_features=train_features,
            test_features=test_features,
            train_targets=train_targets,
            test_targets=test_targets,
            class_names=class_names,
        )

        results.append(result)

    results = sorted(
        results,
        key=lambda row: float(row["accuracy"]),
        reverse=True,
    )

    print("\nFinal comparison")
    print("----------------")

    for row in results:
        print(
            f"{row['model']}: "
            f"accuracy={row['accuracy']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}, "
            f"mistakes={row['mistakes']}"
        )

    save_results_csv(
        results
    )

    save_accuracy_plot(
        results
    )

    print(
        f"\nSaved comparison CSV to: "
        f"{COMPARISON_CSV_FILE}"
    )

    print(
        f"Saved comparison plot to: "
        f"{COMPARISON_PLOT_FILE}"
    )


if __name__ == "__main__":
    main()