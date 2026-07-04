from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


DATASET_FILE = Path(
    "data/processed/medium_dataset.npz"
)

MODEL_FILE = Path(
    "models/random_forest_classifier.joblib"
)

CONFUSION_MATRIX_FILE = Path(
    "outputs/random_forest_confusion_matrix.png"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    """Train and evaluate a Random Forest classifier."""

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
    spectrum_ids = dataset["spectrum_ids"]

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
        train_spectrum_ids,
        test_spectrum_ids,
    ) = train_test_split(
        features,
        targets,
        spectrum_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    print("\nDataset split")
    print("-------------")
    print(f"Training spectra: {len(train_targets)}")
    print(f"Testing spectra: {len(test_targets)}")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(
        train_features,
        train_targets,
    )

    predictions = model.predict(
        test_features
    )

    probabilities = model.predict_proba(
        test_features
    )

    accuracy = accuracy_score(
        test_targets,
        predictions,
    )

    print("\nModel evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.3f}")

    print("\nClassification report")
    print("---------------------")
    print(
        classification_report(
            test_targets,
            predictions,
            target_names=class_names,
        )
    )

    print("Incorrect predictions")
    print("---------------------")

    incorrect_positions = np.flatnonzero(
        predictions != test_targets
    )

    if len(incorrect_positions) == 0:
        print("No incorrect predictions.")
    else:
        for position in incorrect_positions:
            actual_index = int(
                test_targets[position]
            )

            predicted_index = int(
                predictions[position]
            )

            actual_class = str(
                class_names[actual_index]
            )

            predicted_class = str(
                class_names[predicted_index]
            )

            confidence = float(
                probabilities[position].max()
            )

            spectrum_id = str(
                test_spectrum_ids[position]
            )

            print(
                f"ID: {spectrum_id} | "
                f"Actual: {actual_class} | "
                f"Predicted: {predicted_class} | "
                f"Confidence: {confidence:.3f}"
            )

    matrix = confusion_matrix(
        test_targets,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    CONFUSION_MATRIX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=150,
    )

    plt.close()

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "class_names": class_names,
            "test_accuracy": accuracy,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_estimators": 500,
        },
        MODEL_FILE,
    )

    print(f"\nSaved model to: {MODEL_FILE}")
    print(
        "Saved confusion matrix to: "
        f"{CONFUSION_MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()