from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path(
    "data/processed/small_dataset.npz"
)

MODEL_FILE = Path(
    "models/baseline_classifier.joblib"
)

CONFUSION_MATRIX_FILE = Path(
    "outputs/baseline_confusion_matrix.png"
)

RANDOM_STATE = 42


def main() -> None:
    """Train and evaluate a baseline spectrum classifier."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}. "
            "Run build_processed_dataset.py first."
        )

    with np.load(
        DATASET_FILE,
        allow_pickle=False,
    ) as data:
        features = data["X"]
        targets = data["y"]
        class_names = data["class_names"].astype(str)
        spectrum_ids = data["spectrum_ids"].astype(str)

    print("Loaded dataset")
    print("--------------")
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Classes: {class_names}")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
    ) = train_test_split(
        features,
        targets,
        spectrum_ids,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    print("\nDataset split")
    print("-------------")
    print(f"Training spectra: {len(X_train)}")
    print(f"Testing spectra: {len(X_test)}")

    number_of_components = min(
        10,
        X_train.shape[0] - 1,
        X_train.shape[1],
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=number_of_components,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\nModel evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.3f}")

    print("\nClassification report")
    print("---------------------")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Individual test predictions")
    print("---------------------------")

    for spectrum_id, correct, predicted, probability in zip(
        ids_test,
        y_test,
        predictions,
        probabilities,
    ):
        confidence = probability[predicted]

        print(
            f"ID: {spectrum_id} | "
            f"Actual: {class_names[correct]} | "
            f"Predicted: {class_names[predicted]} | "
            f"Confidence: {confidence:.3f}"
        )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "class_names": class_names.tolist(),
        },
        MODEL_FILE,
    )

    CONFUSION_MATRIX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        display_labels=class_names,
    )

    display.ax_.set_title(
        "Baseline Spectrum Classifier"
    )

    display.figure_.tight_layout()

    display.figure_.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=150,
    )

    plt.close(display.figure_)

    print(f"\nSaved model to: {MODEL_FILE}")
    print(
        "Saved confusion matrix to: "
        f"{CONFUSION_MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()