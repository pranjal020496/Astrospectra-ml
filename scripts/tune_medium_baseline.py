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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path(
    "data/processed/medium_dataset.npz"
)

MODEL_FILE = Path(
    "models/tuned_medium_baseline.joblib"
)

CONFUSION_MATRIX_FILE = Path(
    "outputs/tuned_medium_confusion_matrix.png"
)

RANDOM_STATE = 42


def main() -> None:
    """Tune and evaluate the medium-dataset baseline model."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}. "
            "Run build_medium_dataset.py first."
        )

    with np.load(
        DATASET_FILE,
        allow_pickle=False,
    ) as data:
        features = data["X"]
        targets = data["y"]
        class_names = data["class_names"].astype(str)
        spectrum_ids = data["spectrum_ids"].astype(str)

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
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    print("Dataset")
    print("-------")
    print(f"All spectra: {len(features)}")
    print(f"Training spectra: {len(X_train)}")
    print(f"Testing spectra: {len(X_test)}")

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    parameter_grid = {
        "pca__n_components": [
            5,
            10,
            20,
            30,
            50,
        ],
        "classifier__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
    }

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=True,
        verbose=1,
    )

    print("\nSearching for the best configuration...")

    search.fit(
        X_train,
        y_train,
    )

    print("\nBest cross-validation result")
    print("----------------------------")
    print(f"Best parameters: {search.best_params_}")
    print(
        f"Mean validation accuracy: "
        f"{search.best_score_:.3f}"
    )

    print("\nTop configurations")
    print("------------------")

    results = search.cv_results_
    ranked_indices = np.argsort(
        results["rank_test_score"]
    )

    for result_index in ranked_indices[:5]:
        parameters = results["params"][result_index]
        mean_score = results[
            "mean_test_score"
        ][result_index]
        standard_deviation = results[
            "std_test_score"
        ][result_index]
        training_score = results[
            "mean_train_score"
        ][result_index]

        print(
            f"PCA={parameters['pca__n_components']:>2} | "
            f"C={parameters['classifier__C']:<5} | "
            f"train={training_score:.3f} | "
            f"validation={mean_score:.3f} "
            f"± {standard_deviation:.3f}"
        )

    best_model = search.best_estimator_

    predictions = best_model.predict(
        X_test
    )

    probabilities = best_model.predict_proba(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\nFinal untouched-test evaluation")
    print("-------------------------------")
    print(f"Test accuracy: {test_accuracy:.3f}")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Incorrect predictions")
    print("---------------------")

    mistakes = 0

    for (
        spectrum_id,
        correct,
        predicted,
        probability,
    ) in zip(
        ids_test,
        y_test,
        predictions,
        probabilities,
    ):
        if correct == predicted:
            continue

        mistakes += 1
        confidence = probability[predicted]

        print(
            f"ID: {spectrum_id} | "
            f"Actual: {class_names[correct]} | "
            f"Predicted: {class_names[predicted]} | "
            f"Confidence: {confidence:.3f}"
        )

    if mistakes == 0:
        print("No incorrect predictions.")

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": best_model,
            "class_names": class_names.tolist(),
            "best_parameters": search.best_params_,
            "cross_validation_accuracy": search.best_score_,
            "test_accuracy": test_accuracy,
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
        "Tuned Medium Dataset Classifier"
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