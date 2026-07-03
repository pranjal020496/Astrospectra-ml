import csv
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


DATASET_FILE = Path(
    "data/processed/medium_dataset.npz"
)

MODEL_FILE = Path(
    "models/tuned_medium_baseline.joblib"
)

OUTPUT_DIRECTORY = Path(
    "outputs/tuned_misclassifications"
)

SUMMARY_FILE = (
    OUTPUT_DIRECTORY / "mistakes.csv"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42

WAVELENGTH_MIN = 3800
WAVELENGTH_MAX = 9000


def main() -> None:
    """Plot spectra incorrectly classified by the tuned model."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    dataset = np.load(
        DATASET_FILE
    )

    features = dataset["X"]
    targets = dataset["y"]
    class_names = dataset["class_names"]
    spectrum_ids = dataset["spectrum_ids"]

    all_indices = np.arange(
        len(targets)
    )

    _, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    test_features = features[
        test_indices
    ]

    test_targets = targets[
        test_indices
    ]

    test_spectrum_ids = spectrum_ids[
        test_indices
    ]

    # Load the saved model information.
    saved_object = joblib.load(
        MODEL_FILE
    )

    # The training script saved a dictionary containing the model.
    if isinstance(saved_object, dict):
        possible_model_keys = (
            "model",
            "pipeline",
            "estimator",
            "best_estimator",
        )

        model = None

        for key in possible_model_keys:
            candidate = saved_object.get(
                key
            )

            if hasattr(candidate, "predict"):
                model = candidate

                print(
                    "Loaded model from dictionary "
                    f"key: {key}"
                )

                break

        # Fallback: search every dictionary value.
        if model is None:
            for key, candidate in saved_object.items():
                if hasattr(candidate, "predict"):
                    model = candidate

                    print(
                        "Loaded model from dictionary "
                        f"key: {key}"
                    )

                    break

        if model is None:
            raise TypeError(
                "No trained model was found in the "
                "saved dictionary. Available keys: "
                f"{list(saved_object.keys())}"
            )

    else:
        model = saved_object

    # This must be outside the if/else block because both branches
    # produce a model.
    predictions = model.predict(
        test_features
    )

    probabilities = model.predict_proba(
        test_features
    )

    incorrect_positions = np.flatnonzero(
        predictions != test_targets
    )

    wavelength = np.linspace(
        WAVELENGTH_MIN,
        WAVELENGTH_MAX,
        features.shape[1],
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    print("\nTuned-model mistakes")
    print("--------------------")
    print(
        f"Incorrect predictions: "
        f"{len(incorrect_positions)}"
    )

    for mistake_number, position in enumerate(
        incorrect_positions,
        start=1,
    ):
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

        flux = test_features[position]

        output_file = OUTPUT_DIRECTORY / (
            f"mistake_{mistake_number}_"
            f"{actual_class}_as_"
            f"{predicted_class}.png"
        )

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            wavelength,
            flux,
            linewidth=0.8,
        )

        plt.axhline(
            0,
            linewidth=0.8,
            linestyle="--",
        )

        plt.title(
            f"Spectrum {spectrum_id}\n"
            f"Actual: {actual_class} | "
            f"Predicted: {predicted_class} | "
            f"Confidence: {confidence:.3f}"
        )

        plt.xlabel(
            "Observed wavelength (Å)"
        )

        plt.ylabel(
            "Standardized flux"
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=150,
        )

        plt.close()

        summary_rows.append(
            {
                "spectrum_id": spectrum_id,
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "plot_file": str(output_file),
            }
        )

        print(
            f"{mistake_number}. "
            f"{spectrum_id}: "
            f"{actual_class} → "
            f"{predicted_class} "
            f"({confidence:.3f})"
        )

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "spectrum_id",
                "actual_class",
                "predicted_class",
                "confidence",
                "plot_file",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print(
        f"\nPlots saved in: "
        f"{OUTPUT_DIRECTORY}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()