import csv
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = Path(
    "data/processed/medium_dataset.npz"
)

MODEL_FILE = Path(
    "models/isolation_forest_anomaly_detector.joblib"
)

OUTPUT_DIRECTORY = Path(
    "outputs/anomalies"
)

TOP_ANOMALIES_FILE = (
    OUTPUT_DIRECTORY / "top_anomalies.csv"
)

SCORE_HISTOGRAM_FILE = (
    OUTPUT_DIRECTORY / "anomaly_score_histogram.png"
)

RANDOM_STATE = 42
PCA_COMPONENTS = 20
CONTAMINATION = 0.10
TOP_N = 12

WAVELENGTH_MIN = 3800
WAVELENGTH_MAX = 9000


def build_anomaly_detector() -> Pipeline:
    """Create an anomaly detection pipeline."""

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=PCA_COMPONENTS,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "detector",
                IsolationForest(
                    n_estimators=500,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_anomaly_plot(
    wavelength: np.ndarray,
    flux: np.ndarray,
    output_file: Path,
    spectrum_id: str,
    object_class: str,
    redshift: float,
    anomaly_score: float,
    rank: int,
) -> None:
    """Save one spectrum plot."""

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
        linestyle="--",
        linewidth=0.8,
    )

    plt.title(
        f"Anomaly rank {rank}: {object_class}\n"
        f"Spectrum ID: {spectrum_id} | "
        f"Redshift: {redshift:.4f} | "
        f"Anomaly score: {anomaly_score:.4f}"
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


def save_score_histogram(
    anomaly_scores: np.ndarray,
) -> None:
    """Save a histogram of anomaly scores."""

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        anomaly_scores,
        bins=20,
    )

    plt.xlabel(
        "Anomaly score"
    )

    plt.ylabel(
        "Number of spectra"
    )

    plt.title(
        "Distribution of Anomaly Scores"
    )

    plt.tight_layout()

    plt.savefig(
        SCORE_HISTOGRAM_FILE,
        dpi=150,
    )

    plt.close()


def main() -> None:
    """Detect unusual spectra using Isolation Forest."""

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
    redshifts = dataset["redshifts"]
    spectrum_ids = dataset["spectrum_ids"]
    file_paths = dataset["file_paths"]

    print("Dataset")
    print("-------")
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Classes: {class_names}")

    detector = build_anomaly_detector()

    print("\nTraining anomaly detector...")
    print("----------------------------")

    detector.fit(
        features
    )

    # IsolationForest returns:
    #  1  = normal
    # -1  = anomaly
    anomaly_labels = detector.predict(
        features
    )

    # decision_function gives higher values for more normal spectra.
    # We multiply by -1 so higher score means more anomalous.
    normality_scores = detector.decision_function(
        features
    )

    anomaly_scores = -normality_scores

    anomaly_count = int(
        np.sum(
            anomaly_labels == -1
        )
    )

    print(f"Detected anomalies: {anomaly_count}")
    print(f"Contamination setting: {CONTAMINATION}")

    sorted_indices = np.argsort(
        anomaly_scores
    )[::-1]

    top_indices = sorted_indices[
        :TOP_N
    ]

    wavelength = np.linspace(
        WAVELENGTH_MIN,
        WAVELENGTH_MAX,
        features.shape[1],
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("\nTop anomalies")
    print("-------------")

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        class_index = int(
            targets[index]
        )

        object_class = str(
            class_names[class_index]
        )

        spectrum_id = str(
            spectrum_ids[index]
        )

        redshift = float(
            redshifts[index]
        )

        anomaly_score = float(
            anomaly_scores[index]
        )

        is_flagged_anomaly = (
            anomaly_labels[index] == -1
        )

        output_file = OUTPUT_DIRECTORY / (
            f"anomaly_{rank:02d}_"
            f"{object_class}_"
            f"{spectrum_id}.png"
        )

        save_anomaly_plot(
            wavelength=wavelength,
            flux=features[index],
            output_file=output_file,
            spectrum_id=spectrum_id,
            object_class=object_class,
            redshift=redshift,
            anomaly_score=anomaly_score,
            rank=rank,
        )

        rows.append(
            {
                "rank": rank,
                "spectrum_id": spectrum_id,
                "object_class": object_class,
                "redshift": redshift,
                "anomaly_score": anomaly_score,
                "flagged_anomaly": is_flagged_anomaly,
                "file_path": str(file_paths[index]),
                "plot_file": str(output_file),
            }
        )

        print(
            f"{rank:02d}. "
            f"{object_class} | "
            f"ID: {spectrum_id} | "
            f"score={anomaly_score:.4f} | "
            f"flagged={is_flagged_anomaly}"
        )

    with TOP_ANOMALIES_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "rank",
            "spectrum_id",
            "object_class",
            "redshift",
            "anomaly_score",
            "flagged_anomaly",
            "file_path",
            "plot_file",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    save_score_histogram(
        anomaly_scores
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": detector,
            "class_names": class_names,
            "pca_components": PCA_COMPONENTS,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
        },
        MODEL_FILE,
    )

    print(
        f"\nSaved anomaly detector to: "
        f"{MODEL_FILE}"
    )

    print(
        f"Saved top anomalies CSV to: "
        f"{TOP_ANOMALIES_FILE}"
    )

    print(
        f"Saved score histogram to: "
        f"{SCORE_HISTOGRAM_FILE}"
    )

    print(
        f"Saved anomaly plots in: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()