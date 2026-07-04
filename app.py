from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


DATASET_FILE = Path(
    "data/processed/medium_dataset.npz"
)

MODEL_COMPARISON_CSV = Path(
    "outputs/classical_model_comparison.csv"
)

MODEL_COMPARISON_PLOT = Path(
    "outputs/classical_model_comparison.png"
)

ANOMALIES_CSV = Path(
    "outputs/anomalies/top_anomalies.csv"
)

TUNED_MODEL_FILE = Path(
    "models/tuned_medium_baseline.joblib"
)

RANDOM_FOREST_MODEL_FILE = Path(
    "models/random_forest_classifier.joblib"
)

ANOMALY_MODEL_FILE = Path(
    "models/isolation_forest_anomaly_detector.joblib"
)

WAVELENGTH_MIN = 3800
WAVELENGTH_MAX = 9000


st.set_page_config(
    page_title="AstroSpectra ML Workbench",
    page_icon="🔭",
    layout="wide",
)


@st.cache_data
def load_dataset() -> dict[str, np.ndarray]:
    """Load the processed spectrum dataset."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    dataset = np.load(
        DATASET_FILE
    )

    return {
        "X": dataset["X"],
        "masks": dataset["masks"],
        "y": dataset["y"],
        "class_names": dataset["class_names"],
        "redshifts": dataset["redshifts"],
        "spectrum_ids": dataset["spectrum_ids"],
        "file_paths": dataset["file_paths"],
    }


@st.cache_resource
def load_joblib_model(
    model_file: str,
):
    """Load a saved scikit-learn model or model dictionary."""

    path = Path(
        model_file
    )

    if not path.exists():
        return None

    saved_object = joblib.load(
        path
    )

    if isinstance(saved_object, dict):
        for key in (
            "model",
            "pipeline",
            "estimator",
            "best_estimator",
        ):
            candidate = saved_object.get(
                key
            )

            if hasattr(candidate, "predict"):
                return candidate

        for candidate in saved_object.values():
            if hasattr(candidate, "predict"):
                return candidate

        return None

    return saved_object


def make_wavelength_grid(
    number_of_points: int,
) -> np.ndarray:
    """Create the wavelength grid used for plotting."""

    return np.linspace(
        WAVELENGTH_MIN,
        WAVELENGTH_MAX,
        number_of_points,
    )


def plot_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    title: str,
):
    """Create a spectrum plot."""

    figure, axis = plt.subplots(
        figsize=(12, 4)
    )

    axis.plot(
        wavelength,
        flux,
        linewidth=0.8,
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Observed wavelength (Å)"
    )

    axis.set_ylabel(
        "Standardized flux"
    )

    figure.tight_layout()

    return figure


def predict_with_model(
    model,
    features: np.ndarray,
    class_names: np.ndarray,
) -> tuple[str, float]:
    """Predict a class and confidence for one spectrum."""

    reshaped_features = features.reshape(
        1,
        -1,
    )

    prediction = model.predict(
        reshaped_features
    )[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            reshaped_features
        )[0]

        confidence = float(
            np.max(probabilities)
        )

    else:
        confidence = float("nan")

    predicted_class = str(
        class_names[int(prediction)]
    )

    return predicted_class, confidence


def score_anomaly(
    anomaly_model,
    features: np.ndarray,
) -> tuple[float, bool]:
    """Calculate anomaly score for one spectrum."""

    reshaped_features = features.reshape(
        1,
        -1,
    )

    normality_score = anomaly_model.decision_function(
        reshaped_features
    )[0]

    anomaly_score = float(
        -normality_score
    )

    anomaly_label = anomaly_model.predict(
        reshaped_features
    )[0]

    is_anomaly = bool(
        anomaly_label == -1
    )

    return anomaly_score, is_anomaly


def show_overview(
    dataset: dict[str, np.ndarray],
) -> None:
    """Show project and dataset overview."""

    features = dataset["X"]
    targets = dataset["y"]
    class_names = dataset["class_names"]

    st.title(
        "🔭 AstroSpectra ML Workbench"
    )

    st.write(
        "An interactive machine-learning workbench for "
        "SDSS astronomical spectra."
    )

    st.subheader(
        "Dataset Summary"
    )

    column_1, column_2, column_3, column_4 = st.columns(
        4
    )

    column_1.metric(
        "Spectra",
        len(features),
    )

    column_2.metric(
        "Flux points per spectrum",
        features.shape[1],
    )

    column_3.metric(
        "Classes",
        len(class_names),
    )

    column_4.metric(
        "Wavelength range",
        f"{WAVELENGTH_MIN}–{WAVELENGTH_MAX} Å",
    )

    class_rows = []

    for index, class_name in enumerate(
        class_names
    ):
        count = int(
            np.sum(
                targets == index
            )
        )

        class_rows.append(
            {
                "Class": str(class_name),
                "Count": count,
            }
        )

    st.dataframe(
        pd.DataFrame(class_rows),
        use_container_width=True,
    )

    st.subheader(
        "Current Best Classical Models"
    )

    st.write(
        "The strongest classical models so far are tuned logistic "
        "regression and Random Forest, both with 86.7% test accuracy "
        "on the medium dataset."
    )

    st.info(
        "The CNN was successfully trained, but underperformed on the "
        "small 150-spectrum dataset. This is expected because deep "
        "learning usually needs more data."
    )


def show_spectrum_explorer(
    dataset: dict[str, np.ndarray],
) -> None:
    """Show an interactive spectrum viewer and prediction panel."""

    features = dataset["X"]
    targets = dataset["y"]
    class_names = dataset["class_names"]
    redshifts = dataset["redshifts"]
    spectrum_ids = dataset["spectrum_ids"]
    file_paths = dataset["file_paths"]

    st.title(
        "Spectrum Explorer"
    )

    st.write(
        "Select one spectrum and compare model predictions, "
        "confidence scores, and anomaly status."
    )

    selected_class = st.selectbox(
        "Choose an object class",
        options=[
            "ALL",
            *[
                str(class_name)
                for class_name in class_names
            ],
        ],
    )

    available_indices = np.arange(
        len(targets)
    )

    if selected_class != "ALL":
        class_index = int(
            np.where(
                class_names == selected_class
            )[0][0]
        )

        available_indices = available_indices[
            targets == class_index
        ]

    spectrum_options = []

    for index in available_indices:
        class_name = str(
            class_names[int(targets[index])]
        )

        spectrum_options.append(
            f"{index} | {class_name} | ID {spectrum_ids[index]}"
        )

    selected_option = st.selectbox(
        "Choose a spectrum",
        options=spectrum_options,
    )

    selected_index = int(
        selected_option.split("|")[0].strip()
    )

    selected_features = features[
        selected_index
    ]

    actual_class = str(
        class_names[int(targets[selected_index])]
    )

    redshift = float(
        redshifts[selected_index]
    )

    spectrum_id = str(
        spectrum_ids[selected_index]
    )

    st.subheader(
        f"Spectrum ID: {spectrum_id}"
    )

    column_1, column_2, column_3 = st.columns(
        3
    )

    column_1.metric(
        "Actual class",
        actual_class,
    )

    column_2.metric(
        "Redshift",
        f"{redshift:.4f}",
    )

    column_3.metric(
        "Dataset index",
        selected_index,
    )

    wavelength = make_wavelength_grid(
        features.shape[1]
    )

    figure = plot_spectrum(
        wavelength=wavelength,
        flux=selected_features,
        title=f"{actual_class} spectrum",
    )

    st.pyplot(
        figure
    )

    plt.close(
        figure
    )

    st.caption(
        f"Original file: {file_paths[selected_index]}"
    )

    st.subheader(
        "Model Predictions"
    )

    tuned_model = load_joblib_model(
        str(TUNED_MODEL_FILE)
    )

    random_forest_model = load_joblib_model(
        str(RANDOM_FOREST_MODEL_FILE)
    )

    prediction_rows = []

    if tuned_model is not None:
        predicted_class, confidence = predict_with_model(
            tuned_model,
            selected_features,
            class_names,
        )

        prediction_rows.append(
            {
                "Model": "Tuned Logistic Regression",
                "Prediction": predicted_class,
                "Confidence": round(confidence, 3),
                "Matches actual": predicted_class == actual_class,
            }
        )

    if random_forest_model is not None:
        predicted_class, confidence = predict_with_model(
            random_forest_model,
            selected_features,
            class_names,
        )

        prediction_rows.append(
            {
                "Model": "Random Forest",
                "Prediction": predicted_class,
                "Confidence": round(confidence, 3),
                "Matches actual": predicted_class == actual_class,
            }
        )

    if prediction_rows:
        st.dataframe(
            pd.DataFrame(prediction_rows),
            use_container_width=True,
        )
    else:
        st.warning(
            "No saved classical model files were found in models/. "
            "Run the training scripts first."
        )

    st.subheader(
        "Anomaly Check"
    )

    anomaly_model = load_joblib_model(
        str(ANOMALY_MODEL_FILE)
    )

    if anomaly_model is None:
        st.warning(
            "No anomaly detector was found. Run "
            "`python scripts/detect_anomalies.py` first."
        )

    else:
        anomaly_score, is_anomaly = score_anomaly(
            anomaly_model,
            selected_features,
        )

        anomaly_column_1, anomaly_column_2 = st.columns(
            2
        )

        anomaly_column_1.metric(
            "Anomaly score",
            f"{anomaly_score:.4f}",
        )

        anomaly_column_2.metric(
            "Anomaly status",
            "Flagged" if is_anomaly else "Normal",
        )

        if is_anomaly:
            st.warning(
                "This spectrum was flagged as unusual by the "
                "Isolation Forest anomaly detector."
            )
        else:
            st.success(
                "This spectrum was not flagged as anomalous."
            )

    st.subheader(
        "How to read this page"
    )

    st.write(
        "A high model confidence means the classifier strongly prefers "
        "one class. A high anomaly score means the spectrum looks unusual "
        "compared with the rest of the dataset. A spectrum can be correctly "
        "classified and still be anomalous, or incorrectly classified because "
        "it is unusual."
    )


def show_model_comparison() -> None:
    """Show model comparison results."""

    st.title(
        "Model Comparison"
    )

    fallback_results = pd.DataFrame(
        [
            {
                "Model": "Baseline Logistic Regression",
                "Test Accuracy": 0.833,
                "Correct / 30": "25 / 30",
                "Mistakes": 5,
                "Macro F1-score": 0.836,
            },
            {
                "Model": "Tuned Logistic Regression",
                "Test Accuracy": 0.867,
                "Correct / 30": "26 / 30",
                "Mistakes": 4,
                "Macro F1-score": 0.864,
            },
            {
                "Model": "Random Forest",
                "Test Accuracy": 0.867,
                "Correct / 30": "26 / 30",
                "Mistakes": 4,
                "Macro F1-score": 0.869,
            },
            {
                "Model": "1D CNN",
                "Test Accuracy": 0.733,
                "Correct / 30": "22 / 30",
                "Mistakes": 8,
                "Macro F1-score": 0.730,
            },
        ]
    )

    if MODEL_COMPARISON_CSV.exists():
        comparison = pd.read_csv(
            MODEL_COMPARISON_CSV
        )

        st.write(
            "Loaded model comparison from generated CSV file."
        )

        st.dataframe(
            comparison,
            use_container_width=True,
        )

    else:
        st.write(
            "Using recorded comparison results from the README."
        )

        st.dataframe(
            fallback_results,
            use_container_width=True,
        )

    if MODEL_COMPARISON_PLOT.exists():
        st.image(
            str(MODEL_COMPARISON_PLOT),
            caption="Classical model comparison",
        )

    st.info(
        "The best classical models currently outperform the CNN. "
        "This is likely because the dataset is still small for "
        "deep learning."
    )


def show_anomalies() -> None:
    """Show anomaly detection results."""

    st.title(
        "Anomaly Detection"
    )

    st.write(
        "Anomaly detection finds spectra that look unusual compared "
        "with the rest of the dataset."
    )

    if not ANOMALIES_CSV.exists():
        st.warning(
            "Anomaly results not found. Run "
            "`python scripts/detect_anomalies.py` first."
        )

        return

    anomalies = pd.read_csv(
        ANOMALIES_CSV
    )

    st.subheader(
        "Top Anomalies"
    )

    st.dataframe(
        anomalies,
        use_container_width=True,
    )

    selected_rank = st.selectbox(
        "Choose anomaly rank",
        options=anomalies["rank"].tolist(),
    )

    selected_row = anomalies[
        anomalies["rank"] == selected_rank
    ].iloc[0]

    st.write(
        f"Class: **{selected_row['object_class']}**"
    )

    st.write(
        f"Spectrum ID: `{selected_row['spectrum_id']}`"
    )

    st.write(
        f"Anomaly score: `{selected_row['anomaly_score']:.4f}`"
    )

    plot_file = Path(
        selected_row["plot_file"]
    )

    if plot_file.exists():
        st.image(
            str(plot_file),
            caption=f"Anomaly rank {selected_rank}",
        )
    else:
        st.warning(
            f"Plot file not found: {plot_file}"
        )


def main() -> None:
    """Run the Streamlit application."""

    st.sidebar.title(
        "AstroSpectra ML"
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Spectrum Explorer",
            "Model Comparison",
            "Anomaly Detection",
        ],
    )

    try:
        dataset = load_dataset()
    except FileNotFoundError as error:
        st.error(
            str(error)
        )

        st.stop()

    if page == "Overview":
        show_overview(
            dataset
        )

    elif page == "Spectrum Explorer":
        show_spectrum_explorer(
            dataset
        )

    elif page == "Model Comparison":
        show_model_comparison()

    elif page == "Anomaly Detection":
        show_anomalies()


if __name__ == "__main__":
    main()