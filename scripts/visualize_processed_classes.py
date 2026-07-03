from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_FILE = Path(
    "data/processed/small_dataset.npz"
)

OUTPUT_FILE = Path(
    "outputs/class_examples.png"
)

WAVELENGTH_MIN = 3800.0
WAVELENGTH_MAX = 9000.0


def main() -> None:
    """Plot one processed spectrum from each object class."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}. "
            "Run build_processed_dataset.py first."
        )

    with np.load(DATASET_FILE) as data:
        features = data["X"]
        targets = data["y"]
        class_names = data["class_names"]

    wavelength = np.linspace(
        WAVELENGTH_MIN,
        WAVELENGTH_MAX,
        features.shape[1],
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        len(class_names),
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for class_index, class_name in enumerate(class_names):
        matching_indices = np.flatnonzero(
            targets == class_index
        )

        if len(matching_indices) == 0:
            raise ValueError(
                f"No spectrum found for class: {class_name}"
            )

        spectrum_index = matching_indices[0]
        spectrum_flux = features[spectrum_index]

        axis = axes[class_index]

        axis.plot(
            wavelength,
            spectrum_flux,
            linewidth=0.8,
        )

        axis.set_title(
            f"Example {class_name} Spectrum"
        )

        axis.set_ylabel("Normalized flux")
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Wavelength (Å)")

    figure.suptitle(
        "Processed SDSS Spectra by Object Class"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FILE,
        dpi=150,
    )

    plt.close(figure)

    print("Class comparison created")
    print("------------------------")
    print(f"Dataset shape: {features.shape}")
    print(f"Classes: {class_names}")
    print(f"Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()