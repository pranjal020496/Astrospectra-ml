from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from astrospectra.preprocessing import preprocess_spectrum
from astrospectra.spectrum import load_sdss_spectrum


INPUT_FILE = Path(
    "data/raw/sample_spectrum.fits"
)

OUTPUT_DATA = Path(
    "data/processed/sample_spectrum.npz"
)

OUTPUT_PLOT = Path(
    "outputs/preprocessed_spectrum.png"
)


def main() -> None:
    """Preprocess one SDSS spectrum."""

    spectrum = load_sdss_spectrum(INPUT_FILE)

    processed = preprocess_spectrum(spectrum)

    OUTPUT_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_DATA,
        wavelength=processed.wavelength,
        flux=processed.flux,
        valid_mask=processed.valid_mask,
        object_class=processed.object_class or "",
        redshift=(
            processed.redshift
            if processed.redshift is not None
            else np.nan
        ),
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
    )

    axes[0].plot(
        spectrum.wavelength,
        spectrum.flux,
        linewidth=0.8,
    )

    axes[0].set_title("Raw SDSS Spectrum")
    axes[0].set_xlabel("Wavelength (Å)")
    axes[0].set_ylabel("Flux")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        processed.wavelength,
        processed.flux,
        linewidth=0.8,
    )

    axes[1].set_title(
        "Normalized Spectrum on Common Grid"
    )

    axes[1].set_xlabel("Wavelength (Å)")
    axes[1].set_ylabel("Normalized flux")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PLOT,
        dpi=150,
    )

    plt.close(figure)

    print("Preprocessing complete")
    print("----------------------")
    print(
        f"Object class: "
        f"{processed.object_class or 'Unknown'}"
    )
    print(
        f"Original measurements: "
        f"{len(spectrum.flux)}"
    )
    print(
        f"Processed measurements: "
        f"{len(processed.flux)}"
    )
    print(
        f"Valid processed points: "
        f"{processed.valid_mask.sum()}"
    )
    print(
        f"Normalized mean: "
        f"{processed.flux[processed.valid_mask].mean():.4f}"
    )
    print(
        f"Normalized standard deviation: "
        f"{processed.flux[processed.valid_mask].std():.4f}"
    )
    print(f"Saved data to: {OUTPUT_DATA}")
    print(f"Saved plot to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()