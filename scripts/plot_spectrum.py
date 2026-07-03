from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


INPUT_FILE = Path("data/raw/sample_spectrum.fits")
OUTPUT_FILE = Path("outputs/sample_spectrum.png")


def main() -> None:
    """Read and plot one downloaded SDSS spectrum."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Spectrum not found: {INPUT_FILE}. "
            "Run download_one_spectrum.py first."
        )

    with fits.open(INPUT_FILE) as hdul:
        spectrum_table = hdul[1].data

        print("Available columns:")
        print(spectrum_table.names)

        flux = np.asarray(
            spectrum_table["flux"],
            dtype=float,
        )

        log_wavelength = np.asarray(
            spectrum_table["loglam"],
            dtype=float,
        )

    wavelength = 10**log_wavelength

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(12, 5),
    )

    axis.plot(
        wavelength,
        flux,
        linewidth=0.8,
    )

    axis.set_title("SDSS Astronomical Spectrum")
    axis.set_xlabel("Wavelength (Å)")
    axis.set_ylabel(
        "Flux "
        "(10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)"
    )
    axis.grid(
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FILE,
        dpi=150,
    )

    print(f"Number of measurements: {len(flux)}")
    print(
        f"Wavelength range: "
        f"{wavelength.min():.1f}–"
        f"{wavelength.max():.1f} Å"
    )
    print(f"Saved plot to: {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()