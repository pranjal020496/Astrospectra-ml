from pathlib import Path

import numpy as np

from astrospectra.spectrum import load_sdss_spectrum


INPUT_FILE = Path("data/raw/sample_spectrum.fits")


def main() -> None:
    """Display information about one SDSS spectrum."""

    spectrum = load_sdss_spectrum(INPUT_FILE)

    uncertainty = 1 / np.sqrt(
        spectrum.inverse_variance
    )

    print("Spectrum information")
    print("--------------------")

    print(
        f"Object class: "
        f"{spectrum.object_class or 'Unknown'}"
    )

    if spectrum.redshift is None:
        print("Redshift: Unknown")
    else:
        print(
            f"Redshift: "
            f"{spectrum.redshift:.5f}"
        )

    print(
        f"Valid measurements: "
        f"{len(spectrum.flux)}"
    )

    print(
        f"Wavelength range: "
        f"{spectrum.wavelength.min():.1f}–"
        f"{spectrum.wavelength.max():.1f} Å"
    )

    print(
        f"Minimum flux: "
        f"{spectrum.flux.min():.4f}"
    )

    print(
        f"Maximum flux: "
        f"{spectrum.flux.max():.4f}"
    )

    print(
        f"Median flux: "
        f"{np.median(spectrum.flux):.4f}"
    )

    print(
        f"Median uncertainty: "
        f"{np.median(uncertainty):.4f}"
    )


if __name__ == "__main__":
    main()