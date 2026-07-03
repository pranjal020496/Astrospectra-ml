from pathlib import Path

from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS


DATA_RELEASE = 17
OUTPUT_FILE = Path("data/raw/sample_spectrum.fits")


def main() -> None:
    """Download one real spectrum from SDSS."""

    position = SkyCoord(
        "0h8m05.63s +14d50m23.3s",
        frame="icrs",
    )

    print("Searching SDSS for a spectrum...")

    matches = SDSS.query_region(
        position,
        radius="5 arcsec",
        spectro=True,
        data_release=DATA_RELEASE,
    )

    if matches is None or len(matches) == 0:
        raise RuntimeError("No SDSS spectrum was found.")

    print(f"Found {len(matches)} spectroscopic match(es).")

    spectra = SDSS.get_spectra(
        matches=matches[:1],
        data_release=DATA_RELEASE,
    )

    if spectra is None or len(spectra) == 0:
        raise RuntimeError("The spectrum could not be downloaded.")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spectrum = spectra[0]

    try:
        spectrum.writeto(
            OUTPUT_FILE,
            overwrite=True,
        )
    finally:
        spectrum.close()

    print(f"Saved spectrum to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()