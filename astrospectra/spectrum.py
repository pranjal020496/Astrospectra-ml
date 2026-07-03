from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass
class Spectrum:
    """Represent one processed SDSS spectrum."""

    wavelength: np.ndarray
    flux: np.ndarray
    inverse_variance: np.ndarray
    object_class: str | None
    redshift: float | None


def load_sdss_spectrum(file_path: str | Path) -> Spectrum:
    """Load wavelength, flux, uncertainty, and metadata from an SDSS FITS file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {path}")

    with fits.open(path, memmap=False) as hdul:
        spectrum_table = hdul[1].data

        required_columns = {
            "flux",
            "loglam",
            "ivar",
        }

        available_columns = set(spectrum_table.names)

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                f"Missing required FITS columns: {sorted(missing_columns)}"
            )

        flux = np.asarray(
            spectrum_table["flux"],
            dtype=float,
        )

        log_wavelength = np.asarray(
            spectrum_table["loglam"],
            dtype=float,
        )

        inverse_variance = np.asarray(
            spectrum_table["ivar"],
            dtype=float,
        )

        wavelength = 10**log_wavelength

        object_class = None
        redshift = None

        if (
            len(hdul) > 2
            and hdul[2].data is not None
            and len(hdul[2].data) > 0
        ):
            metadata = hdul[2].data

            metadata_columns = {
                name.lower(): name
                for name in metadata.names
            }

            if "class" in metadata_columns:
                class_column = metadata_columns["class"]
                class_value = metadata[class_column][0]

                if isinstance(class_value, bytes):
                    object_class = class_value.decode().strip()
                else:
                    object_class = str(class_value).strip()

            if "z" in metadata_columns:
                redshift_column = metadata_columns["z"]
                redshift = float(
                    metadata[redshift_column][0]
                )

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & np.isfinite(inverse_variance)
        & (inverse_variance > 0)
    )

    wavelength = wavelength[valid]
    flux = flux[valid]
    inverse_variance = inverse_variance[valid]

    if len(wavelength) == 0:
        raise ValueError(
            "The spectrum contains no valid measurements."
        )

    return Spectrum(
        wavelength=wavelength,
        flux=flux,
        inverse_variance=inverse_variance,
        object_class=object_class,
        redshift=redshift,
    )