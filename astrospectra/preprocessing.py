from dataclasses import dataclass

import numpy as np

from .spectrum import Spectrum


@dataclass
class ProcessedSpectrum:
    """Represent a spectrum prepared for machine learning."""

    wavelength: np.ndarray
    flux: np.ndarray
    valid_mask: np.ndarray
    object_class: str | None
    redshift: float | None


def preprocess_spectrum(
    spectrum: Spectrum,
    wavelength_min: float = 3800.0,
    wavelength_max: float = 9000.0,
    number_of_points: int = 2048,
) -> ProcessedSpectrum:
    """Interpolate and normalize an SDSS spectrum."""

    if number_of_points < 2:
        raise ValueError(
            "number_of_points must be at least 2."
        )

    if wavelength_min >= wavelength_max:
        raise ValueError(
            "wavelength_min must be smaller than wavelength_max."
        )

    # Ensure the original measurements are ordered by wavelength.
    order = np.argsort(spectrum.wavelength)

    wavelength = spectrum.wavelength[order]
    flux = spectrum.flux[order]
    inverse_variance = spectrum.inverse_variance[order]

    # Every processed spectrum will use this same wavelength grid.
    common_grid = np.linspace(
        wavelength_min,
        wavelength_max,
        number_of_points,
    )

    interpolated_flux = np.interp(
        common_grid,
        wavelength,
        flux,
        left=np.nan,
        right=np.nan,
    )

    interpolated_inverse_variance = np.interp(
        common_grid,
        wavelength,
        inverse_variance,
        left=0.0,
        right=0.0,
    )

    valid_mask = (
        np.isfinite(interpolated_flux)
        & np.isfinite(interpolated_inverse_variance)
        & (interpolated_inverse_variance > 0)
    )

    if not np.any(valid_mask):
        raise ValueError(
            "The spectrum has no valid measurements "
            "inside the selected wavelength range."
        )

    valid_flux = interpolated_flux[valid_mask]

    median_flux = np.median(valid_flux)
    flux_scale = np.std(valid_flux)

    if flux_scale == 0:
        raise ValueError(
            "The spectrum has no measurable flux variation."
        )

    normalized_flux = np.zeros(
        number_of_points,
        dtype=float,
    )

    normalized_flux[valid_mask] = (
        interpolated_flux[valid_mask] - median_flux
    ) / flux_scale

    return ProcessedSpectrum(
        wavelength=common_grid,
        flux=normalized_flux,
        valid_mask=valid_mask,
        object_class=spectrum.object_class,
        redshift=spectrum.redshift,
    )