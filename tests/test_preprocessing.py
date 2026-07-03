import numpy as np
import pytest

from astrospectra.preprocessing import preprocess_spectrum
from astrospectra.spectrum import Spectrum


def create_example_spectrum() -> Spectrum:
    """Create a small artificial spectrum for testing."""

    return Spectrum(
        wavelength=np.array(
            [3800.0, 5000.0, 7000.0, 9000.0]
        ),
        flux=np.array(
            [10.0, 20.0, 40.0, 30.0]
        ),
        inverse_variance=np.array(
            [1.0, 1.0, 1.0, 1.0]
        ),
        object_class="GALAXY",
        redshift=0.12,
    )


def test_preprocessing_creates_fixed_length_spectrum():
    spectrum = create_example_spectrum()

    processed = preprocess_spectrum(
        spectrum,
        wavelength_min=3800.0,
        wavelength_max=9000.0,
        number_of_points=16,
    )

    assert len(processed.wavelength) == 16
    assert len(processed.flux) == 16
    assert len(processed.valid_mask) == 16

    assert processed.valid_mask.all()

    assert processed.object_class == "GALAXY"
    assert processed.redshift == 0.12


def test_preprocessed_flux_has_unit_standard_deviation():
    spectrum = create_example_spectrum()

    processed = preprocess_spectrum(
        spectrum,
        number_of_points=16,
    )

    valid_flux = processed.flux[
        processed.valid_mask
    ]

    assert np.isclose(
        np.std(valid_flux),
        1.0,
    )


def test_preprocessing_rejects_invalid_point_count():
    spectrum = create_example_spectrum()

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        preprocess_spectrum(
            spectrum,
            number_of_points=1,
        )


def test_preprocessing_rejects_non_overlapping_spectrum():
    spectrum = Spectrum(
        wavelength=np.array(
            [1000.0, 1500.0, 2000.0]
        ),
        flux=np.array(
            [1.0, 2.0, 3.0]
        ),
        inverse_variance=np.array(
            [1.0, 1.0, 1.0]
        ),
        object_class="STAR",
        redshift=0.0,
    )

    with pytest.raises(
        ValueError,
        match="no valid measurements",
    ):
        preprocess_spectrum(
            spectrum,
            wavelength_min=3800.0,
            wavelength_max=9000.0,
        )