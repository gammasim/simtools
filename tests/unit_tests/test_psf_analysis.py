#!/usr/bin/python3

import logging

import numpy as np
import pytest
from astropy import units as u

from simtools.psf_analysis import PSFImage

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def psf_image():
    image = PSFImage(focal_length=2800.0)
    rng = np.random.default_rng(seed=42)
    image.photon_pos_x = rng.normal(loc=0.0, scale=1.0, size=1000)
    image.photon_pos_y = rng.normal(loc=0.0, scale=1.0, size=1000)
    image.centroid_x = np.mean(image.photon_pos_x)
    image.centroid_y = np.mean(image.photon_pos_y)
    image._number_of_detected_photons = len(image.photon_pos_x)
    image.photon_r = np.sort(
        np.sqrt(
            (image.photon_pos_x - image.centroid_x) ** 2
            + (image.photon_pos_y - image.centroid_y) ** 2
        )
    )
    return image


def test_reading_simtel_file(args_dict, io_handler):
    test_file = io_handler.get_input_data_file(
        file_name="photons-North-LSTN-01-d10.0km-za20.0deg-off0.000deg_validate_optics.lis.gz",
        test=True,
    )
    image = PSFImage(focal_length=2800.0)
    image.read_photon_list_from_simtel_file(test_file)
    logger.info(image.get_psf(0.8, "cm"))

    assert 3.343415291615846 == pytest.approx(image.get_psf(0.8, "cm"))


def test_get_cumulative_data(psf_image):
    image = psf_image

    cumulative_data = image.get_cumulative_data()
    assert len(cumulative_data) == 30
    assert cumulative_data.Radius[0] == 0
    assert cumulative_data.Intensity[0] == 0

    # test with radius
    radius = 50.0 * u.cm
    cumulative_data_radius = image.get_cumulative_data(radius=radius)
    assert len(cumulative_data_radius) == 1
    assert pytest.approx(cumulative_data_radius.Radius[0]) == 50.0


def test_get_image_data(psf_image):
    image = psf_image

    # Test centralized data
    data_centralized = image.get_image_data(centralized=True)
    assert np.allclose(data_centralized.X, image.photon_pos_x - image.centroid_x)
    assert np.allclose(data_centralized.Y, image.photon_pos_y - image.centroid_y)

    # Test non-centralized data
    data_non_centralized = image.get_image_data(centralized=False)
    assert np.allclose(data_non_centralized.X, image.photon_pos_x)
    assert np.allclose(data_non_centralized.Y, image.photon_pos_y)


def test_set_psf(caplog):
    image = PSFImage(focal_length=2800.0)

    # Test setting PSF in cm
    image.set_psf(1.5, fraction=0.8, unit="cm")
    assert image._stored_psf[0.8] == 1.5

    # Test setting PSF in deg
    image.set_psf(0.05, fraction=0.8, unit="deg")
    expected_value = 0.05 / image._cm_to_deg
    assert image._stored_psf[0.8] == pytest.approx(expected_value)

    # Test setting PSF in deg without focal length
    image_no_focal = PSFImage()
    with caplog.at_level(logging.ERROR):
        image_no_focal.set_psf(0.05, fraction=0.8, unit="deg")
    assert "PSF cannot be set" in caplog.text
    assert 0.8 not in image_no_focal._stored_psf

    # Test setting PSF with different fractions
    image.set_psf(2.0, fraction=0.5, unit="cm")
    assert image._stored_psf[0.5] == 2.0
    image.set_psf(0.1, fraction=0.5, unit="deg")
    expected_value_fraction_0_5 = 0.1 / image._cm_to_deg
    assert image._stored_psf[0.5] == pytest.approx(expected_value_fraction_0_5)


def test_find_radius_by_scanning(psf_image):
    image = psf_image

    target_number = 680
    radius_sig = 1.0

    radius = image._find_radius_by_scanning(target_number, radius_sig)
    assert radius > 0

    # Test with a target number that is too high
    with pytest.raises(RuntimeError):
        image._find_radius_by_scanning(2000, radius_sig)

    # Test with a target number that is too low
    with pytest.raises(RuntimeError):
        image._find_radius_by_scanning(0, radius_sig)


def test_sum_photons_in_radius(psf_image):
    image = psf_image

    # Test with a radius that includes all photons
    radius = np.max(image.photon_r)
    assert image._sum_photons_in_radius(radius * 1.1) == image._number_of_detected_photons

    # Test with a radius that includes none of the photons
    radius = np.min(image.photon_r) - 1
    assert image._sum_photons_in_radius(radius) == 0

    # Test with a radius that includes half of the photons
    median_radius = np.median(image.photon_r)
    assert image._sum_photons_in_radius(median_radius) == image._number_of_detected_photons // 2


def test_find_psf(psf_image):
    image = psf_image

    psf = image._find_psf(fraction=0.8)
    assert psf > 0


def test_get_effective_area(psf_image, caplog):
    image = psf_image

    # Test when effective area is set
    image.set_effective_area(100.0)
    assert image.get_effective_area() == 100.0

    # Test when effective area is not set
    image_no_effective_area = PSFImage()
    with caplog.at_level(logging.ERROR):
        assert image_no_effective_area.get_effective_area() is None
    assert "Effective Area could not be calculated" in caplog.text


def test_get_psf(psf_image, caplog):
    image = psf_image

    psf_cm = image.get_psf(fraction=0.8, unit="cm")
    assert psf_cm > 0
    assert 0.8 in image._stored_psf

    psf_deg = image.get_psf(fraction=0.8, unit="deg")
    expected_value = image._stored_psf[0.8] * image._cm_to_deg
    assert psf_deg == pytest.approx(expected_value)

    image_no_focal = PSFImage()
    with caplog.at_level(logging.ERROR):
        psf_no_focal = image_no_focal.get_psf(fraction=0.8, unit="deg")
    assert psf_no_focal is None
    assert "PSF cannot be computed in deg because focal length is not set" in caplog.text

    psf_fraction_0_5 = image.get_psf(fraction=0.5, unit="cm")
    assert psf_fraction_0_5 > 0
    assert 0.5 in image._stored_psf
