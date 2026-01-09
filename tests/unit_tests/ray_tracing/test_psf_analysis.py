#!/usr/bin/python3

import gzip
import logging
import shutil

import numpy as np
import pytest
from astropy import units as u

from simtools import settings
from simtools.ray_tracing.psf_analysis import PSFImage


@pytest.fixture
def dummy_photon_file():
    return "dummy_file.gz"


@pytest.fixture
def mocker_gzip_open():
    return "gzip.open"


@pytest.fixture
def shutil_copyfileobj():
    return "shutil.copyfileobj"


@pytest.fixture
def psf_image():
    image = PSFImage(focal_length=2800.0, containment_fraction=0.8, total_scattered_area=100)
    rng = np.random.default_rng(seed=42)
    image.photon_pos_x = rng.normal(loc=0.0, scale=1.0, size=1000).tolist()
    image.photon_pos_y = rng.normal(loc=0.0, scale=1.0, size=1000).tolist()
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


def test_init_zero_focal_length(caplog):
    with caplog.at_level(logging.WARNING):
        PSFImage(focal_length=0.0)
    assert "Focal length is zero; no conversion from cm to deg possible." in caplog.text


def test_reading_simtel_file(io_handler, tmp_test_directory, mocker, caplog):
    test_file = io_handler.get_test_data_file(
        file_name=(
            "ray_tracing_photons_North_LSTN-01_d10.0km_za20.0deg_off0.000deg_validate_optics.lis.gz"
        ),
    )
    image = PSFImage(focal_length=2800.0)
    image.read_photon_list_from_simtel_file(test_file)
    image.get_psf(0.8, "cm")

    assert image.get_psf(0.8, "cm") == pytest.approx(3.343415291615846)

    # Copy the file to the temporary test directory
    shutil.copy(test_file, tmp_test_directory)

    # Unzip the file in the temporary test directory
    unzipped_file_path = (
        tmp_test_directory
        / "ray_tracing_photons_North_LSTN-01_d10.0km_za20.0deg_off0.000deg_validate_optics.lis"
    )
    with gzip.open(tmp_test_directory / test_file.name, "rb") as f_in:
        with open(unzipped_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    image_unzipped = PSFImage(focal_length=2800.0)
    image_unzipped.read_photon_list_from_simtel_file(unzipped_file_path)
    image_unzipped.get_psf(0.8, "cm")

    assert image_unzipped.get_psf(0.8, "cm") == pytest.approx(3.343415291615846)

    image_not_ok = PSFImage(focal_length=2800.0)
    mocker.patch.object(image_not_ok, "_is_photon_positions_ok", return_value=False)
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match=r"Problems reading sim_telarray.*file.*- invalid data"),
    ):
        image_not_ok.read_photon_list_from_simtel_file(test_file)


def test_get_cumulative_data(psf_image):
    image = psf_image

    cumulative_data = image.get_cumulative_data()
    assert len(cumulative_data) == 30
    assert cumulative_data["Radius [cm]"][0] == 0
    assert cumulative_data["Cumulative PSF"][0] == 0

    # test with radius
    radius = np.array([50.0] * u.cm)
    cumulative_data_radius = image.get_cumulative_data(radius=radius)
    assert len(cumulative_data_radius) == 1
    assert cumulative_data_radius["Radius [cm]"][0] == pytest.approx(50.0)


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
    assert image._stored_psf[0.8] == pytest.approx(1.5)

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
    assert image._stored_psf[0.5] == pytest.approx(2.0)
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
    assert image.get_effective_area() == pytest.approx(100.0)

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


def test_process_photon_list_with_rx(mocker, psf_image, dummy_photon_file):
    image = psf_image
    mocker.patch.object(image, "_process_simtel_file_using_rx")
    mocker.patch.object(image, "read_photon_list_from_simtel_file")

    use_rx = True
    image.process_photon_list(dummy_photon_file, use_rx)
    image._process_simtel_file_using_rx.assert_called_once_with(dummy_photon_file)
    image.read_photon_list_from_simtel_file.assert_not_called()


def test_process_photon_list_without_rx(mocker, psf_image, dummy_photon_file):
    image = psf_image
    mocker.patch.object(image, "_process_simtel_file_using_rx")
    mocker.patch.object(image, "read_photon_list_from_simtel_file")

    use_rx = False
    image.process_photon_list(dummy_photon_file, use_rx)
    image.read_photon_list_from_simtel_file.assert_called_once_with(dummy_photon_file)
    image._process_simtel_file_using_rx.assert_not_called()


def test_process_simtel_file_using_rx_success(
    mocker, psf_image, dummy_photon_file, mocker_gzip_open, shutil_copyfileobj
):
    image = psf_image
    mock_rx_output = "0.5 0.1 0.2 0.0 0.0 100.0\n"

    mock_result = mocker.Mock()
    mock_result.stdout = mock_rx_output
    mock_job_submit = mocker.patch(
        "simtools.job_execution.job_manager.submit", return_value=mock_result
    )

    # Mock tempfile creation
    mock_temp_file = mocker.Mock()
    mock_temp_file.name = "/tmp/temp_photon_file"
    mock_named_temp_file = mocker.patch("tempfile.NamedTemporaryFile", return_value=mock_temp_file)
    mock_temp_file.__enter__ = mocker.Mock(return_value=mock_temp_file)
    mock_temp_file.__exit__ = mocker.Mock(return_value=None)

    # Mock Path.unlink for cleanup
    mock_path_unlink = mocker.patch("pathlib.Path.unlink")

    mock_gzip_open = mocker.patch(mocker_gzip_open, mocker.mock_open(read_data=b"dummy data"))

    image._process_simtel_file_using_rx(dummy_photon_file)

    mock_job_submit.assert_called_once()
    call_args = mock_job_submit.call_args
    assert (
        call_args[0][0]
        == f"{settings.config.sim_telarray_path}/bin/rx -f {image._containment_fraction:.2f} -v"
    )
    assert call_args[1]["stdin"] == "/tmp/temp_photon_file"

    mock_gzip_open.assert_called_once_with(dummy_photon_file, "rb")
    mock_named_temp_file.assert_called_once_with(mode="wb", delete=False)
    mock_path_unlink.assert_called_once()

    assert image._stored_psf[image._containment_fraction] == pytest.approx(1.0)  # 2 * 0.5
    assert image.centroid_x == pytest.approx(0.1)
    assert image.centroid_y == pytest.approx(0.2)
    assert image._effective_area == pytest.approx(100.0)


def test_process_simtel_file_using_rx_file_not_found(mocker, psf_image, mocker_gzip_open):
    image = psf_image
    photon_file = "non_existent_file.gz"

    mocker.patch(mocker_gzip_open, side_effect=FileNotFoundError)

    with pytest.raises(FileNotFoundError, match=f"Photon list file not found: {photon_file}"):
        image._process_simtel_file_using_rx(photon_file)


def test_process_simtel_file_using_rx_unexpected_output_format(
    mocker, psf_image, dummy_photon_file, mocker_gzip_open, shutil_copyfileobj
):
    image = psf_image

    # Test case 1: Empty output
    mock_result = mocker.Mock()
    mock_result.stdout = ""
    mocker.patch("simtools.job_execution.job_manager.submit", return_value=mock_result)

    dummy_data = b"dummy data"
    mocker.patch(mocker_gzip_open, mocker.mock_open(read_data=dummy_data))

    with pytest.raises(IndexError, match="Unexpected output format from rx"):
        image._process_simtel_file_using_rx(dummy_photon_file)

    # Test case 2: Insufficient data
    mock_result.stdout = "1.0\n"
    mocker.patch("simtools.job_execution.job_manager.submit", return_value=mock_result)

    with pytest.raises(IndexError, match=r"^Unexpected output format from rx"):
        image._process_simtel_file_using_rx(dummy_photon_file)

    # Test case 3: Invalid data format
    mock_result.stdout = "not_a_good_return_value\n"
    mocker.patch("simtools.job_execution.job_manager.submit", return_value=mock_result)

    with pytest.raises(ValueError, match=r"^Invalid output format from rx"):
        image._process_simtel_file_using_rx(dummy_photon_file)


def test_read_photon_list_from_simtel_file_file_not_found(mocker, mocker_gzip_open):
    image = PSFImage(focal_length=2800.0)
    photon_file = "non_existent_file.gz"

    mocker.patch(mocker_gzip_open, side_effect=FileNotFoundError)

    with pytest.raises(FileNotFoundError):
        image.read_photon_list_from_simtel_file(photon_file)


def test_is_photon_positions_ok(psf_image):
    image = psf_image

    assert image._is_photon_positions_ok() is True

    image.photon_pos_x = []
    assert image._is_photon_positions_ok() is False

    image.photon_pos_x = psf_image.photon_pos_x  # Reset to original
    image.photon_pos_y = []
    assert image._is_photon_positions_ok() is False

    image.photon_pos_y = psf_image.photon_pos_y  # Reset to original
    image.photon_pos_x.append(0.0)  # Add an extra element to make lengths different
    assert image._is_photon_positions_ok() is False


def test_process_simtel_line_total_photons(psf_image, caplog):
    image = psf_image
    image._total_photons = 0
    line = b"# Telescope 1 with 100 photons from 1 star(s) falling on an area of 683. m^2"
    image._process_simtel_line(line)
    assert image._total_photons == 100
    assert image._total_area == pytest.approx(100.0)

    # Test conflicting total area
    line_conflict = b"# Telescope 1 with 100 photons from 1 star(s) falling on an area of 683. m^2"
    with caplog.at_level(logging.WARNING):
        image._process_simtel_line(line_conflict)
    assert "Conflicting value of the total area found" in caplog.text
    assert image._total_photons == 200
    assert image._total_area == pytest.approx(100.0)


def test_process_simtel_line_photon_positions(psf_image):
    image = psf_image
    line = b"0 0 1.0 2.0"
    image._process_simtel_line(line)
    assert image.photon_pos_x[-1] == pytest.approx(1.0)
    assert image.photon_pos_y[-1] == pytest.approx(2.0)

    line = b"0 0 3.0 4.0"
    image._process_simtel_line(line)
    assert image.photon_pos_x[-1] == pytest.approx(3.0)
    assert image.photon_pos_y[-1] == pytest.approx(4.0)


def test_process_simtel_line_comments(psf_image):
    image = psf_image
    line = b"# This is a comment"
    image._process_simtel_line(line)
    assert len(image.photon_pos_x) == 1000
    assert len(image.photon_pos_y) == 1000

    line = b""
    image._process_simtel_line(line)
    assert len(image.photon_pos_x) == 1000
    assert len(image.photon_pos_y) == 1000


def test_find_psf_brute_force(psf_image, mocker, caplog):
    image = psf_image
    mocker.patch.object(image, "_sum_photons_in_radius", return_value=100)

    fraction = 0.8
    radius_sig = 1.0

    mocker.patch.object(image, "_find_radius_by_scanning", return_value=radius_sig)
    with caplog.at_level(logging.WARNING):
        psf = image._find_psf(fraction)
    assert "Could not find PSF " in caplog.text
    assert psf == pytest.approx(2 * radius_sig)

    # enforce negative dr
    mocker.patch.object(image, "_sum_photons_in_radius", return_value=1000)
    psf = image._find_psf(fraction)
    assert psf == pytest.approx(2 * radius_sig)


def test_plot_cumulative(psf_image, mocker):
    image = psf_image
    mock_subplot = mocker.patch("matplotlib.pyplot.subplots")
    mock_ax = mocker.Mock()
    mock_subplot.return_value = (mocker.Mock(), mock_ax)

    image.plot_cumulative(color="blue", linestyle="--")

    mock_ax.plot.assert_called_once()
    args, kwargs = mock_ax.plot.call_args
    assert np.array_equal(args[0], image.get_cumulative_data()[image._PSFImage__PSF_RADIUS])
    assert np.array_equal(args[1], image.get_cumulative_data()[image._PSFImage__PSF_CUMULATIVE])
    assert kwargs["color"] == "blue"
    assert kwargs["linestyle"] == "--"
