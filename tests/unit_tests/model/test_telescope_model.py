#!/usr/bin/python3

import copy
import logging
from unittest.mock import Mock, mock_open, patch

import astropy.table
import numpy as np
import pytest

import simtools.utils.general as gen
from simtools.model.model_parameter import InvalidModelParameterError

logger = logging.getLogger()


# depends on prod5; prod6 is incomplete in the DB
def test_read_two_dim_wavelength_angle(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.write_sim_telarray_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    assert tel_model.config_file_directory.joinpath(two_dim_file).exists()
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    assert len(two_dim_dist["Wavelength"]) > 0
    assert len(two_dim_dist["Angle"]) > 0
    assert len(two_dim_dist["z"]) > 0
    assert two_dim_dist["Wavelength"][4] == pytest.approx(300)
    assert two_dim_dist["Angle"][4] == pytest.approx(28)
    assert two_dim_dist["z"][4][4] == pytest.approx(0.985199988)


def test_read_incidence_angle_distribution(telescope_model_sst):
    tel_model = telescope_model_sst
    tel_model.export_model_files()

    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    assert tel_model.config_file_directory.joinpath(incidence_angle_file).exists()
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    assert len(incidence_angle_dist["Incidence angle"]) > 0
    assert len(incidence_angle_dist["Fraction"]) > 0
    assert incidence_angle_dist["Fraction"][
        np.nanargmin(np.abs(33.05 - incidence_angle_dist["Incidence angle"].value))
    ].value == pytest.approx(0.027980644661989726)


# depends on prod5 (no 2D camera file file in prod6)
def test_calc_average_curve(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.write_sim_telarray_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    average_dist = tel_model.calc_average_curve(two_dim_dist, incidence_angle_dist)
    assert average_dist["z"][
        np.nanargmin(np.abs(300 - average_dist["Wavelength"]))
    ] == pytest.approx(0.9398265298920796)


# depends on prod5 (no 2D camera file file in prod6)
def test_export_table_to_model_directory(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.write_sim_telarray_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    average_dist = tel_model.calc_average_curve(two_dim_dist, incidence_angle_dist)
    one_dim_file = tel_model.export_table_to_model_directory("test_average_curve.dat", average_dist)
    assert one_dim_file.exists()


def test_get_telescope_effective_focal_length(telescope_model_lst, telescope_model_sst_prod5):
    tel_model_lst = copy.deepcopy(telescope_model_lst)
    assert tel_model_lst.get_telescope_effective_focal_length("m") == pytest.approx(29.237)

    tel_model_sst = copy.deepcopy(telescope_model_sst_prod5)
    assert tel_model_sst.get_telescope_effective_focal_length("m") == pytest.approx(2.15191)

    # test zero case
    tel_model_sst.parameters["effective_focal_length"]["value"] = 0.0
    assert tel_model_sst.get_telescope_effective_focal_length("m") == pytest.approx(0.0)
    assert tel_model_sst.get_telescope_effective_focal_length("m", True) == pytest.approx(2.15)
    tel_model_sst.parameters["effective_focal_length"]["value"] = None
    assert tel_model_sst.get_telescope_effective_focal_length("m", True) == pytest.approx(2.15)


def test_position(telescope_model_lst, caplog):
    tel_model = telescope_model_lst
    xyz = tel_model.position(coordinate_system="ground")
    assert xyz[0].value == pytest.approx(-70.91)
    assert xyz[1].value == pytest.approx(-52.35)
    assert xyz[2].value == pytest.approx(45.0)
    utm_xyz = tel_model.position(coordinate_system="utm")
    assert utm_xyz[0].value == pytest.approx(217659.6)
    assert utm_xyz[1].value == pytest.approx(3184995.1)
    assert utm_xyz[2].value == pytest.approx(2185.0)
    with pytest.raises(InvalidModelParameterError, match=r"Coordinate system invalid not found."):
        tel_model.position(coordinate_system="invalid")


def test_export_single_mirror_list_file(telescope_model_lst, caplog, monkeypatch):
    tel_model = telescope_model_lst
    tel_model.write_sim_telarray_config_file()
    mirror_number = 1

    monkeypatch.setattr(
        tel_model,
        "_load_simtel_config_writer",
        lambda *args, **kwargs: setattr(tel_model, "simtel_config_writer", Mock()),
    )

    # Case 1: Valid mirror number
    tel_model.export_single_mirror_list_file(mirror_number, False)
    assert mirror_number in tel_model._single_mirror_list_file_paths
    tel_model.simtel_config_writer.write_single_mirror_list_file.assert_called_once()

    # Case 2: Invalid mirror number
    mirror_number = 9999
    with caplog.at_level(logging.ERROR):
        tel_model.export_single_mirror_list_file(mirror_number, False)
    assert "mirror_number > number_of_mirrors" in caplog.text
    assert mirror_number not in tel_model._single_mirror_list_file_paths


def test_camera(telescope_model_lst, monkeypatch):
    tel_model = telescope_model_lst
    load_camera_mock = Mock()
    monkeypatch.setattr(tel_model, "_load_camera", load_camera_mock)

    # First call should load the camera
    _ = tel_model.camera
    assert load_camera_mock.call_count == 1

    # Call count should not increase, at it returns the loaded camera if camera is set
    tel_model._camera = "camera"
    _ = tel_model.camera
    assert load_camera_mock.call_count == 1


def test_get_single_mirror_list_file(telescope_model_lst, monkeypatch):
    tel_model = telescope_model_lst
    export_single_mirror_list_file_mock = Mock()
    monkeypatch.setattr(
        tel_model, "export_single_mirror_list_file", export_single_mirror_list_file_mock
    )

    mirror_number = 1
    set_focal_length_to_zero = True
    tel_model._single_mirror_list_file_paths = {mirror_number: "test_path"}

    # Call the method
    result = tel_model.get_single_mirror_list_file(mirror_number, set_focal_length_to_zero)

    # Assert that export_single_mirror_list_file was called with the correct arguments
    export_single_mirror_list_file_mock.assert_called_once_with(
        mirror_number, set_focal_length_to_zero
    )

    # Assert that the method returns the correct path
    assert result == "test_path"


def test_load_mirrors(telescope_model_lst, monkeypatch, caplog):
    tel_model = telescope_model_lst
    mirror_list_file_name = "mirror_list.dat"
    tel_model.get_parameter_value = Mock(return_value=mirror_list_file_name)
    find_file_mock = Mock()
    monkeypatch.setattr(gen, "find_file", find_file_mock)
    mirrors_mock = Mock()
    monkeypatch.setattr(tel_model, "_mirrors", None)
    monkeypatch.setattr("simtools.model.telescope_model.Mirrors", mirrors_mock)

    # Test case 1: File found in config directory
    find_file_mock.return_value = "path/to/mirror_list.dat"
    tel_model._load_mirrors()
    mirrors_mock.assert_called_with("path/to/mirror_list.dat", parameters=tel_model.parameters)
    assert tel_model._mirrors == mirrors_mock.return_value
    find_file_mock.reset_mock()

    # Test case 2: File not found in config directory, found in model_path
    monkeypatch.setattr(tel_model, "_mirrors", None)
    find_file_mock.side_effect = [FileNotFoundError, "path/to/model/mirror_list.dat"]
    tel_model.io_handler.model_path = "model_path"
    with caplog.at_level(logging.WARNING):
        tel_model._load_mirrors()
    assert "Mirror_list_file was not found in the config directory" in caplog.text
    assert "Using the one found in the model_path" in caplog.text
    assert find_file_mock.call_count == 2
    mirrors_mock.assert_called_with(
        "path/to/model/mirror_list.dat", parameters=tel_model.parameters
    )
    assert tel_model._mirrors == mirrors_mock.return_value

    # Test case 3: TypeError
    monkeypatch.setattr(tel_model, "_mirrors", None)
    find_file_mock.side_effect = TypeError("Undefined mirror list")
    with pytest.raises(TypeError, match="Undefined mirror list"):
        tel_model._load_mirrors()


def test_load_camera(telescope_model_lst, monkeypatch, caplog):
    tel_model = telescope_model_lst
    tel_model.write_sim_telarray_config_file()
    camera_config_file = "camera_CTA-LST-1_analogsum21_v2020-04-14.dat"
    focal_length = 100

    # Mock necessary methods and attributes
    tel_model.get_parameter_value = Mock(return_value=camera_config_file)
    tel_model.get_telescope_effective_focal_length = Mock(return_value=focal_length)
    find_file_mock = Mock()
    monkeypatch.setattr(gen, "find_file", find_file_mock)
    camera_mock = Mock()
    monkeypatch.setattr("simtools.model.telescope_model.Camera", camera_mock)

    # Test case 1: File found in config directory
    find_file_mock.return_value = camera_config_file
    tel_model._load_camera()
    camera_mock.assert_called_with(
        telescope_name=tel_model.name,
        camera_config_file=camera_config_file,
        focal_length=focal_length,
    )
    assert tel_model._camera == camera_mock.return_value
    find_file_mock.reset_mock()
    caplog.clear()

    # Test case 2: File not found in config directory, found in model_path
    monkeypatch.setattr(tel_model, "_camera", None)
    find_file_mock.side_effect = [FileNotFoundError, camera_config_file]
    tel_model.io_handler.model_path = "model_path"
    with caplog.at_level(logging.WARNING):
        tel_model._load_camera()
    assert (
        f"Camera config file {camera_config_file} not found in the config directory" in caplog.text
    )
    assert find_file_mock.call_count == 2
    camera_mock.assert_called_with(
        telescope_name=tel_model.name,
        camera_config_file=camera_config_file,
        focal_length=focal_length,
    )
    assert tel_model._camera == camera_mock.return_value
    caplog.clear()

    # Test case 3: TypeError
    monkeypatch.setattr(tel_model, "_camera", None)
    find_file_mock.side_effect = TypeError("Undefined camera config file")
    with pytest.raises(TypeError):
        tel_model._load_camera()
    assert f"Camera config file {camera_config_file} or config file directory" in caplog.text


def test_is_file_2d_true(telescope_model_lst):
    mock_self = Mock()
    mock_self.get_parameter_value.return_value = "file.txt"
    mock_self.config_file_directory.joinpath.return_value = "dummy_path"

    with patch("builtins.open", mock_open(read_data="something @RPOL@ inside")):
        result = telescope_model_lst.is_file_2d("mirror_list")
        assert result is True


def test_is_file_2d_false(telescope_model_lst):
    mock_self = Mock()
    mock_self.get_parameter_value.return_value = "file.txt"
    mock_self.config_file_directory.joinpath.return_value = "dummy_path"

    with patch("builtins.open", mock_open(read_data="no marker here")):
        result = telescope_model_lst.is_file_2d("mirror_list")
        assert result is False


def test_is_file_2d_keyerror(telescope_model_lst, caplog):
    mock_self = Mock()
    mock_self.get_parameter_value.side_effect = KeyError

    result = telescope_model_lst.is_file_2d("missing_param")
    assert result is False
    assert "does not exist" in caplog.text


def test_get_on_axis_eff_optical_area_ok(telescope_model_lst):
    mock_self = Mock()
    mock_self.get_parameter_value.return_value = "optics.txt"
    mock_self.config_file_directory.joinpath.return_value = "dummy_path"

    # Fake astropy table with correct 0 off-axis angle
    fake_table = astropy.table.Table({"Off-axis angle": [0.0], "eff_area": [123.4]})

    with patch("astropy.io.ascii.read", return_value=fake_table):
        result = telescope_model_lst.get_on_axis_eff_optical_area()
        assert result == pytest.approx(123.4)


def test_get_on_axis_eff_optical_area_wrong_angle(telescope_model_lst):
    mock_self = Mock()
    mock_self.get_parameter_value.return_value = "optics.txt"
    mock_self.config_file_directory.joinpath.return_value = "dummy_path"
    mock_self._logger = Mock()

    fake_table = astropy.table.Table({"Off-axis angle": [1.0], "eff_area": [123.4]})

    with patch("astropy.io.ascii.read", return_value=fake_table):
        with pytest.raises(ValueError, match=r"^No value for the on-axis"):
            telescope_model_lst.get_on_axis_eff_optical_area()


def test_get_calibration_device_name():
    """Test get_calibration_device_name method with mocked get_parameter_value."""
    from simtools.model.telescope_model import TelescopeModel

    # Create a mock telescope model instance
    telescope_model = Mock(spec=TelescopeModel)
    telescope_model.get_calibration_device_name = TelescopeModel.get_calibration_device_name

    # Test case 1: Parameter exists and device type found
    mock_devices = {"flasher": "my_flasher_device", "illuminator": "my_illuminator_device"}
    telescope_model.get_parameter_value = Mock(return_value=mock_devices)

    result = telescope_model.get_calibration_device_name(telescope_model, "flasher")
    assert result == "my_flasher_device"
    telescope_model.get_parameter_value.assert_called_with("calibration_devices")

    # Test case 2: Parameter exists but device type not found
    result = telescope_model.get_calibration_device_name(telescope_model, "nonexistent_device")
    assert result is None

    # Test case 3: Parameter exists but is None
    telescope_model.get_parameter_value = Mock(return_value=None)
    result = telescope_model.get_calibration_device_name(telescope_model, "flasher")
    assert result is None

    # Test case 4: Parameter does not exist (InvalidModelParameterError raised)
    telescope_model.get_parameter_value = Mock(
        side_effect=InvalidModelParameterError("Parameter not found")
    )
    result = telescope_model.get_calibration_device_name(telescope_model, "flasher")
    assert result is None
