#!/usr/bin/python3

import logging
from unittest.mock import Mock

import astropy.table
import pytest

import simtools.utils.general as gen
from simtools.model.model_parameter import InvalidModelParameterError

logger = logging.getLogger()


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


def test_load_camera(telescope_model_lst, monkeypatch):
    tel_model = telescope_model_lst
    focal_length = 100
    configuration = {"rotate": 0.0, "pixel_types": [], "pixels": []}

    resolve_mock = Mock(return_value=configuration)
    monkeypatch.setattr(tel_model, "_resolve_camera_components", resolve_mock)
    tel_model.get_telescope_effective_focal_length = Mock(return_value=focal_length)
    camera_mock = Mock()
    monkeypatch.setattr("simtools.model.telescope_model.Camera", camera_mock)

    tel_model._load_camera()

    resolve_mock.assert_called_once_with()
    camera_mock.from_configuration.assert_called_once_with(
        tel_model.name, configuration, focal_length
    )
    assert tel_model._camera == camera_mock.from_configuration.return_value


def test_is_file_2d_true(telescope_model_lst, monkeypatch):
    table = astropy.table.QTable()
    monkeypatch.setattr(telescope_model_lst, "get_parameter_table", Mock(return_value=table))
    assert telescope_model_lst.is_file_2d("mirror_reflectivity") is True


def test_is_file_2d_keyerror(telescope_model_lst, caplog):
    result = telescope_model_lst.is_file_2d("missing_param")
    assert result is False
    assert "does not exist" in caplog.text


def test_get_on_axis_eff_optical_area_ok(telescope_model_lst):
    fake_table = astropy.table.Table({"Off-axis_angle": [0.0], "eff_area": [123.4]})
    telescope_model_lst.get_parameter_table = Mock(return_value=fake_table)
    assert telescope_model_lst.get_on_axis_eff_optical_area() == pytest.approx(123.4)


def test_get_on_axis_eff_optical_area_wrong_angle(telescope_model_lst):
    fake_table = astropy.table.Table({"Off-axis_angle": [1.0], "eff_area": [123.4]})
    telescope_model_lst.get_parameter_table = Mock(return_value=fake_table)
    with pytest.raises(ValueError, match=r"^No value for the on-axis"):
        telescope_model_lst.get_on_axis_eff_optical_area()


def test_get_calibration_device_name(telescope_model_lst):
    tel_model = telescope_model_lst

    # Test case 1: Parameter exists and device type found
    mock_devices = {"flasher": "my_flasher_device", "illuminator": "my_illuminator_device"}
    tel_model.get_parameter_value = Mock(return_value=mock_devices)
    assert tel_model.get_calibration_device_name("flasher") == "my_flasher_device"

    # Test case 2: Device type not found
    assert tel_model.get_calibration_device_name("nonexistent_device") is None

    # Test case 3: Parameter is None
    tel_model.get_parameter_value = Mock(return_value=None)
    assert tel_model.get_calibration_device_name("flasher") is None

    # Test case 4: Parameter does not exist
    tel_model.get_parameter_value = Mock(
        side_effect=InvalidModelParameterError("Parameter not found")
    )
    assert tel_model.get_calibration_device_name("flasher") is None


def test_mirrors_property(telescope_model_lst, monkeypatch):
    tel_model = telescope_model_lst
    load_mirrors_mock = Mock()
    monkeypatch.setattr(tel_model, "_load_mirrors", load_mirrors_mock)

    # First call should load mirrors
    _ = tel_model.mirrors
    assert load_mirrors_mock.call_count == 1

    # Second call should not load again if mirrors already loaded
    tel_model._mirrors = "mock_mirrors"
    result = tel_model.mirrors
    assert result == "mock_mirrors"
    assert load_mirrors_mock.call_count == 1


def test_export_single_mirror_list_file(telescope_model_lst, caplog):
    tel_model = telescope_model_lst
    tel_model._mirrors = Mock()
    tel_model._mirrors.number_of_mirrors = 5

    # Mock simtel_config_writer
    mock_writer = Mock()
    tel_model._load_simtel_config_writer = Mock()
    tel_model.simtel_config_writer = mock_writer

    # Test valid mirror number
    tel_model.export_single_mirror_list_file(mirror_number=1, set_focal_length_to_zero=True)
    assert mock_writer.write_single_mirror_list_file.called
    assert 1 in tel_model._single_mirror_list_file_paths

    # Test invalid mirror number (too high)
    with caplog.at_level(logging.ERROR):
        tel_model.export_single_mirror_list_file(mirror_number=10, set_focal_length_to_zero=False)
    assert "mirror_number > number_of_mirrors" in caplog.text


def test_get_telescope_effective_focal_length(telescope_model_lst):
    import astropy.units as u

    tel_model = telescope_model_lst

    # Test case 1: Normal case with effective_focal_length
    mock_value = 2.15 * u.m
    tel_model.get_parameter_value_with_unit = Mock(return_value=mock_value)
    result = tel_model.get_telescope_effective_focal_length(unit="m")
    assert result == pytest.approx(2.15)

    # Test case 2: effective_focal_length returns tuple
    tel_model.get_parameter_value_with_unit = Mock(return_value=(mock_value,))
    result = tel_model.get_telescope_effective_focal_length(unit="cm")
    assert result == pytest.approx(215.0)

    # Test case 3: effective_focal_length is None (AttributeError)
    tel_model.get_parameter_value_with_unit = Mock(return_value=None)
    result = tel_model.get_telescope_effective_focal_length(unit="m")
    assert result == pytest.approx(0.0)

    # Test case 4: return_focal_length_if_zero=True with 0 value
    tel_model.get_parameter_value_with_unit = Mock(
        side_effect=[
            0.0 * u.m,  # effective_focal_length
            1600.0 * u.cm,  # focal_length fallback
        ]
    )
    result = tel_model.get_telescope_effective_focal_length(
        unit="m", return_focal_length_if_zero=True
    )
    assert result == pytest.approx(16.0)


def test_read_two_dim_wavelength_angle(telescope_model_lst):
    table = astropy.table.QTable(
        {
            "wavelength": [300.0, 300.0, 400.0, 400.0],
            "angle": [0.0, 10.0, 0.0, 10.0],
            "reflectivity": [0.8, 0.75, 0.9, 0.85],
        }
    )
    telescope_model_lst.get_parameter_table = Mock(return_value=table)

    result = telescope_model_lst.read_two_dim_wavelength_angle("mirror_reflectivity")

    assert "Wavelength" in result
    assert "Angle" in result
    assert "z" in result
    assert len(result["Wavelength"]) == 2
    assert len(result["Angle"]) == 2
    assert result["z"].shape == (2, 2)


def test_read_incidence_angle_distribution(telescope_model_lst):
    incidence_table = astropy.table.Table(
        {
            "Incidence angle": [0.0, 10.0],
            "Fraction": [0.5, 0.3],
        }
    )
    telescope_model_lst.get_parameter_table = Mock(return_value=incidence_table)
    result = telescope_model_lst.read_incidence_angle_distribution("incidence_angle")

    assert isinstance(result, astropy.table.Table)
    assert len(result) == 2
    assert "Incidence angle" in result.colnames
    assert "Fraction" in result.colnames


def test_calc_average_curve():
    import numpy as np

    from simtools.model.telescope_model import TelescopeModel

    # Mock curves data
    curves = {
        "Wavelength": np.array([300.0, 400.0, 500.0]),
        "Angle": np.array([0.0, 10.0, 20.0]),
        "z": np.array([[0.8, 0.85, 0.9], [0.75, 0.8, 0.85], [0.7, 0.75, 0.8]]),
    }

    # Mock incidence angle distribution
    incidence_angle_dist = astropy.table.Table(
        {
            "Incidence angle": [0.0, 10.0, 20.0],
            "Fraction": [0.5, 0.3, 0.2],
        }
    )

    result = TelescopeModel.calc_average_curve(curves, incidence_angle_dist)

    assert isinstance(result, astropy.table.Table)
    assert "Wavelength" in result.colnames
    assert "z" in result.colnames
    assert len(result) == 3


def test_export_table_to_model_directory(telescope_model_lst):
    tel_model = telescope_model_lst

    # Create a mock table
    test_table = astropy.table.Table({"Wavelength": [300.0, 400.0], "Value": [0.8, 0.9]})

    file_name = "test_output.dat"
    result = tel_model.export_table_to_model_directory(file_name, test_table)

    expected_path = tel_model.config_file_directory / file_name
    assert result == expected_path.absolute()
    assert expected_path.exists()
