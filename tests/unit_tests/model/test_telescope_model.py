#!/usr/bin/python3

import copy
import logging

import numpy as np
import pytest

from simtools.model.model_parameter import InvalidModelParameterError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.mark.xfail(reason="Missing ray_tracing for prod6 in Derived-DB")
def test_get_on_axis_eff_optical_area(telescope_model_lst):
    tel_model = telescope_model_lst

    assert tel_model.get_on_axis_eff_optical_area().value == pytest.approx(
        365.48310154491
    )  # Value for LST -1


# TODO - remove dependency on prod5 as soon as prod6 is complete in DB
def test_read_two_dim_wavelength_angle(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.export_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    assert tel_model.config_file_directory.joinpath(two_dim_file).exists()
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    assert len(two_dim_dist["Wavelength"]) > 0
    assert len(two_dim_dist["Angle"]) > 0
    assert len(two_dim_dist["z"]) > 0
    assert two_dim_dist["Wavelength"][4] == pytest.approx(300)
    assert two_dim_dist["Angle"][4] == pytest.approx(28)
    assert two_dim_dist["z"][4][4] == pytest.approx(0.985199988)


# TODO - remove dependency on prod5 as soon as prod6 is complete in DB
def test_read_incidence_angle_distribution(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5

    _ = tel_model.derived
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    assert tel_model.config_file_directory.joinpath(incidence_angle_file).exists()
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    assert len(incidence_angle_dist["Incidence angle"]) > 0
    assert len(incidence_angle_dist["Fraction"]) > 0
    assert incidence_angle_dist["Fraction"][
        np.nanargmin(np.abs(33.05 - incidence_angle_dist["Incidence angle"].value))
    ].value == pytest.approx(0.027980644661989726)


# TODO - remove dependency on prod5 as soon as prod6 is complete in DB
def test_calc_average_curve(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.export_config_file()
    _ = tel_model.derived

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    average_dist = tel_model.calc_average_curve(two_dim_dist, incidence_angle_dist)
    assert average_dist["z"][
        np.nanargmin(np.abs(300 - average_dist["Wavelength"]))
    ] == pytest.approx(0.9398265298920796)


# TODO - remove dependency on prod5 as soon as prod6 is complete in DB
def test_export_table_to_model_directory(telescope_model_sst_prod5):
    tel_model = telescope_model_sst_prod5
    tel_model.export_config_file()
    _ = tel_model.derived

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
    tel_model_sst._parameters["effective_focal_length"]["value"] = 0.0
    assert tel_model_sst.get_telescope_effective_focal_length("m") == pytest.approx(0.0)
    assert tel_model_sst.get_telescope_effective_focal_length("m", True) == pytest.approx(2.15)
    tel_model_sst._parameters["effective_focal_length"]["value"] = None
    assert tel_model_sst.get_telescope_effective_focal_length("m", True) == pytest.approx(2.15)


def test_position(telescope_model_lst, caplog):
    tel_model = telescope_model_lst
    xyz = tel_model.position(coordinate_system="ground")
    assert pytest.approx(xyz[0].value) == -70.91
    assert pytest.approx(xyz[1].value) == -52.35
    assert pytest.approx(xyz[2].value) == 45.0
    utm_xyz = tel_model.position(coordinate_system="utm")
    assert pytest.approx(utm_xyz[0].value) == 217659.6
    assert pytest.approx(utm_xyz[1].value) == 3184995.1
    assert pytest.approx(utm_xyz[2].value) == 2185.0

    with pytest.raises(InvalidModelParameterError):
        tel_model.position(coordinate_system="invalid")

    assert any(
        "Coordinate system invalid not found." in message for message in caplog.text.splitlines()
    )
