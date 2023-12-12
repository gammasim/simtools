#!/usr/bin/python3

import filecmp
import logging

import numpy as np
import pytest
from astropy import units as u

from simtools.model.telescope_model import InvalidParameter, TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def lst_config_file():
    """Return the path to test config file for LST-1"""
    return "tests/resources/CTA-North-LST-1-Released_test-telescope-model.cfg"


@pytest.fixture
def telescope_model_from_config_file(lst_config_file):
    label = "test-telescope-model"
    tel_model = TelescopeModel.from_config_file(
        site="North",
        telescope_model_name="LST-1",
        label=label,
        config_file_name=lst_config_file,
    )
    return tel_model


def test_get_parameter_value_with_unit(telescope_model_lst):
    tel_model = telescope_model_lst

    assert isinstance(tel_model.get_parameter_value_with_unit("effective_focal_length"), u.Quantity)
    assert not isinstance(tel_model.get_parameter_value_with_unit("num_gains"), u.Quantity)


def test_handling_parameters(telescope_model_lst):
    tel_model = telescope_model_lst

    logger.info(
        "Old mirror_reflection_random_angle: "
        f"{tel_model.get_parameter_value('mirror_reflection_random_angle')}"
    )
    logger.info("Changing mirror_reflection_random_angle")
    new_mrra = "0.0080 0 0"
    tel_model.change_parameter("mirror_reflection_random_angle", new_mrra)

    assert new_mrra == tel_model.get_parameter_value("mirror_reflection_random_angle")

    logging.info("Adding new_parameter")
    new_par = "23"
    tel_model.add_parameter("new_parameter", new_par)

    assert new_par == tel_model.get_parameter_value("new_parameter")

    with pytest.raises(InvalidParameter):
        tel_model.get_parameter("bla_bla")


def test_change_parameter(telescope_model_lst):
    tel_model = telescope_model_lst

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('camera_pixels')}")
    logger.info("Testing chaging camera_pixels to a different integer")
    new_camera_pixels = 9999
    tel_model.change_parameter("camera_pixels", new_camera_pixels)

    assert new_camera_pixels == tel_model.get_parameter_value("camera_pixels")

    logger.info("Testing chaging camera_pixels to a float")
    new_camera_pixels = 9999.9
    tel_model.change_parameter("camera_pixels", new_camera_pixels)

    assert int(new_camera_pixels) == tel_model.get_parameter_value("camera_pixels")

    with pytest.raises(ValueError):
        logger.info("Testing chaging camera_pixels to a nonsense string")
        new_camera_pixels = "bla_bla"
        tel_model.change_parameter("camera_pixels", new_camera_pixels)


def test_flen_type(telescope_model_lst):
    tel_model = telescope_model_lst
    flen_info = tel_model.get_parameter("focal_length")
    logger.info(f"Focal Length = {flen_info['Value']}, type = {flen_info['Type']}")

    assert isinstance(flen_info["Value"], float)


def test_cfg_file(telescope_model_from_config_file, lst_config_file):
    tel_model = telescope_model_from_config_file

    tel_model.export_config_file()

    logger.info(f"Config file (original): {lst_config_file}")
    logger.info(f"Config file (new): {tel_model.get_config_file()}")

    assert filecmp.cmp(lst_config_file, tel_model.get_config_file())

    cfg_file = tel_model.get_config_file()
    tel = TelescopeModel.from_config_file(
        site="south",
        telescope_model_name="sst-d",
        label="test-sst",
        config_file_name=cfg_file,
    )
    tel.export_config_file()
    logger.info(f"Config file (sst): {tel.get_config_file()}")
    # TODO: testing that file can be written and that it is  different,
    #       but not the file has the
    #       correct contents
    assert False is filecmp.cmp(lst_config_file, tel.get_config_file())


def test_updating_export_model_files(db_config, io_handler):
    """
    It was found in derive_mirror_rnda_angle that the DB was being
    accessed each time the model was changed, because the model
    files were being re-exported. A flag called _is_exported_model_files_up_to_date
    was added to prevent this behavior. This test is meant to assure
    it is working properly.
    """

    # We need a brand new telescope_model to avoid interference
    tel = TelescopeModel(
        site="North",
        telescope_model_name="LST-1",
        model_version="prod4",
        label="test-telescope-model-2",
        mongo_db_config=db_config,
    )

    logger.debug(
        "tel._is_exported_model_files should be False because export_config_file"
        " was not called yet."
    )
    assert False is tel._is_exported_model_files_up_to_date

    # Exporting config file
    tel.export_config_file()
    logger.debug(
        "tel._is_exported_model_files should be True because export_config_file" " was called."
    )
    assert tel._is_exported_model_files_up_to_date

    # Changing a non-file parameter
    logger.info("Changing a parameter that IS NOT a file - mirror_reflection_random_angle")
    tel.change_parameter("mirror_reflection_random_angle", "0.0080 0 0")
    logger.debug(
        "tel._is_exported_model_files should still be True because the changed "
        "parameter was not a file"
    )
    assert tel._is_exported_model_files_up_to_date

    # Testing the DB connection
    logger.info("DB should NOT be read next.")
    tel.export_config_file()

    # Changing a parameter that is a file
    logger.debug("Changing a parameter that IS a file - camera_config_file")
    tel.change_parameter("camera_config_file", tel.get_parameter_value("camera_config_file"))
    logger.debug(
        "tel._is_exported_model_files should be False because a parameter that "
        "is a file was changed."
    )
    assert False is tel._is_exported_model_files_up_to_date


def test_load_reference_data(telescope_model_lst):
    tel_model = telescope_model_lst

    assert tel_model.reference_data["nsb_reference_value"]["Value"] == pytest.approx(0.24)


def test_export_derived_files(telescope_model_lst):
    tel_model = telescope_model_lst

    _ = tel_model.derived
    assert (
        tel_model.get_derived_directory()
        .joinpath("ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv")
        .exists()
    )


def test_get_on_axis_eff_optical_area(telescope_model_lst):
    tel_model = telescope_model_lst

    assert tel_model.get_on_axis_eff_optical_area().value == pytest.approx(
        365.48310154491
    )  # Value for LST -1


def test_read_two_dim_wavelength_angle(telescope_model_sst):
    tel_model = telescope_model_sst
    tel_model.export_config_file()

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    assert tel_model.get_config_directory().joinpath(two_dim_file).exists()
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    assert len(two_dim_dist["Wavelength"]) > 0
    assert len(two_dim_dist["Angle"]) > 0
    assert len(two_dim_dist["z"]) > 0
    assert two_dim_dist["Wavelength"][4] == pytest.approx(300)
    assert two_dim_dist["Angle"][4] == pytest.approx(28)
    assert two_dim_dist["z"][4][4] == pytest.approx(0.985199988)


def test_read_incidence_angle_distribution(telescope_model_sst):
    tel_model = telescope_model_sst

    _ = tel_model.derived
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    assert tel_model.get_derived_directory().joinpath(incidence_angle_file).exists()
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    assert len(incidence_angle_dist["Incidence angle"]) > 0
    assert len(incidence_angle_dist["Fraction"]) > 0
    assert incidence_angle_dist["Fraction"][
        np.nanargmin(np.abs(33.05 - incidence_angle_dist["Incidence angle"].value))
    ].value == pytest.approx(0.027980644661989726)


def test_calc_average_curve(telescope_model_sst):
    tel_model = telescope_model_sst
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


def test_export_table_to_model_directory(telescope_model_sst):
    tel_model = telescope_model_sst
    tel_model.export_config_file()
    _ = tel_model.derived

    two_dim_file = tel_model.get_parameter_value("camera_filter")
    two_dim_dist = tel_model.read_two_dim_wavelength_angle(two_dim_file)
    incidence_angle_file = tel_model.get_parameter_value("camera_filter_incidence_angle")
    incidence_angle_dist = tel_model.read_incidence_angle_distribution(incidence_angle_file)
    average_dist = tel_model.calc_average_curve(two_dim_dist, incidence_angle_dist)
    one_dim_file = tel_model.export_table_to_model_directory("test_average_curve.dat", average_dist)
    assert one_dim_file.exists()
