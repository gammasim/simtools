#!/usr/bin/python3

import filecmp
import logging

import numpy as np
import pytest

import simtools.util.general as gen
from simtools.model.telescope_model import InvalidParameter, TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def lst_config_file(db, io_handler):
    testFileName = "CTA-North-LST-1-Current_test-telescope-model.cfg"
    db.export_file_db(
        dbName="test-data",
        dest=io_handler.get_output_directory(dirType="model", test=True),
        fileName=testFileName,
    )

    cfgFile = gen.find_file(
        testFileName, io_handler.get_output_directory(dirType="model", test=True)
    )
    return cfgFile


@pytest.fixture
def telescope_model_from_config_file(lst_config_file):

    label = "test-telescope-model"
    telModel = TelescopeModel.from_config_file(
        site="North",
        telescopeModelName="LST-1",
        label=label,
        configFileName=lst_config_file,
    )
    return telModel


def test_handling_parameters(telescope_model_lst):

    telModel = telescope_model_lst

    logger.info(
        "Old mirror_reflection_random_angle:{}".format(
            telModel.get_parameter_value("mirror_reflection_random_angle")
        )
    )
    logger.info("Changing mirror_reflection_random_angle")
    new_mrra = "0.0080 0 0"
    telModel.change_parameter("mirror_reflection_random_angle", new_mrra)

    assert new_mrra == telModel.get_parameter_value("mirror_reflection_random_angle")

    logging.info("Adding new_parameter")
    new_par = "23"
    telModel.add_parameter("new_parameter", new_par)

    assert new_par == telModel.get_parameter_value("new_parameter")

    with pytest.raises(InvalidParameter):
        telModel.get_parameter("bla_bla")


def test_flen_type(telescope_model_lst):

    telModel = telescope_model_lst
    flenInfo = telModel.get_parameter("focal_length")
    logger.info("Focal Length = {}, type = {}".format(flenInfo["Value"], flenInfo["Type"]))

    assert isinstance(flenInfo["Value"], float)


def test_cfg_file(telescope_model_from_config_file, lst_config_file):

    telModel = telescope_model_from_config_file

    telModel.export_config_file()

    logger.info("Config file (original): {}".format(lst_config_file))
    logger.info("Config file (new): {}".format(telModel.get_config_file()))

    assert filecmp.cmp(lst_config_file, telModel.get_config_file())

    cfgFile = telModel.get_config_file()
    tel = TelescopeModel.from_config_file(
        site="south",
        telescopeModelName="sst-d",
        label="test-sst",
        configFileName=cfgFile,
    )
    tel.export_config_file()
    logger.info("Config file (sst): {}".format(tel.get_config_file()))
    # TODO: testing that file can be written and that it is  different,
    #       but not the file has the
    #       correct contents
    assert False is filecmp.cmp(lst_config_file, tel.get_config_file())


def test_updating_export_model_files(db_config, io_handler):
    """
    It was found in derive_mirror_rnda_angle that the DB was being
    accessed each time the model was changed, because the model
    files were being re-exported. A flag called _isExportedModelFilesUpToDate
    was added to prevent this behavior. This test is meant to assure
    it is working properly.
    """

    # We need a brand new telescopeModel to avoid interference
    tel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelVersion="prod4",
        label="test-telescope-model-2",
        mongoDBConfig=db_config,
    )

    logger.debug(
        "tel._isExportedModelFiles should be False because export_config_file"
        " was not called yet."
    )
    assert False is tel._isExportedModelFilesUpToDate

    # Exporting config file
    tel.export_config_file()
    logger.debug(
        "tel._isExportedModelFiles should be True because export_config_file" " was called."
    )
    assert tel._isExportedModelFilesUpToDate

    # Changing a non-file parameter
    logger.info("Changing a parameter that IS NOT a file - mirror_reflection_random_angle")
    tel.change_parameter("mirror_reflection_random_angle", "0.0080 0 0")
    logger.debug(
        "tel._isExportedModelFiles should still be True because the changed "
        "parameter was not a file"
    )
    assert tel._isExportedModelFilesUpToDate

    # Testing the DB connection
    logger.info("DB should NOT be read next.")
    tel.export_config_file()

    # Changing a parameter that is a file
    logger.debug("Changing a parameter that IS a file - camera_config_file")
    tel.change_parameter("camera_config_file", tel.get_parameter_value("camera_config_file"))
    logger.debug(
        "tel._isExportedModelFiles should be False because a parameter that "
        "is a file was changed."
    )
    assert False is tel._isExportedModelFilesUpToDate


def test_load_reference_data(telescope_model_lst):

    telModel = telescope_model_lst

    assert telModel.reference_data["nsb_reference_value"]["Value"] == pytest.approx(0.24)


def test_export_derived_files(telescope_model_lst):

    telModel = telescope_model_lst

    _ = telModel.derived
    assert (
        telModel.get_derived_directory()
        .joinpath("ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv")
        .exists()
    )


def test_get_on_axis_eff_optical_area(telescope_model_lst):

    telModel = telescope_model_lst

    assert telModel.get_on_axis_eff_optical_area().value == pytest.approx(
        365.48310154491
    )  # Value for LST -1


def test_read_two_dim_wavelength_angle(telescope_model_sst):

    telModel = telescope_model_sst
    telModel.export_config_file()

    twoDimFile = telModel.get_parameter_value("camera_filter")
    assert telModel.get_config_directory().joinpath(twoDimFile).exists()
    twoDimDist = telModel.read_two_dim_wavelength_angle(twoDimFile)
    assert len(twoDimDist["Wavelength"]) > 0
    assert len(twoDimDist["Angle"]) > 0
    assert len(twoDimDist["z"]) > 0
    assert twoDimDist["Wavelength"][4] == pytest.approx(300)
    assert twoDimDist["Angle"][4] == pytest.approx(28)
    assert twoDimDist["z"][4][4] == pytest.approx(0.985199988)


def test_read_incidence_angle_distribution(telescope_model_sst):

    telModel = telescope_model_sst

    _ = telModel.derived
    incidenceAngleFile = telModel.get_parameter_value("camera_filter_incidence_angle")
    assert telModel.get_derived_directory().joinpath(incidenceAngleFile).exists()
    incidenceAngleDist = telModel.read_incidence_angle_distribution(incidenceAngleFile)
    assert len(incidenceAngleDist["Incidence angle"]) > 0
    assert len(incidenceAngleDist["Fraction"]) > 0
    assert incidenceAngleDist["Fraction"][
        np.nanargmin(np.abs(33.05 - incidenceAngleDist["Incidence angle"].value))
    ].value == pytest.approx(0.027980644661989726)


def test_calc_average_curve(telescope_model_sst):

    telModel = telescope_model_sst
    telModel.export_config_file()
    _ = telModel.derived

    twoDimFile = telModel.get_parameter_value("camera_filter")
    twoDimDist = telModel.read_two_dim_wavelength_angle(twoDimFile)
    incidenceAngleFile = telModel.get_parameter_value("camera_filter_incidence_angle")
    incidenceAngleDist = telModel.read_incidence_angle_distribution(incidenceAngleFile)
    averageDist = telModel.calc_average_curve(twoDimDist, incidenceAngleDist)
    assert averageDist["z"][np.nanargmin(np.abs(300 - averageDist["Wavelength"]))] == pytest.approx(
        0.9398265298920796
    )


def test_export_table_to_model_directory(telescope_model_sst):

    telModel = telescope_model_sst
    telModel.export_config_file()
    _ = telModel.derived

    twoDimFile = telModel.get_parameter_value("camera_filter")
    twoDimDist = telModel.read_two_dim_wavelength_angle(twoDimFile)
    incidenceAngleFile = telModel.get_parameter_value("camera_filter_incidence_angle")
    incidenceAngleDist = telModel.read_incidence_angle_distribution(incidenceAngleFile)
    averageDist = telModel.calc_average_curve(twoDimDist, incidenceAngleDist)
    oneDimFile = telModel.export_table_to_model_directory("test_average_curve.dat", averageDist)
    assert oneDimFile.exists()
