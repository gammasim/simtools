#!/usr/bin/python3

import filecmp
import logging

import numpy as np
import pytest

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.model.telescope_model import InvalidParameter, TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


@pytest.fixture
def lst_config_file(db):
    testFileName = "CTA-North-LST-1-Current_test-telescope-model.cfg"
    db.exportFileDB(
        dbName="test-data",
        dest=io.getOutputDirectory(dirType="model", test=True),
        fileName=testFileName,
    )

    cfgFile = cfg.findFile(testFileName, io.getOutputDirectory(dirType="model", test=True))
    return cfgFile


@pytest.fixture
def telescope_model_from_config_file(cfg_setup, lst_config_file):

    label = "test-telescope-model"
    telModel = TelescopeModel.fromConfigFile(
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
            telModel.getParameterValue("mirror_reflection_random_angle")
        )
    )
    logger.info("Changing mirror_reflection_random_angle")
    new_mrra = "0.0080 0 0"
    telModel.changeParameter("mirror_reflection_random_angle", new_mrra)

    assert new_mrra == telModel.getParameterValue("mirror_reflection_random_angle")

    logging.info("Adding new_parameter")
    new_par = "23"
    telModel.addParameter("new_parameter", new_par)

    assert new_par == telModel.getParameterValue("new_parameter")

    with pytest.raises(InvalidParameter):
        telModel.getParameter("bla_bla")


def test_flen_type(telescope_model_lst):

    telModel = telescope_model_lst
    flenInfo = telModel.getParameter("focal_length")
    logger.info("Focal Length = {}, type = {}".format(flenInfo["Value"], flenInfo["Type"]))

    assert isinstance(flenInfo["Value"], float)


def test_cfg_file(telescope_model_from_config_file, lst_config_file):

    telModel = telescope_model_from_config_file

    telModel.exportConfigFile()

    logger.info("Config file (original): {}".format(lst_config_file))
    logger.info("Config file (new): {}".format(telModel.getConfigFile()))

    assert filecmp.cmp(lst_config_file, telModel.getConfigFile())

    cfgFile = telModel.getConfigFile()
    tel = TelescopeModel.fromConfigFile(
        site="south",
        telescopeModelName="sst-d",
        label="test-sst",
        configFileName=cfgFile,
    )
    tel.exportConfigFile()
    logger.info("Config file (sst): {}".format(tel.getConfigFile()))
    # TODO: testing that file can be written and that it is  different,
    #       but not the file has the
    #       correct contents
    assert False is filecmp.cmp(lst_config_file, tel.getConfigFile())


def test_updating_export_model_files(set_db):
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
    )

    logger.debug(
        "tel._isExportedModelFiles should be False because exportConfigFile" " was not called yet."
    )
    assert False is tel._isExportedModelFilesUpToDate

    # Exporting config file
    tel.exportConfigFile()
    logger.debug("tel._isExportedModelFiles should be True because exportConfigFile" " was called.")
    assert tel._isExportedModelFilesUpToDate

    # Changing a non-file parameter
    logger.info("Changing a parameter that IS NOT a file - mirror_reflection_random_angle")
    tel.changeParameter("mirror_reflection_random_angle", "0.0080 0 0")
    logger.debug(
        "tel._isExportedModelFiles should still be True because the changed "
        "parameter was not a file"
    )
    assert tel._isExportedModelFilesUpToDate

    # Testing the DB connection
    logger.info("DB should NOT be read next.")
    tel.exportConfigFile()

    # Changing a parameter that is a file
    logger.debug("Changing a parameter that IS a file - camera_config_file")
    tel.changeParameter("camera_config_file", tel.getParameterValue("camera_config_file"))
    logger.debug(
        "tel._isExportedModelFiles should be False because a parameter that "
        "is a file was changed."
    )
    assert False is tel._isExportedModelFilesUpToDate


def test_load_reference_data(telescope_model_lst):

    telModel = telescope_model_lst

    assert telModel.referenceData["nsb_reference_value"]["Value"] == pytest.approx(0.24)


def test_export_derived_files(telescope_model_lst):

    telModel = telescope_model_lst

    _ = telModel.derived
    assert (
        telModel.getDerivedDirectory()
        .joinpath("ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv")
        .exists()
    )


def test_get_on_axis_eff_optical_area(telescope_model_lst):

    telModel = telescope_model_lst

    assert telModel.getOnAxisEffOpticalArea().value == pytest.approx(
        365.48310154491
    )  # Value for LST -1


def test_read_two_dim_wavelength_angle(telescope_model_sst):

    telModel = telescope_model_sst
    telModel.exportConfigFile()

    twoDimFile = telModel.getParameterValue("camera_filter")
    assert telModel.getConfigDirectory().joinpath(twoDimFile).exists()
    twoDimDist = telModel.readTwoDimWavelengthAngle(twoDimFile)
    assert len(twoDimDist["Wavelength"]) > 0
    assert len(twoDimDist["Angle"]) > 0
    assert len(twoDimDist["z"]) > 0
    assert twoDimDist["Wavelength"][4] == pytest.approx(300)
    assert twoDimDist["Angle"][4] == pytest.approx(28)
    assert twoDimDist["z"][4][4] == pytest.approx(0.985199988)


def test_read_incidence_angle_distribution(telescope_model_sst):

    telModel = telescope_model_sst

    _ = telModel.derived
    incidenceAngleFile = telModel.getParameterValue("camera_filter_incidence_angle")
    assert telModel.getDerivedDirectory().joinpath(incidenceAngleFile).exists()
    incidenceAngleDist = telModel.readIncidenceAngleDistribution(incidenceAngleFile)
    assert len(incidenceAngleDist["Incidence angle"]) > 0
    assert len(incidenceAngleDist["Fraction"]) > 0
    assert incidenceAngleDist["Fraction"][
        np.nanargmin(np.abs(33.05 - incidenceAngleDist["Incidence angle"].value))
    ].value == pytest.approx(0.027980644661989726)


def test_calc_average_curve(telescope_model_sst):

    telModel = telescope_model_sst
    telModel.exportConfigFile()
    _ = telModel.derived

    twoDimFile = telModel.getParameterValue("camera_filter")
    twoDimDist = telModel.readTwoDimWavelengthAngle(twoDimFile)
    incidenceAngleFile = telModel.getParameterValue("camera_filter_incidence_angle")
    incidenceAngleDist = telModel.readIncidenceAngleDistribution(incidenceAngleFile)
    averageDist = telModel.calcAverageCurve(twoDimDist, incidenceAngleDist)
    assert averageDist["z"][np.nanargmin(np.abs(300 - averageDist["Wavelength"]))] == pytest.approx(
        0.9398265298920796
    )


def test_export_table_to_model_directory(telescope_model_sst):

    telModel = telescope_model_sst
    telModel.exportConfigFile()
    _ = telModel.derived

    twoDimFile = telModel.getParameterValue("camera_filter")
    twoDimDist = telModel.readTwoDimWavelengthAngle(twoDimFile)
    incidenceAngleFile = telModel.getParameterValue("camera_filter_incidence_angle")
    incidenceAngleDist = telModel.readIncidenceAngleDistribution(incidenceAngleFile)
    averageDist = telModel.calcAverageCurve(twoDimDist, incidenceAngleDist)
    oneDimFile = telModel.exportTableToModelDirectory("test_average_curve.dat", averageDist)
    assert oneDimFile.exists()
