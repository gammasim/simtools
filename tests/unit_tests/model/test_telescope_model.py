#!/usr/bin/python3

import filecmp
import logging

import pytest

import simtools.util.general as gen
from simtools.model.telescope_model import InvalidParameter, TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def lst_config_file(db, io_handler):
    testFileName = "CTA-North-LST-1-Current_test-telescope-model.cfg"
    db.exportFileDB(
        dbName="test-data",
        dest=io_handler.getOutputDirectory(dirType="model", test=True),
        fileName=testFileName,
    )

    cfgFile = gen.findFile(testFileName, io_handler.getOutputDirectory(dirType="model", test=True))
    return cfgFile


@pytest.fixture
def telescope_model(db_config, io_handler):
    telescopeModel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelVersion="Prod5",
        label="test-telescope-model",
        mongoDBConfig=db_config,
    )
    return telescopeModel


@pytest.fixture
def telescope_model_from_config_file(lst_config_file):

    label = "test-telescope-model"
    telModel = TelescopeModel.fromConfigFile(
        site="North",
        telescopeModelName="LST-1",
        label=label,
        configFileName=lst_config_file,
    )
    return telModel


def test_handling_parameters(telescope_model):

    telModel = telescope_model

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


def test_flen_type(telescope_model):

    telModel = telescope_model
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


def test_load_reference_data(telescope_model):

    telModel = telescope_model

    assert telModel.referenceData["nsb_reference_value"]["Value"] == pytest.approx(0.24)


def test_export_derived_files(telescope_model):

    telModel = telescope_model

    telModel.exportDerivedFiles("ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv")
    assert (
        telModel.getDerivedDirectory()
        .joinpath("ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv")
        .exists()
    )


def test_get_on_axis_eff_optical_area(telescope_model):

    telModel = telescope_model

    assert telModel.getOnAxisEffOpticalArea().value == pytest.approx(
        365.48310154491
    )  # Value for LST -1
