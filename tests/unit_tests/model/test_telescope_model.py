#!/usr/bin/python3

import filecmp
import pytest
import logging
from pathlib import Path

from simtools.model.telescope_model import TelescopeModel, InvalidParameter


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

MODULE_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def lst_config_file():
    return MODULE_DIR.parent / '../resources/CTA-North-LST-1-Current_test-telescope-model.cfg'


@pytest.fixture
def telescope_model(cfg_setup, lst_config_file):

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
    logger.info(
        "Focal Length = {}, type = {}".format(flenInfo["Value"], flenInfo["Type"])
    )

    assert isinstance(flenInfo["Value"], float)


def test_cfg_file(telescope_model, lst_config_file):

    telModel = telescope_model

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
    assert False == filecmp.cmp(lst_config_file, tel.getConfigFile())


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
        "tel._isExportedModelFiles should be False because exportConfigFile"
        " was not called yet."
    )
    assert False == tel._isExportedModelFilesUpToDate

    # Exporting config file
    tel.exportConfigFile()
    logger.debug(
        "tel._isExportedModelFiles should be True because exportConfigFile"
        " was called."
    )
    assert tel._isExportedModelFilesUpToDate

    # Changing a non-file parameter
    logger.info(
        "Changing a parameter that IS NOT a file - mirror_reflection_random_angle"
    )
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
    logger.debug(
        "Changing a parameter that IS a file - camera_config_file"
    )
    tel.changeParameter(
        "camera_config_file",
        tel.getParameterValue("camera_config_file")
    )
    logger.debug(
        "tel._isExportedModelFiles should be False because a parameter that "
        "is a file was changed."
    )
    assert False == tel._isExportedModelFilesUpToDate

    # TODO - test without assert / raise
    # Testing the DB connection
    logger.info("DB should be read next.")
    tel.exportConfigFile()
