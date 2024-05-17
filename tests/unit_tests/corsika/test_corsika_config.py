#!/usr/bin/python3

import logging
from copy import copy

import pytest
from astropy import units as u

from simtools.corsika.corsika_config import (
    CorsikaConfig,
    InvalidCorsikaInput,
    MissingRequiredInputInCorsikaConfigData,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def corsika_config_data():
    return {
        "nshow": 100,
        "nrun": 10,
        "zenith": 20 * u.deg,
        "viewcone": 5 * u.deg,
        "erange": [10 * u.GeV, 10 * u.TeV],
        "eslope": -2,
        "phi": 0 * u.deg,
        "cscat": [10, 1500 * u.m, 0],
        "primary": "proton",
    }


@pytest.fixture
def corsika_config(io_handler, db_config, corsika_config_data, model_version):
    corsika_config = CorsikaConfig(
        mongo_db_config=db_config,
        model_version=model_version,
        site="South",
        layout_name="4LST",
        label="test-corsika-config",
        corsika_config_data=corsika_config_data,
    )
    return corsika_config


def test_repr(corsika_config):
    logger.info("test_repr")
    text = repr(corsika_config)

    assert "site" in text


def test_user_parameters(corsika_config):
    logger.info("test_user_parameters")

    assert corsika_config.get_user_parameter("nshow") == 100
    assert corsika_config.get_user_parameter("thetap") == [20, 20]
    assert corsika_config.get_user_parameter("erange") == [10.0, 10000.0]
    # Testing conversion between AZM (sim_telarray) and PHIP (corsika)
    assert corsika_config.get_user_parameter("azm") == [0.0, 0.0]
    assert corsika_config.get_user_parameter("phip") == [180.0, 180.0]

    with pytest.raises(KeyError):
        corsika_config.get_user_parameter("inexistent_par")


def test_export_input_file(corsika_config):
    logger.info("test_export_input_file")
    corsika_config.export_input_file()
    input_file = corsika_config.get_input_file()
    assert input_file.exists()
    with open(input_file, "r") as f:
        assert "TELFIL |" not in f.read()


def test_export_input_file_multipipe(corsika_config):
    logger.info("test_export_input_file")
    corsika_config.export_input_file(use_multipipe=True)
    input_file = corsika_config.get_input_file()
    assert input_file.exists()
    with open(input_file, "r") as f:
        assert "TELFIL |" in f.read()


def test_wrong_par_in_config_data(corsika_config, corsika_config_data, db_config, model_version):
    logger.info("test_wrong_primary_name")
    new_config_data = copy(corsika_config_data)
    new_config_data["wrong_par"] = 20 * u.m
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            model_version=model_version,
            site="North",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_units_of_config_data(corsika_config, corsika_config_data, db_config, model_version):
    logger.info("test_units_of_config_data")
    new_config_data = copy(corsika_config_data)
    new_config_data["zenith"] = 20 * u.m
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            model_version=model_version,
            site="North",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_len_of_config_data(corsika_config, corsika_config_data, db_config, model_version):
    logger.info("test_len_of_config_data")
    new_config_data = copy(corsika_config_data)
    new_config_data["erange"] = [20 * u.TeV]
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            model_version=model_version,
            site="North",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_wrong_primary_name(corsika_config, corsika_config_data, db_config, model_version):
    logger.info("test_wrong_primary_name")
    new_config_data = copy(corsika_config_data)
    new_config_data["primary"] = "rock"
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            model_version=model_version,
            site="North",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_missing_input(corsika_config, corsika_config_data, db_config, model_version):
    logger.info("test_missing_input")
    new_config_data = copy(corsika_config_data)
    new_config_data.pop("primary")
    with pytest.raises(MissingRequiredInputInCorsikaConfigData):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            model_version=model_version,
            site="North",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_set_user_parameters(corsika_config_data, corsika_config):
    logger.info("test_set_user_parameters")
    new_config_data = copy(corsika_config_data)
    new_config_data["zenith"] = 0 * u.deg
    new_corsika_config = copy(corsika_config)
    new_corsika_config.set_user_parameters(new_config_data)

    assert new_corsika_config.get_user_parameter("thetap") == [0, 0]


def test_config_data_from_yaml_file(io_handler, db_config, model_version):
    logger.info("test_config_data_from_yaml_file")
    cc = CorsikaConfig(
        mongo_db_config=db_config,
        model_version=model_version,
        site="South",
        layout_name="4LST",
        label="test-corsika-config",
        corsika_config_file="tests/resources/corsikaConfigTest.yml",
    )
    cc.print_user_parameters()

    test_dict = {
        "RUNNR": [10],
        "NSHOW": [100],
        "PRMPAR": [14],
        "ERANGE": [0.01, 10.0],
        "ESLOPE": [-2],
        "THETAP": [20.0, 20.0],
        "AZM": [0.0, 0.0],
        "CSCAT": [10, 150000.0, 0],
        "VIEWCONE": [0.0, 5.0],
        "EVTNR": [1],
        "PHIP": [180.0, 180.0],
    }
    assert test_dict == cc._user_parameters


def test_get_file_name(corsika_config, io_handler):
    file_name = "proton_South_4LST_za020-azm000deg_cone0-5_test-corsika-config"

    assert (
        corsika_config.get_file_name("config_tmp", run_number=1)
        == f"corsika_config_run000001_{file_name}.txt"
    )
    with pytest.raises(ValueError):
        assert (
            corsika_config.get_file_name("config_tmp")
            == f"corsika_config_run000001_{file_name}.txt"
        )

    assert corsika_config.get_file_name("config") == f"corsika_config_{file_name}.input"
    # The test below includes the placeholder XXXXXX for the run number because
    # that is the way we get the run number later in the CORSIKA input file with zero padding.
    assert corsika_config.get_file_name("output_generic") == (
        "corsika_runXXXXXX_proton_za020deg_azm000deg_South_4LST_test-corsika-config.zst"
    )
    assert corsika_config.get_file_name("multipipe") == "multi_cta-South-4LST.cfg"
    with pytest.raises(ValueError):
        corsika_config.get_file_name("foobar")


def test_convert_to_quantities(corsika_config):

    assert corsika_config._convert_to_quantities("10 m") == [10 * u.m]
    assert corsika_config._convert_to_quantities("simple_string") == ["simple_string"]
    assert corsika_config._convert_to_quantities({"value": 10, "unit": "m"}) == [10 * u.m]
    assert corsika_config._convert_to_quantities({"not_value": 10, "not_unit": "m"}) == [
        {"not_value": 10, "not_unit": "m"}
    ]
    assert corsika_config._convert_to_quantities(["10 m", "20 m", "simple_string"]) == [
        10 * u.m,
        20 * u.m,
        "simple_string",
    ]
    assert corsika_config._convert_to_quantities(
        [{"value": 10, "unit": "m"}, "20 m", "simple_string"]
    ) == [10 * u.m, 20 * u.m, "simple_string"]
