#!/usr/bin/python3

import logging
from copy import copy

import pytest
from astropy import units as u

import simtools.utils.general as gen
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
def corsika_config(io_handler, db_config, corsika_config_data):
    corsika_config = CorsikaConfig(
        mongo_db_config=db_config,
        site="Paranal",
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


def test_wrong_par_in_config_data(corsika_config, corsika_config_data, db_config):
    logger.info("test_wrong_primary_name")
    new_config_data = copy(corsika_config_data)
    new_config_data["wrong_par"] = 20 * u.m
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            site="LaPalma",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_units_of_config_data(corsika_config, corsika_config_data, db_config):
    logger.info("test_units_of_config_data")
    new_config_data = copy(corsika_config_data)
    new_config_data["zenith"] = 20 * u.m
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            site="LaPalma",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_len_of_config_data(corsika_config, corsika_config_data, db_config):
    logger.info("test_len_of_config_data")
    new_config_data = copy(corsika_config_data)
    new_config_data["erange"] = [20 * u.TeV]
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            site="LaPalma",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_wrong_primary_name(corsika_config, corsika_config_data, db_config):
    logger.info("test_wrong_primary_name")
    new_config_data = copy(corsika_config_data)
    new_config_data["primary"] = "rock"
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            site="LaPalma",
            layout_name="1LST",
            label="test-corsika-config",
            corsika_config_data=new_config_data,
        )
        corsika_test_Config.print_user_parameters()


def test_missing_input(corsika_config, corsika_config_data, db_config):
    logger.info("test_missing_input")
    new_config_data = copy(corsika_config_data)
    new_config_data.pop("primary")
    with pytest.raises(MissingRequiredInputInCorsikaConfigData):
        corsika_test_Config = CorsikaConfig(
            mongo_db_config=db_config,
            site="LaPalma",
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


def test_config_data_from_yaml_file(db, io_handler, db_config):
    logger.info("test_config_data_from_yaml_file")
    test_file_name = "corsikaConfigTest.yml"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        file_name=test_file_name,
    )

    corsika_config_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(sub_dir="model", dir_type="test")
    )
    cc = CorsikaConfig(
        mongo_db_config=db_config,
        site="Paranal",
        layout_name="4LST",
        label="test-corsika-config",
        corsika_config_file=corsika_config_file,
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


def test_load_corsika_parameters_file(corsika_config, io_handler, caplog):
    corsika_parameters_file = io_handler.get_input_data_file("parameters", "corsika_parameters.yml")
    corsika_dict = corsika_config.load_corsika_parameters_file(corsika_parameters_file)
    assert "Loading CORSIKA parameters from file" in caplog.text
    sphere_center = {
        "LST": {"value": 16, "unit": "m"},
        "MST": {"value": 9, "unit": "m"},
        "SCT": {"value": 6.1, "unit": "m"},
        "SST": {"value": 3.25, "unit": "m"},
    }
    sphere_radius = {
        "LST": {"value": 12.5, "unit": "m"},
        "MST": {"value": 9.15, "unit": "m"},
        "SCT": {"value": 7.15, "unit": "m"},
        "SST": {"value": 3, "unit": "m"},
    }
    assert isinstance(corsika_dict, dict)
    for tel_type in sphere_center:
        assert (
            sphere_center[tel_type]["value"]
            == corsika_dict["corsika_sphere_center"][tel_type]["value"]
        )
        assert (
            sphere_center[tel_type]["unit"]
            == corsika_dict["corsika_sphere_center"][tel_type]["unit"]
        )
    for tel_type in sphere_radius:
        assert (
            sphere_radius[tel_type]["value"]
            == corsika_dict["corsika_sphere_radius"][tel_type]["value"]
        )
        assert (
            sphere_radius[tel_type]["unit"]
            == corsika_dict["corsika_sphere_radius"][tel_type]["unit"]
        )
