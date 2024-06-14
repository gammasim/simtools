#!/usr/bin/python3

import logging

import pytest
from astropy import units as u

from simtools.corsika.corsika_config import CorsikaConfig, InvalidCorsikaInputError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def corsika_config_data():
    return {
        "nshow": 100,
        "start_run": 0,
        "nrun": 10,
        "zenith_angle": 20 * u.deg,
        "azimuth_angle": 0.0 * u.deg,
        "viewcone": "0.0 deg 5.0 deg",
        "erange": "10.0 GeV 10.0 TeV",
        "eslope": -2,
        "core_scatter": "10 1400.0 m",
        "primary": "proton",
        "data_directory": "simtools-output",
    }


@pytest.fixture()
def corsika_config(io_handler, corsika_config_data, array_model_south):
    corsika_config = CorsikaConfig(
        array_model=array_model_south,
        label="test-corsika-config",
        args_dict=corsika_config_data,
    )
    return corsika_config


def test_repr(corsika_config):
    assert "site" in repr(corsika_config)


def test_setup_configuration(io_handler, corsika_config_data, caplog):
    logger.info("test_user_parameters")
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=corsika_config_data,
    )
    assert cc.get_config_parameter("nshow") == 100
    assert cc.get_config_parameter("thetap") == [20, 20]
    assert cc.get_config_parameter("erange") == [10.0, 10000.0]
    # Testing conversion between AZM (sim_telarray) and PHIP (corsika)
    assert cc.get_config_parameter("phip") == [180.0, 180.0]
    assert cc.get_config_parameter("cscat") == [10, 140000.0, 0]
    with pytest.raises(KeyError):
        cc.get_config_parameter("not_really_a_parameter")
    assert "Parameter not_really_a_parameter" in caplog.text


def test_print_config_parameter(corsika_config, capsys):
    logger.info("test_print_config_parameter")
    corsika_config.print_config_parameter()
    assert "NSHOW" in capsys.readouterr().out


def test_export_input_file(corsika_config):
    logger.info("test_export_input_file")
    corsika_config.export_input_file()
    input_file = corsika_config.get_input_file()
    assert input_file.exists()
    with open(input_file) as f:
        assert "TELFIL |" not in f.read()


def test_export_input_file_multipipe(corsika_config):
    logger.info("test_export_input_file")
    corsika_config.export_input_file(use_multipipe=True)
    input_file = corsika_config.get_input_file()
    assert input_file.exists()
    with open(input_file) as f:
        assert "TELFIL |" in f.read()


def test_get_file_name(corsika_config, io_handler):
    file_name = "proton_South_test_layout_za020-azm000deg_cone0-5_test-corsika-config"

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
        "corsika_runXXXXXX_proton_za020deg_azm000deg_South_test_layout_test-corsika-config.zst"
    )
    assert corsika_config.get_file_name("multipipe") == "multi_cta-South-test_layout.cfg"
    with pytest.raises(ValueError):
        corsika_config.get_file_name("foobar")


def test_convert_primary_input_and_store_primary_name(io_handler):
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=None,
    )
    assert cc._convert_primary_input_and_store_primary_name("Gamma") == 1
    assert cc._convert_primary_input_and_store_primary_name("proton") == 14
    assert cc._convert_primary_input_and_store_primary_name("Helium") == 402
    assert cc._convert_primary_input_and_store_primary_name("IRON") == 5626

    with pytest.raises(InvalidCorsikaInputError):
        cc._convert_primary_input_and_store_primary_name("banana")


def test_load_corsika_default_parameters_file(io_handler):
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=None,
    )
    corsika_parameters = cc._load_corsika_default_parameters_file()
    assert isinstance(corsika_parameters, dict)
    assert "CHERENKOV_EMISSION_PARAMETERS" in corsika_parameters


def test_rotate_azimuth_by_180deg(io_handler):
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=None,
    )
    assert pytest.approx(cc._rotate_azimuth_by_180deg(0.0)) == 180.0
    assert pytest.approx(cc._rotate_azimuth_by_180deg(360.0)) == 180.0
    assert pytest.approx(cc._rotate_azimuth_by_180deg(180.0)) == 0.0
    assert pytest.approx(cc._rotate_azimuth_by_180deg(-180.0)) == 0.0


def test_get_text_single_line(io_handler):
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=None,
    )
    assert cc._get_text_single_line({"EVTNR": [1], "RUNNR": [10]}) == "EVTNR 1 \nRUNNR 10 \n"


def test_get_text_multiple_lines(io_handler):
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=None,
    )
    assert (
        cc._get_text_multiple_lines(
            {"IACT": [["SPLIT_AUTO", "15M"], ["IO_BUFFER", "800MB"], ["MAX_BUNCHES", "1000000"]]}
        )
        == "IACT SPLIT_AUTO 15M \nIACT IO_BUFFER 800MB \nIACT MAX_BUNCHES 1000000 \n"
    )
