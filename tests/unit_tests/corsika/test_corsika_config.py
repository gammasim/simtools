#!/usr/bin/python3

import logging
import pathlib
from unittest.mock import Mock, patch

import pytest

from simtools.corsika.corsika_config import CorsikaConfig, InvalidCorsikaInputError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_repr(corsika_config):
    assert "site" in repr(corsika_config)


def test_setup_configuration(io_handler, corsika_config_data):
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=corsika_config_data,
    )
    assert cc.get_config_parameter("NSHOW") == 100
    assert cc.get_config_parameter("THETAP") == [20, 20]
    assert cc.get_config_parameter("ERANGE") == [10.0, 10000.0]
    # Testing conversion between AZM (sim_telarray) and PHIP (corsika)
    assert cc.get_config_parameter("PHIP") == [180.0, 180.0]
    assert cc.get_config_parameter("CSCAT") == [10, 140000.0, 0]

    assert isinstance(cc.setup_configuration(), dict)
    cc.args_dict = None
    assert cc.setup_configuration() is None


def test_get_config_parameter(io_handler, corsika_config_data, caplog):
    cc = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=corsika_config_data,
    )
    assert isinstance(cc.get_config_parameter("NSHOW"), int)
    assert isinstance(cc.get_config_parameter("THETAP"), list)
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
    input_file = corsika_config.get_corsika_input_file()
    assert input_file.exists()
    with open(input_file) as f:
        assert "TELFIL |" not in f.read()


def test_export_input_file_multipipe(corsika_config):
    logger.info("test_export_input_file")
    corsika_config.export_input_file(use_multipipe=True)
    input_file = corsika_config.get_corsika_input_file()
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
    assert (
        cc._get_text_single_line(
            {"SPLIT_AUTO": ["15M"], "IO_BUFFER": ["800MB"], "MAX_BUNCHES": ["1000000"]}, "IACT "
        )
        == "IACT SPLIT_AUTO 15M \nIACT IO_BUFFER 800MB \nIACT MAX_BUNCHES 1000000 \n"
    )


def test_set_output_file_and_directory(corsika_config):
    cc = corsika_config
    output_file = cc._set_output_file_and_directory()
    assert (
        str(output_file)
        == "corsika_runXXXXXX_proton_za020deg_azm000deg_South_test_layout_test-corsika-config.zst"
    )
    assert isinstance(cc.config_file_path, pathlib.Path)


def test_write_seeds(io_handler):
    mock_file = Mock()
    corsika_config = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=None,
    )
    corsika_config.config = {"PRMPAR": [14], "RUNNR": [10]}
    with patch("io.open", return_value=mock_file):
        corsika_config._write_seeds(mock_file)
    assert mock_file.write.call_count == 4

    expected_calls = [_call.args[0] for _call in mock_file.write.call_args_list]
    for _call in expected_calls:
        assert _call.startswith("SEED ")
        assert _call.endswith(" 0 0\n")


def test_get_corsika_input_file(corsika_config):
    empty_config = CorsikaConfig(
        array_model=None,
        label="test-corsika-config",
        args_dict=None,
    )
    assert not empty_config._is_file_updated

    cc = corsika_config

    assert not cc._is_file_updated
    input_file = cc.get_corsika_input_file()

    assert isinstance(input_file, pathlib.Path)
    assert cc._is_file_updated


def test_get_corsika_telescope_list(corsika_config):
    cc = corsika_config
    telescope_list_str = cc.get_corsika_telescope_list()
    assert telescope_list_str.count("TELESCOPE") > 0
    assert telescope_list_str.count("LSTS") > 0
