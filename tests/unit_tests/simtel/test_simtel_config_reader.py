#!/usr/bin/python3

import logging

import pytest

from simtools.simtel.simtel_config_reader import SimtelConfigReader
from simtools.utils import general as gen

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def simtel_config_file():
    return "tests/resources/simtel_config_test_la_palma.cfg"


@pytest.fixture
def schema_num_gains():
    return gen.collect_data_from_file_or_dict(
        file_name="tests/resources/num_gains.schema.yml", in_dict=None
    )


@pytest.fixture
def schema_telescope_transmission():
    return gen.collect_data_from_file_or_dict(
        file_name="tests/resources/telescope_transmission.schema.yml", in_dict=None
    )


@pytest.fixture
def config_reader_num_gains(simtel_config_file, schema_num_gains):
    return SimtelConfigReader(
        schema_dict=schema_num_gains,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
        return_arrays_as_strings=True,
    )


@pytest.fixture
def config_reader_telescope_transmission(simtel_config_file, schema_telescope_transmission):
    return SimtelConfigReader(
        schema_dict=schema_telescope_transmission,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
        return_arrays_as_strings=True,
    )


def test_simtel_config_reader_num_gains(config_reader_num_gains):
    _config = config_reader_num_gains
    assert isinstance(_config.parameter_dict, dict)
    assert _config.parameter_name == "num_gains"
    assert _config.simtel_parameter_name == "NUM_GAINS"

    assert _config.parameter_dict == {
        "type": "int",
        "dimension": 1,
        "limits": "1 2",
        "default": 2,
        "CT2": 2,
    }


def test_simtel_config_reader_telescope_transmission(
    config_reader_telescope_transmission, simtel_config_file, schema_telescope_transmission
):

    _config = config_reader_telescope_transmission
    assert isinstance(_config.parameter_dict, dict)
    assert _config.parameter_name == "telescope_transmission"
    assert _config.simtel_parameter_name == "TELESCOPE_TRANSMISSION"

    assert _config.parameter_dict == {
        "type": "double",
        "dimension": 6,
        "default": "0.89 0 0 0 0",
        "CT2": "0.969 0 0 0 0 0",
    }

    _config_list = SimtelConfigReader(
        schema_dict=schema_telescope_transmission,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT8",
        return_arrays_as_strings=False,
    )

    print(_config_list.parameter_dict)
    assert _config_list.parameter_dict["dimension"] == 6
    assert _config_list.parameter_dict["type"] == "double"
    assert len(_config_list.parameter_dict["default"]) == 5
    assert _config_list.parameter_dict["default"][0] == pytest.approx(0.89)
    assert _config_list.parameter_dict["CT8"][0] == pytest.approx(0.898)
    assert _config_list.parameter_dict["CT8"][4] == pytest.approx(1.705)


def test_get_validated_parameter_dict(config_reader_num_gains):

    _config = config_reader_num_gains
    assert _config.get_validated_parameter_dict(telescope_name="MSTN-01", model_version="Test") == {
        "parameter": "num_gains",
        "instrument": "MSTN-01",
        "site": "North",
        "version": "Test",
        "value": 2,
        "unit": None,
        "type": "string",
        "applicable": True,
        "file": False,
    }


def test_export_parameter_dict_to_json(tmp_test_directory, config_reader_num_gains):

    _config = config_reader_num_gains
    _json_file = tmp_test_directory / "num_gains.json"
    _config.export_parameter_dict_to_json(
        _json_file,
        _config.get_validated_parameter_dict(telescope_name="MSTN-01", model_version="Test"),
    )

    assert _json_file.exists()


def test_compare_simtel_config_with_schema(
    config_reader_num_gains, config_reader_telescope_transmission, capfd
):

    _config_ng = config_reader_num_gains
    _config_ng.compare_simtel_config_with_schema()
    out, _ = capfd.readouterr()
    assert "from simtel: 2" in out
    assert "from schema: {'min': 1, 'max': 2})" in out

    _config_tt = config_reader_telescope_transmission
    _config_tt.compare_simtel_config_with_schema()
    out, _ = capfd.readouterr()
    assert "from simtel: 0.89 0 0 0 0" in out
    assert "from schema: None" in out


def test_read_simtel_config_file(config_reader_num_gains, simtel_config_file):

    _config_ng = config_reader_num_gains

    with pytest.raises(FileNotFoundError):
        _config_ng._read_simtel_config_file("non_existing_file.cfg", "CT1")

    # existing telescope
    _para_dict = _config_ng._read_simtel_config_file(simtel_config_file, "CT1")
    assert "CT1" in _para_dict
    # non existing telescope
    _para_dict = _config_ng._read_simtel_config_file(simtel_config_file, "CT1000")
    assert "CT1000" not in _para_dict


def test_add_value_from_simtel_cfg(config_reader_num_gains):

    _config = config_reader_num_gains

    # type
    assert _config._add_value_from_simtel_cfg(["Int", "1"], "type", None) == ("int", 1)
    assert _config._add_value_from_simtel_cfg(["Double", "5"], "type", None) == ("double", 5)
    assert _config._add_value_from_simtel_cfg(["Text", "55"], "type", None) == ("str", 1)

    # default
    assert _config._add_value_from_simtel_cfg(["2"], "default", dtype="int") == (2, 1)
    # default (comma separated, return array as string)
    assert _config._add_value_from_simtel_cfg(["0.89,0,0,0,0"], "default", dtype="double") == (
        "0.89 0 0 0 0",
        5,
    )
    # default (comma separated, return array as list)
    _config.return_arrays_as_strings = False
    _list, _ndim = _config._add_value_from_simtel_cfg(["0.89,0,0,0,0"], "default", dtype="double")
    assert _list[0] == pytest.approx(0.89)
    assert _list[2] == pytest.approx(0.0)
    assert (len(_list), _ndim) == (5, 5)
