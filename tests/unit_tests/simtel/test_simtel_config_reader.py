#!/usr/bin/python3

import copy
import logging

import numpy as np
import pytest

from simtools.simtel.simtel_config_reader import SimtelConfigReader

logger = logging.getLogger()


@pytest.fixture
def simtel_config_file():
    return "tests/resources/simtel_config_test_la_palma.cfg"


@pytest.fixture
def schema_num_gains():
    return "tests/resources/num_gains.schema.yml"


@pytest.fixture
def schema_telescope_transmission():
    return "tests/resources/telescope_transmission.schema.yml"


@pytest.fixture
def config_reader_num_gains(simtel_config_file, schema_num_gains):
    return SimtelConfigReader(
        schema_file=schema_num_gains,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
    )


@pytest.fixture
def config_reader_telescope_transmission(simtel_config_file, schema_telescope_transmission):
    return SimtelConfigReader(
        schema_file=schema_telescope_transmission,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
    )


def test_simtel_config_reader_num_gains(config_reader_num_gains):
    _config = config_reader_num_gains
    assert isinstance(_config.parameter_dict, dict)
    assert _config.parameter_name == "num_gains"
    assert _config.simtel_parameter_name == "NUM_GAINS"

    expected_dict = {
        "type": "int64",
        "dimension": 1,
        "limits": np.array([1, 2]),
        "default": 2,
        "CT2": 2,
    }

    assert all(
        np.array_equal(_config.parameter_dict[key], expected_dict[key])
        for key in _config.parameter_dict
    )


def test_simtel_config_reader_telescope_transmission(
    config_reader_telescope_transmission, simtel_config_file, schema_telescope_transmission
):
    _config = config_reader_telescope_transmission
    assert isinstance(_config.parameter_dict, dict)
    assert _config.parameter_name == "telescope_transmission"
    assert _config.simtel_parameter_name == "TELESCOPE_TRANSMISSION"

    assert _config.parameter_dict["dimension"] == 6
    assert _config.parameter_dict["type"] == "float64"
    assert len(_config.parameter_dict["default"]) == 6
    assert _config.parameter_dict["default"][0] == pytest.approx(0.89)
    assert _config.parameter_dict["CT2"][0] == pytest.approx(0.969)
    assert _config.parameter_dict["CT2"][4] == pytest.approx(0.0)
    assert len(_config.parameter_dict["CT2"]) == 6


def test_compare_simtel_config_with_schema(
    config_reader_num_gains, config_reader_telescope_transmission, caplog
):
    _config_ng = config_reader_num_gains

    # no differences; should result in no output
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _config_ng.compare_simtel_config_with_schema()
        assert caplog.text == ""

    # no limits defined for telescope_transmission
    caplog.clear()
    _config_tt = config_reader_telescope_transmission
    with caplog.at_level(logging.WARNING):
        _config_tt.compare_simtel_config_with_schema()
    assert "Values for limits do not match" in caplog.text
    assert "from simtel: TELESCOPE_TRANSMISSION None" in caplog.text
    assert "from schema: telescope_transmission [0. 1.]" in caplog.text

    # change parameter type to bool; should result in no limit check
    caplog.clear()
    with caplog.at_level("WARNING"):
        _config_tt.parameter_dict["type"] = "bool"
        _config_tt.compare_simtel_config_with_schema()
    assert "Values for limits do not match" not in caplog.text

    # remove keys and elements to enforce error tests
    caplog.clear()
    with caplog.at_level("WARNING"):
        _config_ng.schema_dict["data"][0].pop("default")
        _config_ng.compare_simtel_config_with_schema()
    assert "from schema: num_gains None" in caplog.text
    with caplog.at_level("WARNING"):
        _config_ng.schema_dict["data"].pop(0)
        _config_ng.compare_simtel_config_with_schema()
    assert "from schema: num_gains None" in caplog.text


def test_read_simtel_config_file(config_reader_num_gains, simtel_config_file, caplog):
    _config_ng = config_reader_num_gains

    with pytest.raises(FileNotFoundError):
        _config_ng.read_simtel_config_file("non_existing_file.cfg", "CT1")

    # existing telescope
    _para_dict = _config_ng.read_simtel_config_file(simtel_config_file, "CT1")
    assert "CT1" in _para_dict
    # non existing telescope
    _para_dict = _config_ng.read_simtel_config_file(simtel_config_file, "CT1000")
    assert "CT1000" not in _para_dict

    # non existing parameter
    with caplog.at_level("INFO"):
        _config_ng.simtel_parameter_name = "this parameter does not exist"
    assert _config_ng.read_simtel_config_file(simtel_config_file, "CT1") is None
    assert "not found" in caplog.text


def test_get_type_and_dimension_from_simtel_cfg(config_reader_num_gains):
    _config = copy.deepcopy(config_reader_num_gains)

    assert _config._get_type_and_dimension_from_simtel_cfg(["Int", "1"]) == ("int64", 1)
    assert _config._get_type_and_dimension_from_simtel_cfg(["Double", "5"]) == ("float64", 5)
    assert _config._get_type_and_dimension_from_simtel_cfg(["Text", "55"]) == ("str", 1)
    assert _config._get_type_and_dimension_from_simtel_cfg(["IBool", "1"]) == ("bool", 1)
    assert _config._get_type_and_dimension_from_simtel_cfg(["FUnc", "55"]) == ("str", 1)

    # fixed values for camera pixel
    _config.simtel_parameter_name = "NIGHTSKY_BACKGROUND"
    assert _config._get_type_and_dimension_from_simtel_cfg(["Double", "5"]) == ("float64", 5)
    _config.camera_pixels = 1855
    assert _config._get_type_and_dimension_from_simtel_cfg(["Double", "5"]) == ("float64", 1855)


def test_resolve_all_in_column(config_reader_num_gains):
    _config = config_reader_num_gains

    # empty
    assert _config._resolve_all_in_column([]) == ([], {})
    # no all
    assert _config._resolve_all_in_column(["1", "2", "3"]) == (["1", "2", "3"], {})
    # "all:"
    assert _config._resolve_all_in_column(["all:", "2"]) == (["2"], {})
    # "all:1"
    assert _config._resolve_all_in_column(["all:1"]) == (["1"], {})
    # "all: 1"
    assert _config._resolve_all_in_column(["all: 1"]) == (["1"], {})

    # "all: 0, 3:500"
    assert _config._resolve_all_in_column(["all:1", "3:5"]) == (["1"], {"3": "5"})


def test_add_value_from_simtel_cfg(config_reader_num_gains):
    _config = config_reader_num_gains

    # None
    assert _config._add_value_from_simtel_cfg(["None"], dtype="str") == (None, 1)
    assert _config._add_value_from_simtel_cfg(["none"], dtype="str") == (None, 1)
    assert _config._add_value_from_simtel_cfg(["none"], dtype=None) == (None, 1)
    assert _config._add_value_from_simtel_cfg(["22"], dtype=None) == ("22", 1)

    # default
    assert _config._add_value_from_simtel_cfg(["2"], dtype="int") == (2, 1)
    assert _config._add_value_from_simtel_cfg(["all", "5"], dtype="int") == (5, 1)
    assert _config._add_value_from_simtel_cfg(["all:5"], dtype="int") == (5, 1)
    assert _config._add_value_from_simtel_cfg(["all: 5"], dtype="int") == (5, 1)
    value, ndim = _config._add_value_from_simtel_cfg(["all:5", "2:1"], dtype="int", n_dim=4)
    assert list(value) == [5, 5, 1, 5]
    assert ndim == 4

    # comma separated
    _list, _ndim = _config._add_value_from_simtel_cfg(["0.89,0,0,0,0"], dtype="double")
    assert _list[0] == pytest.approx(0.89)
    assert _list[2] == pytest.approx(0.0)
    assert (len(_list), _ndim) == (5, 5)

    # boolean values with 0,1 as input
    assert _config._add_value_from_simtel_cfg(["0"], dtype="bool") == (False, 1)
    assert _config._add_value_from_simtel_cfg(["1"], dtype="bool") == (True, 1)
    _list, _ndim = _config._add_value_from_simtel_cfg(["0", "1", "5"], dtype="bool")
    assert _ndim == 3
    assert not _list[0]
    assert _list[1]
    assert _list[2]

    # no input / output
    assert _config._add_value_from_simtel_cfg([], dtype="double") == (None, None)


def test_get_simtel_parameter_name(config_reader_num_gains):
    _config = copy.deepcopy(config_reader_num_gains)
    assert _config._get_simtel_parameter_name("num_gains") == "NUM_GAINS"
    assert _config._get_simtel_parameter_name("telescope_transmission") == "TELESCOPE_TRANSMISSION"
    assert _config._get_simtel_parameter_name("NUM_GAINS") == "NUM_GAINS"
    # test pass on TypeError
    _config.schema_dict = None
    assert _config._get_simtel_parameter_name("num_gains") == "NUM_GAINS"
