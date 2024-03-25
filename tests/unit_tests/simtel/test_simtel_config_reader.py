#!/usr/bin/python3

import copy
import logging

import astropy.units as u
import numpy as np
import pytest

from simtools.simtel.simtel_config_reader import JsonNumpyEncoder, SimtelConfigReader

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
        return_arrays_as_strings=True,
    )


@pytest.fixture
def config_reader_telescope_transmission(simtel_config_file, schema_telescope_transmission):
    return SimtelConfigReader(
        schema_file=schema_telescope_transmission,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
        return_arrays_as_strings=True,
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


def test_get_validated_parameter_dict(config_reader_num_gains):

    _config = config_reader_num_gains
    assert _config.get_validated_parameter_dict(telescope_name="MSTN-01", model_version="Test") == {
        "parameter": "num_gains",
        "instrument": "MSTN-01",
        "site": "North",
        "version": "Test",
        "value": 2,
        "unit": u.Unit(""),
        "type": "int64",
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
    config_reader_num_gains, config_reader_telescope_transmission, caplog
):

    _config_ng = config_reader_num_gains

    with caplog.at_level(logging.WARNING):
        _config_ng.compare_simtel_config_with_schema()
        assert "from simtel: NUM_GAINS" in caplog.text
        assert "from schema: num_gains" in caplog.text

    caplog.clear()
    _config_tt = config_reader_telescope_transmission
    with caplog.at_level(logging.WARNING):
        _config_tt.compare_simtel_config_with_schema()
        assert "from simtel: TELESCOPE_TRANSMISSION [0.89" in caplog.text
        assert "from schema: telescope_transmission [0.0, 1.0]" in caplog.text


def test_read_simtel_config_file(config_reader_num_gains, simtel_config_file, caplog):

    _config_ng = config_reader_num_gains

    with pytest.raises(FileNotFoundError):
        _config_ng._read_simtel_config_file("non_existing_file.cfg", "CT1")

    # existing telescope
    _para_dict = _config_ng._read_simtel_config_file(simtel_config_file, "CT1")
    assert "CT1" in _para_dict
    # non existing telescope
    _para_dict = _config_ng._read_simtel_config_file(simtel_config_file, "CT1000")
    assert "CT1000" not in _para_dict

    # non existing parameter
    _config_ng.simtel_parameter_name = "this parameter does not exist"
    assert _config_ng._read_simtel_config_file(simtel_config_file, "CT1") is None
    assert "No entries found for parameter" in caplog.text


def test_get_type_from_simtel_cfg(config_reader_num_gains):

    _config = config_reader_num_gains

    # type
    assert _config._get_type_from_simtel_cfg(["Int", "1"]) == ("int64", 1)
    assert _config._get_type_from_simtel_cfg(["Double", "5"]) == ("float64", 5)
    assert _config._get_type_from_simtel_cfg(["Text", "55"]) == ("str", 1)
    assert _config._get_type_from_simtel_cfg(["IBool", "1"]) == ("bool", 1)
    assert _config._get_type_from_simtel_cfg(["FUnc", "55"]) == ("str", 1)
    _config.return_arrays_as_strings = False


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
    assert _config._add_value_from_simtel_cfg(["None"], dtype="str") == ("None", 1)
    assert _config._add_value_from_simtel_cfg(["none"], dtype="str") == ("None", 1)

    # default
    assert _config._add_value_from_simtel_cfg(["2"], dtype="int") == (2, 1)
    assert _config._add_value_from_simtel_cfg(["all", "5"], dtype="int") == (5, 1)
    assert _config._add_value_from_simtel_cfg(["all:5"], dtype="int") == (5, 1)
    assert _config._add_value_from_simtel_cfg(["all: 5"], dtype="int") == (5, 1)
    value, ndim = _config._add_value_from_simtel_cfg(["all:5", "2:1"], dtype="int", ndim=4)
    assert list(value) == [5, 5, 1, 5]
    assert ndim == 4

    # comma separated, return array as list
    _config.return_arrays_as_strings = False
    _list, _ndim = _config._add_value_from_simtel_cfg(["0.89,0,0,0,0"], dtype="double")
    assert _list[0] == pytest.approx(0.89)
    assert _list[2] == pytest.approx(0.0)
    assert (len(_list), _ndim) == (5, 5)

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


def test_check_parameter_applicability(schema_num_gains, simtel_config_file):

    _config = SimtelConfigReader(
        schema_file=schema_num_gains,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
        return_arrays_as_strings=True,
    )

    assert _config._check_parameter_applicability("LSTN-01")

    # illuminator does not have gains
    assert not _config._check_parameter_applicability("ILLN-01")

    # change schema dict
    _config.schema_dict["instrument"]["type"].append("LSTN-55")
    assert _config._check_parameter_applicability("LSTN-55")

    # change schema dict
    _config.schema_dict["instrument"].pop("type")
    with pytest.raises(KeyError):
        _config._check_parameter_applicability("LSTN-01")


def test_parameter_is_a_file(schema_num_gains, simtel_config_file):

    _config = SimtelConfigReader(
        schema_file=schema_num_gains,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
        return_arrays_as_strings=True,
    )

    assert not _config._parameter_is_a_file()

    _config.schema_dict["data"][0]["type"] = "file"
    assert _config._parameter_is_a_file()

    _config.schema_dict["data"][0].pop("type")
    assert not _config._parameter_is_a_file()

    _config.schema_dict["data"] = []
    assert not _config._parameter_is_a_file()


def test_get_unit_from_schema(schema_num_gains, simtel_config_file):

    _config = SimtelConfigReader(
        schema_file=schema_num_gains,
        simtel_config_file=simtel_config_file,
        simtel_telescope_name="CT2",
        return_arrays_as_strings=True,
    )

    assert _config._get_unit_from_schema() is None

    _config.schema_dict["data"][0]["unit"] = "m"
    assert _config._get_unit_from_schema() == "m"

    _config.schema_dict["data"][0]["unit"] = "dimensionless"
    assert _config._get_unit_from_schema() is None

    _config.schema_dict["data"][0].pop("unit")
    assert _config._get_unit_from_schema() is None


def test_validate_parameter_dict(config_reader_num_gains, caplog):

    _config = config_reader_num_gains

    _temp_dict = {
        "parameter": "num_gains",
        "instrument": "MSTN-01",
        "site": "North",
        "version": "Test",
        "value": 2,
        "unit": None,
        "type": "int",
        "applicable": True,
        "file": False,
    }
    _config._validate_parameter_dict(_temp_dict)
    _temp_dict["value"] = 25
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            _config._validate_parameter_dict(_temp_dict)
        assert "out of range" in caplog.text


def test_output_format_for_arrays(config_reader_num_gains):

    _config = config_reader_num_gains

    assert _config._output_format_for_arrays(None) is None
    assert _config._output_format_for_arrays("a") == "a"
    assert _config._output_format_for_arrays(5) == 5
    _config.return_arrays_as_strings = False
    assert np.array_equal(_config._output_format_for_arrays(np.array([1, 2, 3])), [1, 2, 3])
    _config.return_arrays_as_strings = True
    assert _config._output_format_for_arrays(np.array([1, 2, 3])) == "1 2 3"


def test_jsonnumpy_encoder():

    encoder = JsonNumpyEncoder()
    assert isinstance(encoder.default(np.float64(3.14)), float)
    assert isinstance(encoder.default(np.int64(3.14)), int)
    assert isinstance(encoder.default(np.array([])), list)
    assert isinstance(encoder.default(u.Unit("m")), str)
    assert encoder.default(u.Unit("")) is None
    assert isinstance(encoder.default(u.Unit("m/s")), str)
    assert isinstance(encoder.default(np.bool_(True)), bool)

    with pytest.raises(TypeError):
        encoder.default("abc")
