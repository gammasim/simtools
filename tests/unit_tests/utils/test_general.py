#!/usr/bin/python3

import logging
from copy import copy
from pathlib import Path

import astropy.units as u
import pytest
import yaml

import simtools.util.general as gen
from simtools.util.general import InvalidConfigEntry

logging.getLogger().setLevel(logging.DEBUG)


def test_collect_dict_data(args_dict, io_handler):
    inDict = {"k1": 2, "k2": "bla"}
    dictForYaml = {"k3": {"kk3": 4, "kk4": 3.0}, "k4": ["bla", 2]}
    testYamlFile = io_handler.getOutputFile(fileName="test_collect_dict_data.yml", test=True)
    if not Path(testYamlFile).exists():
        with open(testYamlFile, "w") as output:
            yaml.safe_dump(dictForYaml, output, sort_keys=False)

    d1 = gen.collectDataFromYamlOrDict(None, inDict)
    assert "k2" in d1.keys()
    assert d1["k1"] == 2

    d2 = gen.collectDataFromYamlOrDict(testYamlFile, None)
    assert "k3" in d2.keys()
    assert d2["k4"] == ["bla", 2]

    d3 = gen.collectDataFromYamlOrDict(testYamlFile, inDict)
    assert d3 == d2


def test_validate_config_data(args_dict, io_handler):

    parameterFile = io_handler.getInputDataFile(fileName="test_parameters.yml", test=True)
    parameters = gen.collectDataFromYamlOrDict(parameterFile, None)

    configData = {
        "zenith": 0 * u.deg,
        "offaxis": [0 * u.deg, 0.2 * u.rad, 3 * u.deg],
        "cscat": [0, 10 * u.m, 3 * u.km],
        "sourceDistance": 20000 * u.m,
        "testName": 10,
        "dictPar": {"blah": 10, "bleh": 5 * u.m},
    }

    validatedData = gen.validateConfigData(configData=configData, parameters=parameters)

    # Testing undefined len
    assert len(validatedData.offAxisAngle) == 3

    # Testing name validation
    assert validatedData.validatedName == 10

    # Testing unit conversion
    assert validatedData.sourceDistance == 20

    # Testing dict par
    assert validatedData.dictPar["bleh"] == 500


def test_checkValueEntryLength():

    _par_info = {}
    _par_info["len"] = 2
    assert gen._checkValueEntryLength([1, 4], "test_1", _par_info) == (2, False)
    _par_info["len"] = None
    assert gen._checkValueEntryLength([1, 4], "test_1", _par_info) == (2, True)
    _par_info["len"] = 3
    with pytest.raises(InvalidConfigEntry):
        gen._checkValueEntryLength([1, 4], "test_1", _par_info)
    _par_info.pop("len")
    with pytest.raises(KeyError):
        gen._checkValueEntryLength([1, 4], "test_1", _par_info)


def test_validateAndConvertValue_with_units():

    _parname = "cscat"
    _parinfo = {"len": 4, "unit": [None, u.Unit("m"), u.Unit("m"), None], "names": ["scat"]}
    _value = [0, 10 * u.m, 3 * u.km, None]
    _value_keys = ["a", "b", "c", "d"]

    assert gen._validateAndConvertValue_with_units(_value, None, _parname, _parinfo) == [
        0,
        10.0,
        3000.0,
        None,
    ]

    assert gen._validateAndConvertValue_with_units(_value, _value_keys, _parname, _parinfo) == {
        "a": 0,
        "b": 10.0,
        "c": 3000.0,
        "d": None,
    }

    _parinfo = {"len": None, "unit": [None, u.Unit("m"), u.Unit("m"), None], "names": ["scat"]}
    with pytest.raises(InvalidConfigEntry):
        gen._validateAndConvertValue_with_units(_value, None, _parname, _parinfo)
    _parinfo = {"len": 4, "unit": [None, u.Unit("kg"), u.Unit("m"), None], "names": ["scat"]}
    with pytest.raises(InvalidConfigEntry):
        gen._validateAndConvertValue_with_units(_value, None, _parname, _parinfo)


def test_validateAndConvertValue_without_units():

    _parname = "cscat"
    _parinfo = {"len": 3, "names": ["scat"]}
    _value = [0, 10.0, 3.0]
    _value_keys = ["a", "b", "c"]

    assert gen._validateAndConvertValue_without_units(_value, None, _parname, _parinfo) == [
        0.0,
        10.0,
        3.0,
    ]
    assert gen._validateAndConvertValue_without_units(_value, _value_keys, _parname, _parinfo) == {
        "a": 0,
        "b": 10.0,
        "c": 3.0,
    }
    _value = [0, 10.0 * u.m, 3.0]
    with pytest.raises(InvalidConfigEntry):
        gen._validateAndConvertValue_without_units(_value, None, _parname, _parinfo)


def test_program_is_executable():

    # (assume 'ls' exist on any system the test is running)
    assert gen.program_is_executable("ls") is not None
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None


def test_change_dict_keys_case():

    _upper_dict = {
        "REFERENCE": {"VERSION": "0.1.0"},
        "ACTIVITY": {"NAME": "submit", "ID": "84890304", "DESCRIPTION": "Set data"},
    }
    _lower_dict = {
        "reference": {"version": "0.1.0"},
        "activity": {"name": "submit", "id": "84890304", "description": "Set data"},
    }
    _no_change_dict_upper = gen.change_dict_keys_case(copy(_upper_dict), False)
    assert _no_change_dict_upper == _upper_dict

    _no_change_dict_lower = gen.change_dict_keys_case(copy(_lower_dict), True)
    assert _no_change_dict_lower == _lower_dict

    _changed_to_lower = gen.change_dict_keys_case(copy(_upper_dict), True)
    assert _changed_to_lower == _lower_dict

    _changed_to_upper = gen.change_dict_keys_case(copy(_lower_dict), False)
    assert _changed_to_upper == _upper_dict
