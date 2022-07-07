#!/usr/bin/python3

import logging
import yaml
from pathlib import Path
import astropy.units as u

import simtools.util.general as gen
import simtools.io_handler as io

logging.getLogger().setLevel(logging.DEBUG)


def test_collect_dict_data(cfg_setup):
    inDict = {"k1": 2, "k2": "bla"}
    dictForYaml = {
        "k3": {
          "kk3": 4,
          "kk4": 3.
        },
        "k4": ["bla", 2]
    }
    inYaml = io.getTestOutputFile("test_collect_dict_data.yml")
    if not Path(inYaml).exists():
        with open(inYaml, "w") as output:
            yaml.safe_dump(
                dictForYaml,
                output,
                sort_keys=False
            )

    d1 = gen.collectDataFromYamlOrDict(None, inDict)
    assert "k2" in d1.keys()
    assert d1["k1"] == 2

    d2 = gen.collectDataFromYamlOrDict(inYaml, None)
    assert "k3" in d2.keys()
    assert d2["k4"] == ["bla", 2]

    d3 = gen.collectDataFromYamlOrDict(inYaml, inDict)
    assert d3 == d2


def test_validate_config_data(cfg_setup):

    parameterFile = io.getTestDataFile('test_parameters.yml')
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
