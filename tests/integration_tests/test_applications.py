#!/usr/bin/python3

import os
import pytest
import logging

import simtools.config as cfg

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# This module perform tests on the application by running them with a set
# of arguments. Each applications to be tested correspond to an key in
# APP_LIST, that contains a list of list of arguments to be tested, so that
# the same application can be tested with a number of different set of arguments.

APP_LIST = {
    # Optics
    "tune_psf": [
        [
            "-s",
            "North",
            "-t",
            "LST-1",
            "--model_version",
            "prod4",
            "--data",
            "PSFcurve_data_v2.txt",
            "--zenith",
            "20",
            "--test",
        ]
    ],
    "compare_cumulative_psf": [
        [
            "-s",
            "North",
            "-t",
            "LST-1",
            "--model_version",
            "prod4",
            "--data",
            "PSFcurve_data_v2.txt",
            "--zenith",
            "20",
            "--test",
        ]
    ],
    "derive_mirror_rnda": [
        [
            "-s",
            "North",
            "-t",
            "MST-FlashCam-D",
            "--containment_fraction",
            "0.8",
            "--mirror_list",
            "tests/resources/MLTdata-preproduction.ecsv",
            "--psf_measurement",
            "tests/resources/MLTdata-preproduction.ecsv",
            "--rnda",
            "0.0063",
            " --test",
        ]
    ],
    "validate_optics": [
        [
            "-s",
            "North",
            "-t",
            "LST-1",
            "--max_offset",
            "1.0",
            "--src_distance",
            "11",
            "--zenith",
            "20",
            "--test",
        ]
    ],
    # Camera
    "validate_camera_efficiency": [
        ["-s", "North", "-t", "MST-NectarCam-D", "--model_version", "prod4"]
    ],
    "validate_camera_fov": [
        ["-s", "North", "-t", "MST-NectarCam-D", "--model_version", "prod4"]
    ],
    # Layout
    "make_regular_arrays": [[]],
    # Production
    "produce_array_config": [
        ["--array_config", "data/test-data/arrayConfigTest.yml"]
    ],
    # Trigger
    "sim_showers_for_trigger_rates": [
        [
            "-a",
            "4LST",
            "-s",
            "North",
            "--primary",
            "proton",
            "--nruns",
            "2",
            "--nevents",
            "10000",
            "--test",
        ]
    ],
    # Database
    "get_parameter": [
        ["-s", "North", "-t", "LST-1", "-p", "mirror_list", "--model_version", "prod4"]
    ],
    "production": [
        ["-c", "data/test-data/prodConfigTest.yml", "-t", "simulate", "--test"]
    ],
}

# List of applications that require sim_telarray installation
REQUIRE_SIMTEL = (
    "compare_cumulative_psf",
    "tune_psf",
    "derive_mirror_rnda",
    "validate_optics",
    "validate_camera_efficiency",
)

REQUIRE_DB_CONNECTION = (
    "compare_cumulative_psf",
    "tune_psf",
    "derive_mirror_rnda",
    "validate_optics",
    "validate_camera_efficiency",
    "validate_camera_fov",
    "make_regular_arrays",
    "produce_array_config",
    "get_parameter",
    "production",
)

REQUIRE_CFG_FILE = (
    "derive_mirror_rnda",
    "production",
)


@pytest.mark.parametrize("application", APP_LIST.keys())
def test_applications(set_simtools, application):
    logger.info("Testing {}".format(application))

    def makeCommand(app, args):
        cmd = "python applications/" + app + ".py"
        for aa in args:
            cmd += " " + aa
        cmd += " --configFile " + cfg.CONFIG_FILE_NAME
        return cmd

    for args in APP_LIST[application]:
        # TODO: skip all but derive_mirror_rando
        # (no configFile implemented for others)
        if application != 'derive_mirror_rnda':
            continue
        logger.info("Running with args: {}".format(args))
        cmd = makeCommand(application, args)
        out = os.system(cmd)
        isOutputValid = out == 0
        assert isOutputValid
