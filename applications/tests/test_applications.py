#!/usr/bin/python3

import os
import pytest
import logging

from simtools.util.tests import (
    has_db_connection,
    simtel_installed,
    DB_CONNECTION_MSG,
    SIMTEL_MSG,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

""" This module perform tests on the application by running them with a set
of arguments. Each applications to be tested correspond to an key in
APP_LIST, that contains a list of list of arguments to be tested, so that
the same application can be tested with a number of different set of arguments.
"""

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
            "--model_version",
            "prod4",
            "--mean_d80",
            "1.4",
            "--sig_d80",
            "0.16",
            "--mirror_list",
            "mirror_MST_focal_lengths.dat",
            "--d80_list",
            "mirror_MST_D80.dat",
            "--rnda",
            "0.0075",
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


@pytest.mark.parametrize("application", APP_LIST.keys())
def test_applications(application):
    logger.info("Testing {}".format(application))

    # Checking for DB connection
    if application in REQUIRE_DB_CONNECTION and not has_db_connection():
        pytest.skip(DB_CONNECTION_MSG)

    # Checking for sim_telarray installation
    if application in REQUIRE_SIMTEL and not simtel_installed():
        pytest.skip(SIMTEL_MSG)

    def makeCommand(app, args):
        cmd = "python applications/" + app + ".py"
        for aa in args:
            cmd += " " + aa
        return cmd

    for args in APP_LIST[application]:
        logger.info("Running with args: {}".format(args))
        cmd = makeCommand(application, args)
        out = os.system(cmd)
        isOutputValid = out == 0
        assert isOutputValid
