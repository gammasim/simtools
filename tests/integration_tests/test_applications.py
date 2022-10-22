#!/usr/bin/python3
#

import logging
import os

import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# This module perform tests on the application by running them with a set
# of arguments. Each applications to be tested correspond to an key in
# APP_LIST, that contains a list of list of arguments to be tested, so that
# the same application can be tested with a number of different set of arguments.


APP_LIST = {
    # Optics
    "compare_cumulative_psf": [
        [
            "--site",
            "North",
            "--telescope",
            "LST-1",
            "--model_version",
            "prod5",
            "--data",
            "TESTMODELDIR/PSFcurve_data_v2.txt",
            "--zenith",
            "20",
            "--test",
        ]
    ],
    "submit_data_from_external::help": [
        [
            "--help",
        ]
    ],
    "submit_data_from_external::submit": [
        [
            "--workflow_config_file",
            "tests/resources/set_MST_mirror_2f_measurements_from_external.config.yml",
            "--input_meta_file",
            "TESTMODELDIR/MLTdata-preproduction.usermeta.yml",
            "--input_data_file",
            "TESTMODELDIR/MLTdata-preproduction.ecsv",
            " --test",
        ]
    ],
    "derive_mirror_rnda::help": [
        [
            "--help",
        ]
    ],
    "derive_mirror_rnda::psf_random_flen": [
        [
            "--site",
            "North",
            "--telescope",
            "MST-FlashCam-D",
            "--containment_fraction",
            "0.8",
            "--psf_measurement_containment_mean",
            "1.4",
            "--use_random_flen",
            "--rnda",
            "0.0063",
            " --test",
        ]
    ],
    "derive_mirror_rnda::psf_notuning": [
        [
            "--site",
            "North",
            "--telescope",
            "MST-FlashCam-D",
            "--containment_fraction",
            "0.8",
            "--mirror_list",
            "TESTMODELDIR/MLTdata-preproduction.ecsv",
            "--psf_measurement",
            "TESTMODELDIR/MLTdata-preproduction.ecsv",
            "--rnda",
            "0.0063",
            " --test",
        ]
    ],
    "derive_mirror_rnda::psf_measurement": [
        [
            "--site",
            "North",
            "--telescope",
            "MST-FlashCam-D",
            "--containment_fraction",
            "0.8",
            "--mirror_list",
            "TESTMODELDIR/MLTdata-preproduction.ecsv",
            "--psf_measurement",
            "TESTMODELDIR/MLTdata-preproduction.ecsv",
            "--rnda",
            "0.0063",
            " --test",
        ]
    ],
    "derive_mirror_rnda::psf_mean": [
        [
            "--site",
            "North",
            "--telescope",
            "MST-FlashCam-D",
            "--containment_fraction",
            "0.8",
            "--mirror_list",
            "TESTMODELDIR/MLTdata-preproduction.ecsv",
            "--psf_measurement_containment_mean",
            "1.4",
            "--rnda",
            "0.0063",
            " --test",
        ]
    ],
    "validate_optics": [
        [
            "--site",
            "North",
            "--telescope",
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
    "tune_psf": [
        [
            "--site",
            "North",
            "--telescope",
            "LST-1",
            "--model_version",
            "prod5",
            "--data",
            "TESTMODELDIR/PSFcurve_data_v2.txt",
            "--zenith",
            "20",
            "--test",
        ]
    ],
    # Camera
    "validate_camera_efficiency": [
        ["--site", "North", "--telescope", "MST-NectarCam-D", "--model_version", "prod5"]
    ],
    "validate_camera_fov": [
        ["--site", "North", "--telescope", "MST-NectarCam-D", "--model_version", "prod5"]
    ],
    "plot_simtel_histograms::help": [
        [
            "--help",
        ]
    ],
    # Layout
    "make_regular_arrays": [[]],
    # Production
    "produce_array_config": [["--array_config", "./tests/resources/arrayConfigTest.yml"]],
    # Trigger
    "sim_showers_for_trigger_rates": [
        [
            "--array",
            "4LST",
            "--site",
            "North",
            "--primary",
            "proton",
            "--nruns",
            "2",
            "--nevents",
            "10000",
            "--test",
            "--submit_command",
            "local",
        ]
    ],
    # Database
    "get_parameter": [
        [
            "--site",
            "North",
            "--telescope",
            "LST-1",
            "--parameter",
            "mirror_list",
            "--model_version",
            "prod5",
        ]
    ],
    # Production
    "production::showers_only": [
        [
            "--productionconfig",
            "./tests/resources/prodConfigTest.yml",
            "--task",
            "simulate",
            "--showers_only",
            "--test",
            "--submit_command",
            "local",
        ]
    ],
    "production::array_only": [
        [
            "--productionconfig",
            "./tests/resources/prodConfigTest.yml",
            "--task",
            "simulate",
            "--array_only",
            "--test",
            "--submit_command",
            "local",
        ]
    ],
    # print_array
    "print_array_elements::print_all": [
        ["--array_element_list", "tests/resources/telescope_positions-South-4MST.ecsv"],
    ],
    "print_array_elements::print_compact": [
        [
            "--array_element_list",
            "tests/resources/telescope_positions-South-4MST.ecsv",
            "--compact",
            "corsika",
        ],
    ],
    "print_array_elements::export_utm": [
        [
            "--array_element_list",
            "tests/resources/telescope_positions-South-4MST.ecsv",
            "--export",
            "utm",
        ],
    ],
    "print_array_elements::export_corsika": [
        [
            "--array_element_list",
            "tests/resources/telescope_positions-South-4MST.ecsv",
            "--export",
            "corsika",
            "--use_corsika_telescope_height",
        ],
    ],
    # files without corsika_spheres definition
    "print_array_elements::print_compact_nocors_utm": [
        [
            "--array_element_list",
            "tests/resources/telescope_positions-North-utm.ecsv",
            "--compact",
            "utm",
        ],
    ],
    "print_array_elements::print_compact_nocors_corsika": [
        [
            "--array_element_list",
            "tests/resources/telescope_positions-North-utm.ecsv",
            "--compact",
            "corsika",
        ],
    ],
}


@pytest.mark.parametrize("application", APP_LIST.keys())
def test_applications(application, io_handler, db, simtelpath_no_mock):
    logger.info("Testing {}".format(application))

    def prepare_one_file(fileName):
        db.exportFileDB(
            dbName="test-data",
            dest=io_handler.getOutputDirectory(dirType="model", test=True),
            fileName=fileName,
        )

    prepare_one_file("PSFcurve_data_v2.txt")
    prepare_one_file("MLTdata-preproduction.usermeta.yml")
    prepare_one_file("MLTdata-preproduction.ecsv")

    def makeCommand(app, args):
        cmd = "python applications/" + app + ".py"
        for aa in args:
            aa = aa.replace(
                "TESTMODELDIR", str(io_handler.getOutputDirectory(dirType="model", test=True))
            )
            cmd += " " + aa
        return cmd

    for args in APP_LIST[application]:
        app_name = application.partition("::")[0]
        logger.info("Running with args: {}".format(args))
        cmd = makeCommand(app_name, args)
        logger.info("Running command: {}".format(cmd))
        out = os.system(cmd)
        isOutputValid = out == 0
        assert isOutputValid
