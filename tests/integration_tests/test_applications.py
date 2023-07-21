#!/usr/bin/python3
#

import logging
import os
from io import StringIO

import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# This module perform tests on the application by running them with a set
# of arguments. Each application to be tested corresponds to a key in
# APP_LIST, that contains a list of arguments to be tested, so that
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
            "--workflow_config",
            "tests/resources/set_MST_mirror_2f_measurements_from_external.config.yml",
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
            "--no_tuning",
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
    "validate_camera_efficiency::MST-NectarCam-D": [
        ["--site", "North", "--telescope", "MST-NectarCam-D", "--model_version", "prod5"]
    ],
    "validate_camera_efficiency::SST-D": [
        ["--site", "South", "--telescope", "SST-D", "--model_version", "prod5"]
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
    "plot_layout_array::one_file": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv",
        ]
    ],
    "plot_layout_array::one_file_with_name": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv",
            "--figure_name test",
        ]
    ],
    "plot_layout_array::one_file_with_name_and_extension": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv",
            "--figure_name test.png",
        ]
    ],
    "plot_layout_array::one_file_one_angle": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv",
            "--rotate_angle 20",
        ]
    ],
    "plot_layout_array::one_file_two_angles": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv",
            "--rotate_angle 20 30",
        ]
    ],
    "plot_layout_array::two_files": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv "
            "data/layout/telescope_positions-South-TestLayout.ecsv",
        ]
    ],
    "plot_layout_array::two_files_one_angle": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv "
            "data/layout/telescope_positions-South-TestLayout.ecsv",
            "--rotate_angle 20",
        ]
    ],
    "plot_layout_array::two_files_two_angles": [
        [
            "--telescope_list data/layout/telescope_positions-North-TestLayout.ecsv "
            "data/layout/telescope_positions-South-TestLayout.ecsv",
            "--rotate_angle 20 30",
        ]
    ],
    "plot_layout_array::layout_name": [
        [
            "--layout_array_name North-4LST",
        ]
    ],
    "plot_layout_array::layout_name_one_angle": [
        [
            "--layout_array_name North-4LST",
            "--rotate_angle 20",
        ]
    ],
    "plot_layout_array::layout_name_two_angles": [
        [
            "--layout_array_name North-4LST",
            "--rotate_angle 20 30",
        ]
    ],
    "plot_layout_array::layout_two_names_one_angle": [
        [
            "--layout_array_name North-4LST South-4LST",
            "--rotate_angle 20",
        ]
    ],
    "plot_layout_array::layout_two_names_two_angles": [
        [
            "--layout_array_name North-4LST South-4LST",
            "--rotate_angle 20 30",
        ]
    ],

    # CORSIKA photons
    "generate_corsika_histograms::figures":
        [["--IACT_file",
          "/workdir/external/simtools/tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South."
          "corsikaio", "--png"]],
    "generate_corsika_histograms::tables":
        [["--IACT_file",
          "/workdir/external/simtools/tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South."
          "corsikaio", "--ecsv"]],
    "generate_corsika_histograms::figures_and_tables":
        [["--IACT_file", "/workdir/external/simtools/tests/resources/"
                         "tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio", "--png", "--ecsv"]],
    "generate_corsika_histograms::output":
        [["--IACT_file", "/workdir/external/simtools/tests/resources/"
                         "tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio",
          "--png", "--ecsv", "--output_directory", "."]],
    "generate_corsika_histograms::1D_and_2D_run_header":
        [["--IACT_file",
          "/workdir/external/simtools/tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South."
          "corsikaio",
          "--png", "--ecsv",
          "--event_2D_histograms", "zenith",
          "azimuth", "--event_1D_histograms",
          "total_energy"]],
    "generate_corsika_histograms::1D_run_header":
        [["--IACT_file",
          "/workdir/external/simtools/tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South."
          "corsikaio",
          "--png", "--ecsv", "--event_1D_histograms",
          "total_energy"]],
    "generate_corsika_histograms::2D_run_header":
        [["--IACT_file",
          "/workdir/external/simtools/tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South."
          "corsikaio",
          "--png", "--ecsv",
          "--event_2D_histograms", "zenith",
          "azimuth"]],
    "generate_corsika_histograms::individual_telescopes_and_indices":
        [["--IACT_file",
          "/workdir/external/simtools/tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South."
          "corsikaio",
          "--png", "--ecsv",
          "--telescope_indices",
          "0 1 2",
          "--individual_telescopes"]],

    "generate_corsika_histograms::telescope_indices":
        [["--IACT_file",
          "/workdir/external/simtools/tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South."
          "corsikaio",
          "--png", "--ecsv", "--telescope_indices", "0 1 2"]],

    # Production
    "produce_array_config": [["--array_config", "./tests/resources/array_config_test.yml"]],
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
    "get_file_from_db::CTA-Simulation-Model": [["--file_name", "mirror_CTA-S-LST_v2020-04-07.dat"]],
    "get_file_from_db::test-data": [["--file_name", "PSFcurve_data_v2.txt"]],
    "get_file_from_db::CTA-Simulation-Model-Derived-Values": [
        ["--file_name", "ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv"]
    ],
    "get_parameter::telescope_parameter": [
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
    "get_parameter::site_parameter": [
        [
            "--site",
            "North",
            "--parameter",
            "altitude",
        ]
    ],
    "add_file_to_db": [
        [
            "--file_name",
            "TESTMODELDIR/MLTdata-preproduction.usermeta.yml",
            "TESTMODELDIR/MLTdata-preproduction.ecsv",
            "--db",
            "sandbox",
        ]
    ],
    # Production
    "production::showers_only": [
        [
            "--productionconfig",
            "./tests/resources/prod_config_test.yml",
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
            "./tests/resources/prod_config_test.yml",
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
    "print_array_elements::print_compact_corsika_telescopeheights": [
        [
            "--array_element_list",
            "tests/resources/telescope_positions-North-utm.ecsv",
            "--export",
            "corsika",
            "--use_corsika_telescope_height",
        ],
    ],
}


@pytest.mark.parametrize("application", APP_LIST.keys())
def test_applications(application, io_handler, monkeypatch, db):

    logger.info(f"Testing {application}")

    # The add_file_to_db.py application requires a user confirmation.
    # With this line we mock the user confirmation to be y for the test
    # Notice this is done for all tests, so keep in mind if in the future we add tests with input.
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))

    def prepare_one_file(file_name):
        db.export_file_db(
            db_name="test-data",
            dest=io_handler.get_output_directory(dir_type="model", test=True),
            file_name=file_name,
        )

    prepare_one_file("PSFcurve_data_v2.txt")
    prepare_one_file("MLTdata-preproduction.ecsv")

    def make_command(app, args):
        cmd = "python simtools/applications/" + app + ".py"
        for aa in args:
            aa = aa.replace(
                "TESTMODELDIR", str(io_handler.get_output_directory(dir_type="model", test=True))
            )
            cmd += " " + aa
        return cmd

    for args in APP_LIST[application]:
        app_name = application.partition("::")[0]
        logger.info(f"Running with args: {args}")
        cmd = make_command(app_name, args)
        logger.info(f"Running command: {cmd}")
        out = os.system(cmd)
        is_output_valid = out == 0
        assert is_output_valid
