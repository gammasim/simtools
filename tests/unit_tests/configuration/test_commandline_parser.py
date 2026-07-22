import logging
from pathlib import Path

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose

import simtools.configuration.commandline_parser as parser
from simtools.configuration.commandline_parameters import PARAMETER_DEFINITIONS

logger = logging.getLogger()
SIMULATION_MODEL_STRING = "simulation model"


def test_ignore_existing_parameter_version_argument():
    commandline_parser = parser.CommandLineParser()
    commandline_parser.initialize_argument_group(
        "execution", ["all"], PARAMETER_DEFINITIONS["EXECUTION_ARGS"]
    )

    args = commandline_parser.parse_args(["--ignore_existing_parameter_version"])

    assert args.ignore_existing_parameter_version is True


def test_initialize_default_arguments():
    # default arguments
    _parser_1 = parser.CommandLineParser()
    _parser_1.initialize_default_arguments()
    job_groups = _parser_1._action_groups
    for group in job_groups:
        assert str(group.title) in [
            "positional arguments",
            "optional arguments",
            "options",
            "paths",
            "configuration",
            "execution",
            "user",
            "run time",
        ]

    _parser_2 = parser.CommandLineParser()
    _parser_2.initialize_default_arguments(output=True)
    job_groups = _parser_2._action_groups
    assert "output" in [str(group.title) for group in job_groups]


def test_initialize_default_arguments_accepts_activity_id():
    parser_with_defaults = parser.CommandLineParser()
    parser_with_defaults.initialize_default_arguments()

    args = parser_with_defaults.parse_args(["--activity_id", "my-test-activity-id"])

    assert args.activity_id == "my-test-activity-id"


def test_initialize_default_arguments_accepts_figure_format():
    parser_with_defaults = parser.CommandLineParser()
    parser_with_defaults.initialize_default_arguments()

    args = parser_with_defaults.parse_args(["--figure_format", "png", "pdf"])

    assert args.figure_format == ["png", "pdf"]


def test_initialize_default_arguments_accepts_log_file_path():
    parser_with_defaults = parser.CommandLineParser()
    parser_with_defaults.initialize_default_arguments()

    args = parser_with_defaults.parse_args(["--log_file_path", "./custom-logs"])

    assert args.log_file_path == Path("custom-logs")


def test_initialize_default_arguments_accepts_apptainer_image_dict(tmp_test_directory):
    parser_with_defaults = parser.CommandLineParser()
    parser_with_defaults.initialize_default_arguments()

    image_v7 = tmp_test_directory / "v7.sif"
    image_v63 = tmp_test_directory / "v63.sif"

    args = parser_with_defaults.parse_args(
        [
            "--apptainer_image",
            f"{{'7.0.0': '{image_v7}', '6.3.0': '{image_v63}'}}",
        ]
    )

    assert isinstance(args.apptainer_image, dict)
    assert args.apptainer_image == {"7.0.0": str(image_v7), "6.3.0": str(image_v63)}


def test_initialize_argument_group():
    app_parser = parser.CommandLineParser()
    app_parser.initialize_argument_group(
        "application",
        [
            "source_distance",
            "zenith_angle",
            "number_of_photons",
            "off_axis_angles",
            "all_model_versions",
            "data",
            "event_data_file",
            "telescope_ids",
        ],
        PARAMETER_DEFINITIONS["APPLICATION_ARGS"],
    )

    args = app_parser.parse_args(
        [
            "--source_distance",
            "1500 m",
            "--zenith_angle",
            "25 deg",
            "--number_of_photons",
            "1e6",
            "--off_axis_angles",
            "0.5",
            "1 deg",
            "--all_model_versions",
            "--data",
            "psf_data.ecsv",
            "--event_data_file",
            "events.h5",
            "--telescope_ids",
            "layout_ids.txt",
        ]
    )

    assert_quantity_allclose(args.source_distance, 1.5 * u.km)
    assert_quantity_allclose(args.zenith_angle, 25 * u.deg)
    assert args.number_of_photons == 1_000_000
    assert len(args.off_axis_angles) == 2
    assert_quantity_allclose(args.off_axis_angles[0], 0.5 * u.deg)
    assert_quantity_allclose(args.off_axis_angles[1], 1 * u.deg)
    assert args.all_model_versions is True
    assert args.data == "psf_data.ecsv"
    assert args.event_data_file == "events.h5"
    assert args.telescope_ids == "layout_ids.txt"

    job_groups = app_parser._action_groups
    assert "application" in [str(group.title) for group in job_groups]


def test_simulation_model():
    # simulation model is none
    _parser_n = parser.CommandLineParser()
    _parser_n.initialize_default_arguments(simulation_model=None)

    # Model version only, no site or telescope
    _parser_v = parser.CommandLineParser()
    _parser_v.initialize_default_arguments(simulation_model=["model_version"])
    job_groups = _parser_v._action_groups

    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert all(action.dest != "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # Site model can exist without a telescope model, model and parameter version
    _parser_s = parser.CommandLineParser()
    _parser_s.initialize_default_arguments(
        simulation_model=["site", "model_version", "parameter_version"]
    )
    job_groups = _parser_s._action_groups
    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert any(action.dest == "parameter_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # No telescope model without site model; parameter_version only
    _parser_t = parser.CommandLineParser()
    _parser_t.initialize_default_arguments(
        simulation_model=["telescope", "telescopes", "site", "parameter_version"]
    )
    job_groups = _parser_t._action_groups
    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "parameter_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert any(action.dest == "telescope" for action in group._group_actions)
            assert any(action.dest == "telescopes" for action in group._group_actions)


def test_db_configuration():
    _parser_6 = parser.CommandLineParser()
    _parser_6.initialize_default_arguments(db_config=True)
    job_groups = _parser_6._action_groups
    assert "database configuration" in [str(group.title) for group in job_groups]


def test_layout_parsers():
    _parser_7 = parser.CommandLineParser()
    _parser_7.initialize_default_arguments(simulation_model=["layout"])
    job_groups = _parser_7._action_groups
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)

    _parser_8 = parser.CommandLineParser()
    _parser_8.initialize_default_arguments(simulation_model=["layout", "layout_file"])
    job_groups = _parser_8._action_groups
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)
            assert any(action.dest == "array_layout_file" for action in group._group_actions)

    _parser_9 = parser.CommandLineParser()
    _parser_9.initialize_default_arguments(
        simulation_model=["layout", "layout_file", "plot_all_layouts", "layout_parameter_file"]
    )
    job_groups = _parser_9._action_groups
    parser_actions = _parser_9._actions
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)
            assert any(action.dest == "array_layout_file" for action in group._group_actions)
            assert any(action.dest == "plot_all_layouts" for action in group._group_actions)
    assert any(action.dest == "array_layout_parameter_file" for action in parser_actions)

    args = _parser_9.parse_args(
        [
            "--array_layout_name",
            "alpha",
            "--array_layout_parameter_file",
            "array_layouts.json",
        ]
    )
    assert args.array_layout_name == ["alpha"]
    assert args.array_layout_parameter_file == "array_layouts.json"


def test_simulation_configuration():
    _parser_9 = parser.CommandLineParser()
    _parser_9.initialize_default_arguments(
        simulation_configuration={
            "software": None,
            "corsika_configuration": ["all"],
            "sim_telarray_configuration": ["all"],
        }
    )
    job_groups = _parser_9._action_groups
    for group in job_groups:
        if str(group.title) == "simulation software":
            assert any(action.dest == "simulation_software" for action in group._group_actions)
        if str(group.title) == "simulation configuration":
            assert any(action.dest == "primary" for action in group._group_actions)
        if str(group.title) == "shower parameters":
            assert any(action.dest == "view_cone" for action in group._group_actions)
        if str(group.title) == "sim_telarray configuration":
            assert any(
                action.dest == "sim_telarray_instrument_seed" for action in group._group_actions
            )

    _parser_10 = parser.CommandLineParser()
    _parser_10.initialize_default_arguments(
        simulation_configuration={"software": None, "corsika_configuration": ["wrong_parameter"]}
    )


def test_simulation_configuration_uses_defaults_for_optional_arguments():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={
            "software": None,
            "corsika_configuration": ["primary", "azimuth_angle", "zenith_angle", "run_number"],
        }
    )

    args = test_parser.parse_args(["--primary", "gamma"])

    assert args.primary == "gamma"
    assert args.simulation_software == "corsika_sim_telarray"
    assert_quantity_allclose(args.azimuth_angle, 0 * u.deg)
    assert_quantity_allclose(args.zenith_angle, 20 * u.deg)
    assert args.run_number == 1


def test_simulation_configuration_accepts_grid_list_values():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={
            "software": None,
            "corsika_configuration": [
                "primary",
                "azimuth_angle",
                "zenith_angle",
                "corsika_le_interaction",
                "corsika_he_interaction",
            ],
        }
    )

    args = test_parser.parse_args(
        [
            "--primary",
            "gamma",
            "proton",
            "--azimuth_angle",
            "north",
            "south",
            "--zenith_angle",
            "20",
            "40",
            "--corsika_le_interaction",
            "urqmd",
            "fluka",
            "--corsika_he_interaction",
            "epos",
            "qgsjet",
        ]
    )

    assert args.primary == ["gamma", "proton"]
    assert_quantity_allclose(args.azimuth_angle[0], 0 * u.deg)
    assert_quantity_allclose(args.azimuth_angle[1], 180 * u.deg)
    assert_quantity_allclose(args.zenith_angle[0], 20 * u.deg)
    assert_quantity_allclose(args.zenith_angle[1], 40 * u.deg)
    assert args.corsika_le_interaction == ["urqmd", "fluka"]
    assert args.corsika_he_interaction == ["epos", "qgsjet"]


def test_simulation_configuration_parses_hadronic_transition_energy():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={"corsika_configuration": ["all"]}
    )

    args = test_parser.parse_args(
        ["--primary", "gamma", "--corsika_hadronic_transition_energy", "0.12 TeV"]
    )

    assert_quantity_allclose(args.corsika_hadronic_transition_energy, 120 * u.GeV)


def test_simulation_configuration_leaves_hadronic_transition_energy_unset():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={"corsika_configuration": ["all"]}
    )

    args = test_parser.parse_args(["--primary", "gamma"])

    assert args.corsika_hadronic_transition_energy is None


def test_simulation_configuration_accepts_energy_range_list_pair():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={
            "software": None,
            "corsika_configuration": ["primary", "energy_range"],
        }
    )

    args = test_parser.parse_args(
        [
            "--primary",
            "gamma",
            "--energy_range",
            "30 GeV",
            "300 GeV",
        ]
    )

    assert_quantity_allclose(args.energy_range[0], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[1], 300 * u.GeV)


def test_simulation_configuration_accepts_energy_range_list_of_pairs():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={
            "software": None,
            "corsika_configuration": ["primary", "energy_range"],
        }
    )

    args = test_parser.parse_args(
        [
            "--primary",
            "gamma",
            "--energy_range",
            "30 GeV 30 GeV",
            "300 GeV 300 GeV",
        ]
    )

    assert len(args.energy_range) == 2
    assert_quantity_allclose(args.energy_range[0][0], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[0][1], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[1][0], 300 * u.GeV)
    assert_quantity_allclose(args.energy_range[1][1], 300 * u.GeV)


def test_initialize_db_config_arguments_strip_string():
    parser_10 = parser.CommandLineParser()
    parser_10.initialize_argument_group(
        "database configuration", ["all"], PARAMETER_DEFINITIONS["DB_CONFIG_ARGS"]
    )
    for test_string in ["test", " test", "test ", " test "]:
        args = parser_10.parse_args(["--db_simulation_model", test_string])
        assert args.db_simulation_model == "test"


def _parser(*params):
    p = parser.CommandLineParser()
    p.initialize_argument_group(
        "application", list(params), PARAMETER_DEFINITIONS["APPLICATION_ARGS"]
    )
    return p


def test_max_offset_negative_fails():
    p = _parser("max_offset")
    with pytest.raises(SystemExit):
        p.parse_args(["--max_offset", "-0.1"])


def test_max_offset_zero_ok():
    p = _parser("max_offset")
    ns = p.parse_args(["--max_offset", "0"])
    assert isinstance(ns.max_offset, u.Quantity)
    assert ns.max_offset.to("deg").value == pytest.approx(0.0)


def test_offset_step_zero_fails():
    p = _parser("offset_step")
    with pytest.raises(SystemExit):
        p.parse_args(["--offset_step", "0"])


def test_offset_step_positive_ok():
    p = _parser("offset_step")
    ns = p.parse_args(["--offset_step", "0.25"])
    assert isinstance(ns.offset_step, u.Quantity)
    assert ns.offset_step.to("deg").value == pytest.approx(0.25)
