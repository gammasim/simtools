from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import simtools.applications.production_generate_grid as app
from simtools.application_control import build_application_parser


def _parser():
    return build_application_parser(
        application_path=app.__file__,
        description=app.__doc__,
        add_arguments_function=app._add_arguments,
    )


def _full_parser():
    return build_application_parser(
        application_path=app.__file__,
        description=app.__doc__,
        add_arguments_function=app._add_arguments,
        initialization_kwargs=app._INITIALIZATION_KWARGS,
    )


@patch("simtools.applications.production_generate_grid.generate_job_grid")
@patch("simtools.applications.production_generate_grid.build_application")
def test_main_generates_job_grid(mock_build_application, mock_generate_job_grid):
    io_handler = Mock()
    io_handler.get_output_file.return_value = Path("job_grid.ecsv")
    args = {
        "output_file": "job_grid.ecsv",
        "run_number_offset": 10,
    }
    mock_build_application.return_value = SimpleNamespace(args=args, io_handler=io_handler)
    app.main()

    mock_generate_job_grid.assert_called_once_with(args, Path("job_grid.ecsv"))
    mock_build_application.assert_called_once_with(
        add_arguments_function=app._add_arguments,
        initialization_kwargs=app._INITIALIZATION_KWARGS,
        startup_kwargs={"resolve_sim_software_executables": False},
    )


def test_full_parser_contains_only_relevant_shared_arguments():
    parser = _full_parser()
    actions = {action.dest: action for action in parser._actions}

    expected = {
        "array_layout_name",
        "azimuth_angle",
        "core_scatter",
        "energy_range",
        "model_version",
        "output_file",
        "output_path",
        "primary",
        "run_number_offset",
        "showers_per_run",
        "site",
        "view_cone",
        "zenith_angle",
    }
    assert expected <= set(actions)
    assert actions["output_file"].default == "job_grid.ecsv"
    assert actions["output_file"].help == "Output ECSV production job grid."

    irrelevant = {
        "array_element_list",
        "correct_for_b_field_alignment",
        "curved_atmosphere_min_zenith_angle",
        "data_path",
        "eslope",
        "event_number_first_shower",
        "figure_format",
        "model_path",
        "overwrite_model_parameters",
        "primary_id_type",
        "run_number",
        "telescope",
        "user_name",
    }
    assert irrelevant.isdisjoint(actions)


def test_full_parser_accepts_minimum_direct_configuration():
    args = _full_parser().parse_args(
        [
            "--model_version",
            "7.0.0",
            "--site",
            "North",
            "--array_layout_name",
            "LSTN-01",
            "--primary",
            "gamma",
            "--showers_per_run",
            "1000",
        ]
    )

    assert args.model_version == ["7.0.0"]
    assert args.site == "North"
    assert args.array_layout_name == ["LSTN-01"]
    assert args.output_file == "job_grid.ecsv"


def test_add_arguments_accepts_compact_axis_definitions():
    parser = _parser()

    args = parser.parse_args(
        [
            "--axis",
            "azimuth",
            "310",
            "deg",
            "20",
            "deg",
            "3",
            "linear",
            "--axis",
            "offset",
            "0",
            "deg",
            "10",
            "deg",
            "2",
        ]
    )

    assert args.axis == [
        ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
        ["offset", "0", "deg", "10", "deg", "2"],
    ]


def test_add_arguments_accepts_zenith_angle_scaling_factor():
    parser = _parser()

    args = parser.parse_args(["--zenith_angle_scaling_factor", "2.5"])

    assert args.zenith_angle_scaling_factor == pytest.approx(2.5)


def test_add_arguments_rejects_both_shower_count_modes():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--number_of_runs", "2", "--total_showers", "1000"])


def test_add_arguments_accepts_max_total_showers_rounding_warnings():
    parser = _parser()

    args = parser.parse_args(["--max_total_showers_rounding_warnings", "7"])

    assert args.max_total_showers_rounding_warnings == 7


def test_add_arguments_accepts_direction_grid_density():
    parser = _parser()

    args = parser.parse_args(["--direction_grid_density", "1.5"])

    assert args.direction_grid_density == ["1.5"]


def test_add_arguments_accepts_direction_grid_density_with_unit():
    parser = _parser()

    args = parser.parse_args(["--direction_grid_density", "0.25", "1/deg^2"])

    assert args.direction_grid_density == ["0.25", "1/deg^2"]


def test_add_arguments_accepts_showers_per_run_scaling():
    parser = _parser()

    args = parser.parse_args(["--showers_per_run_scaling", "cosine_zenith"])

    assert args.showers_per_run_scaling == "cosine_zenith"


def test_add_arguments_accepts_energy_max_scaling():
    parser = _parser()

    args = parser.parse_args(["--energy_max_scaling", "-2.5", "300", "TeV"])

    assert args.energy_max_scaling == ["-2.5", "300", "TeV"]
