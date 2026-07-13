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
        application_argument_definitions=app._APPLICATION_ARG_DEFINITIONS,
    )


@patch("simtools.applications.production_generate_grid.serialize_job_grid")
@patch("simtools.applications.production_generate_grid.build_job_grid_metadata")
@patch("simtools.applications.production_generate_grid.build_simulation_jobs")
@patch("simtools.applications.production_generate_grid.build_application")
def test_main_serializes_job_grid(
    mock_build_application,
    mock_build_simulation_jobs,
    mock_build_job_grid_metadata,
    mock_serialize_job_grid,
):
    io_handler = Mock()
    io_handler.get_output_file.return_value = Path("job_grid.ecsv")
    args = {
        "output_file": "job_grid.ecsv",
        "run_number_offset": 10,
    }
    mock_build_application.return_value = SimpleNamespace(args=args, io_handler=io_handler)
    mock_build_simulation_jobs.return_value = [{"primary": "gamma"}]
    mock_build_job_grid_metadata.return_value = {"site": "North"}

    app.main()

    mock_build_simulation_jobs.assert_called_once_with(args)
    mock_build_job_grid_metadata.assert_called_once_with(args)
    mock_serialize_job_grid.assert_called_once_with(
        job_rows=[{"primary": "gamma", "run_number": 11}],
        output_file=Path("job_grid.ecsv"),
        metadata={"site": "North"},
    )


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
