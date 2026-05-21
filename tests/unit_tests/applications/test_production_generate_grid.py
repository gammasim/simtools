from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose

import simtools.applications.production_generate_grid as app
from simtools.configuration.commandline_parser import CommandLineParser


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
    }
    mock_build_application.return_value = SimpleNamespace(args=args, io_handler=io_handler)
    mock_build_simulation_jobs.return_value = [{"primary": "gamma"}]
    mock_build_job_grid_metadata.return_value = {"site": "North"}

    app.main()

    mock_build_simulation_jobs.assert_called_once_with(args)
    mock_build_job_grid_metadata.assert_called_once_with(args)
    mock_serialize_job_grid.assert_called_once_with(
        job_rows=[{"primary": "gamma"}],
        output_file=Path("job_grid.ecsv"),
        metadata={"site": "North"},
    )


def test_add_arguments_accepts_spaced_quantity_ranges():
    parser = CommandLineParser()
    app._add_arguments(parser)

    args = parser.parse_args(
        [
            "--azimuth_range",
            "310",
            "deg",
            "20",
            "deg",
            "--azimuth_binning",
            "3",
            "--azimuth_scaling",
            "linear",
        ]
    )

    assert_quantity_allclose(args.azimuth_range[0], 310 * u.deg)
    assert_quantity_allclose(args.azimuth_range[1], 20 * u.deg)
    assert args.azimuth_binning == 3
    assert args.azimuth_scaling == "linear"
