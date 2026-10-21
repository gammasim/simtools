import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import simtools.applications.plot_production_grid as app
from simtools.applications.plot_production_grid import APPLICATION


def test_parser_exposes_shower_statistics_azimuth_options():
    actions = {action.dest: action for action in APPLICATION.build_parser()._actions}

    assert math.isclose(actions["azimuth_center"].default, 180.0)
    assert math.isclose(actions["azimuth_width"].default, 1.0)


def test_main_writes_plots_to_output_path(mocker):
    output_path = Path("plots")
    io_handler = Mock()
    io_handler.get_output_directory.return_value = output_path
    application = mocker.patch("simtools.applications.plot_production_grid.APPLICATION")
    application.start.return_value = SimpleNamespace(
        args={
            "grid_points_file": "grid.ecsv",
            "azimuth_center": 180.0,
            "azimuth_width": 1.0,
        },
        io_handler=io_handler,
    )
    plotter = mocker.patch("simtools.applications.plot_production_grid.ProductionGridPlotter")

    app.main()

    plotter.assert_called_once_with(grid_points_file="grid.ecsv", output_path=output_path)
    plotter.return_value.plot_sky_projection.assert_called_once_with()
    plotter.return_value.plot_limit_projections.assert_called_once_with()
    plotter.return_value.plot_shower_statistics_vs_zenith.assert_called_once_with(
        azimuth_center=180.0,
        azimuth_width=1.0,
    )
