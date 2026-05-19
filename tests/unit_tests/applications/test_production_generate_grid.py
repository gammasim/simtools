from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import simtools.applications.production_generate_grid as app


@patch("simtools.applications.production_generate_grid.build_production_grid_engine")
@patch("simtools.applications.production_generate_grid.build_application")
def test_main_uses_shared_grid_builder(mock_build_application, mock_build_production_grid_engine):
    io_handler = Mock()
    io_handler.get_output_file.return_value = Path("grid_output.ecsv")
    args = {
        "axes": "grid.yml",
        "output_file": "grid_output.ecsv",
    }
    mock_build_application.return_value = SimpleNamespace(args=args, io_handler=io_handler)
    mock_grid_engine = Mock()
    mock_grid_engine.generate_grid.return_value = [{"ra": 1}]
    mock_build_production_grid_engine.return_value = mock_grid_engine

    app.main()

    mock_build_production_grid_engine.assert_called_once_with(args)
    mock_grid_engine.serialize_grid_points.assert_called_once_with(
        [{"ra": 1}],
        output_file=Path("grid_output.ecsv"),
    )
