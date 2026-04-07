from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

from simtools.production_configuration.derive_production_statistics_handler import (
    ProductionStatisticsHandler,
)

# Define constants for frequently used literals
BASE_PATH = "tests/resources/production_dl2_fits/"
FILE_NAME_TEMPLATE = "prod6_LaPalma-{zenith}deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"
OUTPUT_FILE = "output.ecsv"


@pytest.fixture
def grid_points_content():
    """Fixture to provide sample grid points content."""
    return [
        {"azimuth": 0.0, "zenith_angle": 20.0, "nsb_level": 0.005, "offset": 0.5},
        {"azimuth": 0.0, "zenith_angle": 40.0, "nsb_level": 0.005, "offset": 0.5},
    ]


@pytest.fixture
def grid_points_file(tmp_path, grid_points_content):
    """Fixture to create a temporary grid points file."""
    grid_points_file_path = tmp_path / "grid_points.ecsv"
    table = Table(rows=grid_points_content)
    table["azimuth"].unit = "deg"
    table["zenith_angle"].unit = "deg"
    table["nsb_level"].unit = "MHz"
    table["offset"].unit = "deg"
    table.write(grid_points_file_path, format="ascii.ecsv", overwrite=True)
    return grid_points_file_path


@pytest.fixture
def metrics_file():
    """Fixture to return a metrics file."""
    return Path("tests/resources/production_simulation_config_metrics.yml")


@pytest.fixture
def args_dict(tmp_path, metrics_file, grid_points_file):
    """Fixture to provide a mock args_dict for testing."""
    return {
        "base_path": BASE_PATH,
        "zeniths": [20, 40],
        "azimuths": [180],
        "nsb": [0.005],
        "offsets": [0.5, 1.0],
        "query_point": [1.0, 180.0, 20.0, 4.0, 0.5],
        "output_file": "production_statistics.ecsv",
        "output_path": str(tmp_path),
        "metrics_file": str(metrics_file),
        "grid_points_production_file": str(grid_points_file),
        "file_name_template": FILE_NAME_TEMPLATE,
    }


@pytest.fixture
def mock_handler(args_dict, tmp_path):
    """Fixture to provide a mocked ProductionStatisticsHandler."""
    with patch(
        "simtools.production_configuration.derive_production_statistics_handler."
        "ProductionStatisticsHandler._load_grid_points_production",
        return_value=[],
    ):
        return ProductionStatisticsHandler(args_dict, output_path=tmp_path)


def test_init_with_required_arguments(args_dict, tmp_path):
    """Test initialization with required arguments."""
    handler = ProductionStatisticsHandler(args_dict, output_path=tmp_path)
    assert handler.args == args_dict
    assert handler.output_path == tmp_path
    assert isinstance(handler.grid_points_production, Table)
    assert len(handler.grid_points_production) == 2


def test_no_evaluators_initialized(mock_handler):
    """Test behavior when no evaluators are initialized."""
    mock_handler.args["zeniths"] = []  # No zeniths provided

    mock_handler.initialize_evaluators()

    assert len(mock_handler.evaluator_instances) == 0


def test_missing_required_args(args_dict, tmp_path):
    """Test behavior when required arguments are missing."""
    # Create a copy with missing parameters
    args_missing = args_dict.copy()
    args_missing.pop("zeniths", None)

    with patch(
        "simtools.production_configuration.derive_production_statistics_handler."
        "ProductionStatisticsHandler._load_grid_points_production",
        return_value=[],
    ):
        handler = ProductionStatisticsHandler(args_missing, output_path=tmp_path)
        with pytest.raises(KeyError):
            handler.initialize_evaluators()


def test_missing_event_files(mock_handler):
    """Test behavior when event files are missing."""
    mock_handler.args["file_name_template"] = "non_existent_file.fits"

    with patch("pathlib.Path.exists", return_value=False):
        mock_handler.initialize_evaluators()

    assert len(mock_handler.evaluator_instances) == 0


def test_no_base_path(mock_handler):
    """Test behavior when base_path is not provided."""
    mock_handler.args["base_path"] = None

    mock_handler.initialize_evaluators()

    assert len(mock_handler.evaluator_instances) == 0


def test_empty_offsets(mock_handler):
    """Test behavior when offsets are empty."""
    mock_handler.args["offsets"] = []  # Empty offsets

    mock_handler.initialize_evaluators()

    assert len(mock_handler.evaluator_instances) == 0


def test_perform_interpolation_not_initialized(mock_handler):
    """Test perform_interpolation when no evaluators are initialized."""
    mock_handler.logger = MagicMock()

    mock_handler.evaluator_instances = []

    result = mock_handler.perform_interpolation()

    mock_handler.logger.error.assert_called_once_with(
        "No evaluators initialized. Cannot perform interpolation."
    )
    assert result is None


def test_perform_interpolation_with_evaluators(mock_handler):
    """Test perform_interpolation with valid evaluators."""
    mock_handler.evaluator_instances = [MagicMock(), MagicMock()]
    mock_handler.grid_points_production = Table(rows=[{"test": "value1"}, {"test": "value2"}])

    mock_interp_handler = MagicMock()
    mock_interp_handler.interpolate.return_value = np.array([100, 200])
    with patch(
        "simtools.production_configuration.derive_production_statistics_handler.InterpolationHandler",
        return_value=mock_interp_handler,
    ):
        result = mock_handler.perform_interpolation()

    assert isinstance(result, Table)
    assert len(result) == 2
    assert result["interpolated_production_statistics"][0] == 100
    assert result["interpolated_production_statistics"][1] == 200


def test_write_output(mock_handler):
    """Test the write_output method."""
    production_statistics = Table(
        rows=[{"azimuth": 180.0, "zenith_angle": 20.0, "interpolated_production_statistics": 10000}]
    )
    production_statistics["azimuth"].unit = "deg"
    production_statistics["zenith_angle"].unit = "deg"

    mock_handler.write_output(production_statistics)

    expected_path = Path(f"{mock_handler.output_path}/{mock_handler.args['output_file']}")
    assert expected_path.exists()
    written_table = Table.read(expected_path, format="ascii.ecsv")
    assert "interpolated_production_statistics" in written_table.colnames
    assert written_table["interpolated_production_statistics"][0] == 10000


def test_run(mock_handler):
    """Test the run method."""
    mock_handler.initialize_evaluators = MagicMock()
    mock_handler.perform_interpolation = MagicMock(return_value=Table(rows=[{"test": "value"}]))
    mock_handler.write_output = MagicMock()
    mock_handler.logger = MagicMock()

    mock_handler.run()

    mock_handler.initialize_evaluators.assert_called_once()
    mock_handler.perform_interpolation.assert_called_once()
    mock_handler.write_output.assert_called_once()


def test_plot_production_statistics_comparison(mock_handler):
    """Test the plot_production_statistics_comparison method."""
    mock_handler.interpolation_handler = MagicMock()
    mock_axes = MagicMock()
    mock_figure = MagicMock()
    mock_handler.interpolation_handler.plot_comparison.return_value = mock_axes
    mock_axes.figure = mock_figure

    mock_handler.plot_production_statistics_comparison()

    plot_path = mock_handler.output_path.joinpath("production_statistics_comparison.png")
    mock_figure.savefig.assert_called_once_with(plot_path)


def test_run_with_plot_production_statistics(mock_handler):
    """Test the run method when plot_production_statistics is enabled."""
    mock_handler.args["plot_production_statistics"] = True
    mock_handler.initialize_evaluators = MagicMock()
    mock_handler.perform_interpolation = MagicMock(return_value=Table(rows=[{"test": "value"}]))
    mock_handler.write_output = MagicMock()
    mock_handler.plot_production_statistics_comparison = MagicMock()

    mock_handler.run()

    mock_handler.initialize_evaluators.assert_called_once()
    mock_handler.perform_interpolation.assert_called_once()
    mock_handler.plot_production_statistics_comparison.assert_called_once()
    mock_handler.write_output.assert_called_once()


def test_handler_with_grid_points_from_file(grid_points_file, metrics_file, tmp_path):
    """Test handler initialization with grid points from an ECSV file."""
    args_dict = {
        "base_path": BASE_PATH,
        "zeniths": [20, 40],
        "offsets": [0.5],
        "azimuths": [0],
        "nsb": [0.005],
        "grid_points_production_file": str(grid_points_file),
        "metrics_file": str(metrics_file),
        "output_file": OUTPUT_FILE,
        "file_name_template": FILE_NAME_TEMPLATE,
    }

    handler = ProductionStatisticsHandler(args_dict, output_path=tmp_path)

    assert isinstance(handler.grid_points_production, Table)
    assert len(handler.grid_points_production) == 2


def test_load_grid_points_production_file_not_found(args_dict, tmp_path):
    """Test behavior when grid_points_production_file is not found."""
    args_dict["grid_points_production_file"] = "non_existent_file.ecsv"

    with pytest.raises(FileNotFoundError):
        ProductionStatisticsHandler(args_dict, output_path=tmp_path)


def test_empty_grid_points_production_file(metrics_file, tmp_path):
    """Test behavior when grid_points_production_file is empty."""
    grid_points_file = tmp_path / "empty_grid_points.ecsv"
    Table(rows=[], names=["azimuth", "zenith_angle", "nsb_level", "offset"]).write(
        grid_points_file, format="ascii.ecsv", overwrite=True
    )

    args_dict = {
        "base_path": BASE_PATH,
        "zeniths": [20, 40],
        "offsets": [0.5],
        "azimuths": [0],
        "nsb": [0.005],
        "grid_points_production_file": str(grid_points_file),
        "metrics_file": str(metrics_file),
        "output_file": OUTPUT_FILE,
        "file_name_template": FILE_NAME_TEMPLATE,
    }

    handler = ProductionStatisticsHandler(args_dict, output_path=tmp_path)

    assert isinstance(handler.grid_points_production, Table)
    assert len(handler.grid_points_production) == 0


def test_grid_points_with_incorrect_format(metrics_file, tmp_path):
    """Test behavior when grid_points_production_file has unsupported format."""
    grid_points_file = tmp_path / "incorrect_format.json"
    grid_points_file.write_text("{}", encoding="utf-8")

    args_dict = {
        "base_path": BASE_PATH,
        "zeniths": [20, 40],
        "offsets": [0.5],
        "azimuths": [0],
        "nsb": [0.005],
        "grid_points_production_file": str(grid_points_file),
        "metrics_file": str(metrics_file),
        "output_file": OUTPUT_FILE,
        "file_name_template": FILE_NAME_TEMPLATE,
    }

    with pytest.raises(ValueError, match="must be an ECSV file"):
        ProductionStatisticsHandler(args_dict, output_path=tmp_path)


def test_initialize_evaluators_with_valid_files(mock_handler):
    """Test initialize_evaluators with valid files."""

    mock_handler.args["zeniths"] = [20, 40]
    mock_handler.args["azimuths"] = [0]
    mock_handler.args["nsb"] = [0.005]
    mock_handler.args["offsets"] = [0.5, 1.0]
    mock_handler.args["base_path"] = "test/path"
    mock_handler.args["file_name_template"] = "test_{zenith}.fits"
    mock_handler.evaluator_instances = []

    mock_evaluator_instance = MagicMock()
    mock_evaluator_instance.calculate_metrics.return_value = None
    mock_evaluator_instance.calculate_overall_metric.return_value = 0.1

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.glob", return_value=["dummy_path"]),
        patch(
            "simtools.production_configuration.derive_production_statistics_handler."
            "StatisticalUncertaintyEvaluator",
            return_value=mock_evaluator_instance,
        ),
    ):
        mock_handler.initialize_evaluators()
    expected_count = (
        len(mock_handler.args["zeniths"])
        * len(mock_handler.args["azimuths"])
        * len(mock_handler.args["nsb"])
        * len(mock_handler.args["offsets"])
    )
    assert len(mock_handler.evaluator_instances) == expected_count
    assert mock_evaluator_instance.calculate_metrics.call_count == expected_count


def test_perform_interpolation_with_grid_points(mock_handler):
    """Test perform_interpolation with grid points."""
    mock_evaluator = MagicMock()
    mock_evaluator.data = {
        "bin_edges_low": np.array([1.0, 10.0, 100.0]),
        "bin_edges_high": np.array([10.0, 100.0, 1000.0]),
    }
    mock_handler.evaluator_instances = [mock_evaluator]

    mock_handler.grid_points_production = Table(
        rows=[
            {"azimuth": 0.0, "zenith_angle": 20.0, "nsb_level": 0.0, "offset": 0.5},
            {"azimuth": 0.0, "zenith_angle": 40.0, "nsb_level": 0.0, "offset": 0.5},
        ]
    )

    mock_interp = MagicMock()
    mock_interp.interpolate.return_value = np.array([100])

    with patch(
        "simtools.production_configuration.derive_production_statistics_handler."
        "InterpolationHandler",
        return_value=mock_interp,
    ):
        result = mock_handler.perform_interpolation()

        assert isinstance(result, Table)
        assert len(result) == 1
        assert result["interpolated_production_statistics"][0] == 100

        assert mock_interp.interpolate.called
