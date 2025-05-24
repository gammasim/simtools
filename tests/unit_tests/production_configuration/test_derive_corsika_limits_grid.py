import astropy.units as u
import pytest
from astropy.table import Table

from simtools.production_configuration.derive_corsika_limits_grid import (
    _create_results_table,
    _process_file,
    generate_corsika_limits_grid,
    write_results,
)


@pytest.fixture
def mock_args_dict():
    """Create mock arguments dictionary."""
    return {
        "event_data_files": "data_files.yml",
        "telescope_ids": "telescope_ids.yml",
        "loss_fraction": 0.2,
        "plot_histograms": False,
        "output_file": "test_output.ecsv",
    }


@pytest.fixture
def mock_results():
    """Create mock results list."""
    return [
        {
            "primary_particle": "gamma",
            "telescope_ids": [1, 2],
            "zenith": 20.0 * u.deg,
            "azimuth": 180.0 * u.deg,
            "nsb_level": 1.0,
            "lower_energy_limit": 0.5 * u.TeV,
            "upper_radius_limit": 400.0 * u.m,
            "viewcone_radius": 5.0 * u.deg,
            "array_name": "LST",
            "layout": "LST",
        }
    ]


def test_generate_corsika_limits_grid(mocker, mock_args_dict):
    """Test generate_corsika_limits_grid function."""
    # Mock dependencies
    mock_collect = mocker.patch("simtools.utils.general.collect_data_from_file")
    mock_collect.side_effect = [
        {"files": ["file1.fits", "file2.fits"]},
        {"telescope_configs": {"LST": [1, 2], "MST": [3, 4]}},
    ]

    mock_process = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits_grid._process_file"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits_grid.write_results"
    )

    # Run function
    generate_corsika_limits_grid(mock_args_dict)

    # Verify calls
    assert mock_collect.call_count == 2
    assert mock_process.call_count == 4  # 2 files * 2 configs
    assert mock_write.call_count == 1


def test_process_file(mocker):
    """Test _process_file function."""
    mock_calculator = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits_grid.LimitCalculator"
    )
    mock_calculator.return_value.compute_limits.return_value = {"test": "limits"}
    mock_calculator.return_value.plot_data.return_value = None

    mocker.patch("simtools.io_operations.io_handler.IOHandler")

    result = _process_file("test.fits", [1, 2], 0.2, True, "array_name")

    assert result == {"test": "limits"}
    mock_calculator.return_value.plot_data.assert_called_once()


def test_write_results(mocker, mock_args_dict, mock_results, tmp_path):
    """Test write_results function."""
    mock_io = mocker.patch("simtools.io_operations.io_handler.IOHandler")
    mock_io.return_value.get_output_directory.return_value = tmp_path

    mock_dump = mocker.patch("simtools.data_model.metadata_collector.MetadataCollector.dump")

    write_results(mock_results, mock_args_dict)

    # Verify metadata was written
    mock_dump.assert_called_once()
    args = mock_dump.call_args[0]
    assert args[0] == mock_args_dict


def test_create_results_table(mock_results):
    """Test _create_results_table function."""
    table = _create_results_table(mock_results, loss_fraction=0.2)
    table.info()

    assert isinstance(table, Table)
    assert len(table) == 1
    assert "zenith" in table.colnames
    assert table["zenith"].unit == u.deg
    assert table.meta["loss_fraction"] == 0.2
    assert isinstance(table.meta["created"], str)
    assert "description" in table.meta
