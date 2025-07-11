import astropy.units as u
import pytest
from astropy.table import Table

from simtools.production_configuration.derive_corsika_limits import (
    _create_results_table,
    _process_file,
    _read_array_layouts_from_db,
    _round_value,
    generate_corsika_limits_grid,
    write_results,
)


@pytest.fixture
def mock_args_dict():
    """Create mock arguments dictionary."""
    return {
        "event_data_file": "data_files.hdf5",
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
        {"telescope_configs": {"LST": [1, 2], "MST": [3, 4]}},
    ]

    mock_process = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._process_file"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    # Run function
    generate_corsika_limits_grid(mock_args_dict)

    # Verify calls
    assert mock_collect.call_count == 1
    assert mock_process.call_count == 2  # 2 configs
    assert mock_write.call_count == 1


def test_process_file(mocker):
    """Test _process_file function."""
    mock_calculator = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.LimitCalculator"
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


def test_round_value():
    """Test _round_value function for different key types."""

    # Test lower_energy_limit rounding
    assert _round_value("lower_energy_limit", 1.2345) == 1.234
    assert _round_value("lower_energy_limit", 0.9876) == 0.987
    assert _round_value("lower_energy_limit", 2.0) == 2.0

    # Test upper_radius_limit rounding
    assert _round_value("upper_radius_limit", 123.4) == 125
    assert _round_value("upper_radius_limit", 100.0) == 100
    assert _round_value("upper_radius_limit", 101.0) == 125
    assert _round_value("upper_radius_limit", 75.0) == 75

    # Test viewcone_radius rounding
    assert _round_value("viewcone_radius", 1.1) == 1.25
    assert _round_value("viewcone_radius", 2.0) == 2.0
    assert _round_value("viewcone_radius", 2.1) == 2.25
    assert _round_value("viewcone_radius", 0.3) == 0.5

    # Test other keys (no rounding)
    assert _round_value("other_key", 1.2345) == 1.2345
    assert _round_value("zenith", 45.678) == 45.678
    assert _round_value("unknown", "string_value") == "string_value"


def test_read_array_layouts_from_db_specific_layouts(mocker):
    """Test _read_array_layouts_from_db with specific layout names."""
    mock_site_model = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.SiteModel"
    )
    instance = mock_site_model.return_value
    instance.get_array_elements_for_layout.side_effect = (
        lambda name: [1, 2] if name == "LST" else [3, 4]
    )

    layouts = ["LST", "MST"]
    site = "North"
    model_version = "v1.0.0"
    db_config = {"host": "localhost"}

    result = _read_array_layouts_from_db(layouts, site, model_version, db_config)

    assert result == {"LST": [1, 2], "MST": [3, 4]}
    mock_site_model.assert_called_once_with(
        site=site, model_version=model_version, mongo_db_config=db_config
    )
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")


def test_read_array_layouts_from_db_all_layouts(mocker):
    """Test _read_array_layouts_from_db with 'all' layouts."""
    mock_site_model = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.SiteModel"
    )
    instance = mock_site_model.return_value
    instance.get_list_of_array_layouts.return_value = ["LST", "MST"]
    instance.get_array_elements_for_layout.side_effect = (
        lambda name: [10, 20] if name == "LST" else [30, 40]
    )

    layouts = ["all"]
    site = "South"
    model_version = "v2.0.0"
    db_config = {"host": "db"}

    result = _read_array_layouts_from_db(layouts, site, model_version, db_config)

    assert result == {"LST": [10, 20], "MST": [30, 40]}
    instance.get_list_of_array_layouts.assert_called_once()
    assert instance.get_array_elements_for_layout.call_count == 2
    instance.get_array_elements_for_layout.assert_any_call("LST")
    instance.get_array_elements_for_layout.assert_any_call("MST")


def test_generate_corsika_limits_grid_with_db_layouts(mocker, mock_args_dict):
    """Test generate_corsika_limits_grid using _read_array_layouts_from_db."""
    # Prepare args_dict to use array_layout_name
    args = mock_args_dict.copy()
    args["array_layout_name"] = ["LST", "MST"]
    args["site"] = "North"
    args["model_version"] = "v1.2.3"

    mock_read_layouts = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._read_array_layouts_from_db"
    )
    mock_read_layouts.return_value = {"LST": [1, 2], "MST": [3, 4]}

    mock_process = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._process_file"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    generate_corsika_limits_grid(args)

    mock_read_layouts.assert_called_once_with(
        args["array_layout_name"], args["site"], args["model_version"], None
    )
    assert mock_process.call_count == 2  # 2 layouts
    assert mock_write.call_count == 1
