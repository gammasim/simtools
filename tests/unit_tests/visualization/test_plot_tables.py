#!/usr/bin/python3

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from astropy.table import Table

from simtools.visualization import plot_tables


@mock.patch("simtools.visualization.plot_tables.visualize")
@mock.patch("simtools.visualization.plot_tables.read_table_data")
def test_plot(mock_read_table_data, mock_visualize):
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "test_file",
                "type": "csv",
                "column_x": "x",
                "column_y": "y",
            }
        ]
    }
    output_file = "output.png"
    mock_data = {"test_table": "mock_data"}
    mock_read_table_data.return_value = mock_data
    mock_fig = mock.MagicMock()
    mock_visualize.plot_1d.return_value = mock_fig

    plot_tables.plot(config, output_file)

    mock_read_table_data.assert_called_once_with(config, None, None)
    mock_visualize.plot_1d.assert_called_once_with(mock_data, **config)
    mock_visualize.save_figure.assert_called_once_with(mock_fig, output_file)


@mock.patch("simtools.visualization.plot_tables.Table.read")
def test_read_astropy_table_data_from_file(
    mock_table_read,
):
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "test_file",
                "type": "ascii.ecsv",
                "column_x": "x",
                "column_y": "y",
                "select_values": {"column_name": "x", "value": 42},
            },
        ]
    }
    mock_table = mock.MagicMock()
    mock_table_read.return_value = mock_table

    plot_tables.read_table_data(config, None)
    mock_table_read.assert_called_once_with("test_file", format="ascii.ecsv")


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.legacy_data_handler.read_legacy_data_as_table")
@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_read_table_data_from_file(
    mock_db_handler_class,
    mock_read_legacy_data_as_table,
    mock_get_structure_array_from_table,
):
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "test_file",
                "type": "legacy_csv",
                "column_x": "x",
                "column_y": "y",
            },
        ]
    }
    mock_table = mock.MagicMock()
    mock_read_legacy_data_as_table.return_value = mock_table
    mock_structure_array = mock.MagicMock()
    mock_get_structure_array_from_table.return_value = mock_structure_array

    result = plot_tables.read_table_data(config, None)

    mock_read_legacy_data_as_table.assert_called_once_with("test_file", "legacy_csv")
    mock_get_structure_array_from_table.assert_called_once_with(mock_table, ["x", "y", None, None])
    assert result == {"test_table": mock_structure_array}


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.legacy_data_handler.read_legacy_data_as_table")
@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_read_table_data_from_model_database(
    mock_db_handler_class,
    mock_read_legacy_data_as_table,
    mock_get_structure_array_from_table,
):
    config = {
        "tables": [
            {
                "label": "test_table",
                "parameter": "test_parameter",
                "site": "test_site",
                "telescope": "test_telescope",
                "model_version": "test_version",
                "column_x": "x",
                "column_y": "y",
            }
        ]
    }
    db_config = None
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_db_handler.export_model_file.return_value = mock_table
    mock_structure_array = mock.MagicMock()
    mock_get_structure_array_from_table.return_value = mock_structure_array

    result = plot_tables.read_table_data(config, db_config)

    mock_db_handler_class.assert_called_once_with(mongo_db_config=db_config)
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name="test_telescope",
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
    )
    mock_get_structure_array_from_table.assert_called_once_with(mock_table, ["x", "y", None, None])
    assert result == {"test_table": mock_structure_array}


def test_read_table_data_no_table_data_defined():
    config = {"tables": [{"label": "test_table", "column_x": "x", "column_y": "y"}]}
    db_config = None

    with pytest.raises(ValueError, match="No table data defined in configuration."):
        plot_tables.read_table_data(config, db_config)


@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_export_model_file(mock_db_handler_class):
    table_config = {
        "site": "test_site",
        "telescope": "test_telescope",
        "model_version": "test_version",
        "parameter": "test_parameter",
    }
    db_config = None
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_db_handler.export_model_file.return_value = mock_table

    config = {"tables": [table_config]}
    table_config["label"] = "test_label"
    table_config["column_x"] = "x"
    table_config["column_y"] = "y"

    plot_tables.read_table_data(config, db_config)

    mock_db_handler_class.assert_called_once_with(mongo_db_config=db_config)
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name="test_telescope",
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
    )


@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_export_model_file_site_only(mock_db_handler_class):
    table_config = {
        "site": "test_site",
        "model_version": "test_version",
        "parameter": "test_parameter",
    }
    db_config = None
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_db_handler.export_model_file.return_value = mock_table

    config = {"tables": [table_config]}
    table_config["label"] = "test_label"
    table_config["column_x"] = "x"
    table_config["column_y"] = "y"

    plot_tables.read_table_data(config, db_config)

    mock_db_handler_class.assert_called_once_with(mongo_db_config=db_config)
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name=None,
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
    )


def test_read_table_and_normalize():
    config = {
        "tables": [
            {
                "file_name": "tests/resources//SinglePhe_spectrum_totalfit_19pixel-average_20200601.csv",
                "type": "legacy_lst_single_pe",
                "label": "test_table",
                "column_x": "amplitude",
                "column_y": "response",
                "normalize_y": True,
            }
        ]
    }
    data = plot_tables.read_table_data(config, None)
    assert isinstance(data, dict)
    assert data["test_table"]["response"].max() == 1.0


@pytest.mark.parametrize(
    (
        "parameter",
        "parameter_version",
        "site",
        "telescope",
        "plot_type",
        "output_path",
        "file_extension",
        "expected",
    ),
    [
        (
            "par",
            "1.0.0",
            "North",
            "LSTN-01",
            "typeA",
            None,
            ".pdf",
            Path("par_1.0.0_North_LSTN-01_typeA.pdf"),
        ),
        (
            "par",
            "1.0.0",
            "North",
            None,
            "typeA",
            None,
            ".pdf",
            Path("par_1.0.0_North_typeA.pdf"),
        ),
        (
            "par",
            "1.0.0",
            "North",
            "LSTN-01",
            "par",
            None,
            ".pdf",
            Path("par_1.0.0_North_LSTN-01.pdf"),
        ),
        (
            "par",
            "1.0.0",
            "North",
            None,
            "par",
            None,
            ".pdf",
            Path("par_1.0.0_North.pdf"),
        ),
        (
            "par",
            "1.0.0",
            "North",
            "LSTN-01",
            "typeA",
            "plots",
            ".png",
            Path("plots") / "par_1.0.0_North_LSTN-01_typeA.png",
        ),
        (
            "par",
            "1.0.0",
            "North",
            None,
            "typeA",
            "plots",
            ".png",
            Path("plots") / "par_1.0.0_North_typeA.png",
        ),
        (
            "par",
            "1.0.0",
            "North",
            "LSTN-01",
            "par",
            "plots",
            ".png",
            Path("plots") / "par_1.0.0_North_LSTN-01.png",
        ),
        (
            "par",
            "1.0.0",
            "North",
            None,
            "par",
            "plots",
            ".png",
            Path("plots") / "par_1.0.0_North.png",
        ),
    ],
)
def test_generate_output_file_name(
    parameter, parameter_version, site, telescope, plot_type, output_path, file_extension, expected
):
    result = plot_tables._generate_output_file_name(
        parameter=parameter,
        parameter_version=parameter_version,
        site=site,
        telescope=telescope,
        plot_type=plot_type,
        output_path=output_path,
        file_extension=file_extension,
    )
    assert result == expected


@mock.patch("simtools.visualization.plot_tables._read_table_from_model_database")
@mock.patch("simtools.visualization.plot_tables.ascii_handler.collect_data_from_file")
def test_generate_plot_configurations(
    mock_collect_data, mock_read_table, tmp_test_directory, db_config
):
    # Mock the table data
    mock_table = Table()
    mock_table["time"] = [1.0, 2.0, 3.0]
    mock_table["amplitude"] = [0.1, 0.2, 0.3]
    mock_read_table.return_value = mock_table

    # Test with parameter that has no plot configuration
    mock_collect_data.return_value = {}
    assert (
        plot_tables.generate_plot_configurations(
            parameter="num_gains",
            parameter_version="1.0.0",
            site="South",
            telescope="SSTS-design",
            output_path=tmp_test_directory,
            plot_type="all",
            db_config=db_config,
        )
        is None
    )

    # Mock schema for atmospheric_profile
    mock_collect_data.return_value = {
        "plot_configuration": [
            {"type": "profile_plot", "tables": [{"column_x": "time", "column_y": "amplitude"}]}
        ]
    }

    # Test with parameter that has plot configuration and no telescope
    configs, output_files = plot_tables.generate_plot_configurations(
        parameter="atmospheric_profile",
        parameter_version="1.0.0",
        site="South",
        telescope=None,
        output_path=tmp_test_directory,
        plot_type="all",
        db_config=db_config,
    )
    assert len(configs) > 0
    for _file in output_files:
        assert "atmospheric_profile" in str(_file)

    # Mock schema for fadc_pulse_shape
    mock_collect_data.return_value = {
        "plot_configuration": [
            {"type": "fadc_pulse_shape", "tables": [{"column_x": "time", "column_y": "amplitude"}]}
        ]
    }

    # Test with specific plot type
    configs, output_files = plot_tables.generate_plot_configurations(
        parameter="fadc_pulse_shape",
        parameter_version="1.0.0",
        site="South",
        telescope="SSTS-design",
        output_path=tmp_test_directory,
        plot_type="fadc_pulse_shape",
        db_config=db_config,
    )
    assert len(configs) == 1
    assert "tables" in configs[0]
    assert configs[0]["tables"][0]["column_x"] == "time"

    # Test with non-existent plot type should return None
    result = plot_tables.generate_plot_configurations(
        parameter="fadc_pulse_shape",
        parameter_version="1.0.0",
        site="South",
        telescope="SSTS-design",
        output_path=tmp_test_directory,
        plot_type="non_existent_type",
        db_config=db_config,
    )
    assert result is None


def test_get_plotting_label_unique_label():
    config = {"label": "unique_label", "column_x": "x", "column_y": "y"}
    data = {}
    result = plot_tables._get_plotting_label(config, data)
    assert result == "unique_label"


def test_get_plotting_label_default_label():
    config = {"column_x": "x", "column_y": "y"}
    data = {}
    result = plot_tables._get_plotting_label(config, data)
    assert result == "x vs y"


def test_get_plotting_label_duplicate_label():
    config = {"label": "duplicate_label", "column_x": "x", "column_y": "y"}
    data = {"duplicate_label": "data"}
    result = plot_tables._get_plotting_label(config, data)
    assert result == "duplicate_label (1)"


def test_get_plotting_label_multiple_duplicates():
    config = {"label": "duplicate_label", "column_x": "x", "column_y": "y"}
    data = {
        "duplicate_label": "data",
        "duplicate_label (1)": "data",
        "duplicate_label (2)": "data",
    }
    result = plot_tables._get_plotting_label(config, data)
    assert result == "duplicate_label (3)"


@mock.patch("simtools.visualization.plot_tables.ascii_handler.collect_data_from_file")
@mock.patch("simtools.visualization.plot_tables._read_table_from_model_database")
def test_generate_plot_configurations_with_nan_and_missing_columns(
    mock_read_table, mock_collect_data, tmp_test_directory, db_config
):
    """Test handling of NaN values and missing columns in generate_plot_configurations."""
    # Create mock table with valid and NaN columns
    mock_table = Table()
    mock_table["time"] = [1.0, 2.0, 3.0]
    mock_table["amplitude"] = [0.1, 0.2, 0.3]
    mock_table["amplitude_low_gain"] = [np.nan, np.nan, np.nan]

    mock_read_table.return_value = mock_table

    # Mock schema with multiple table configs including one with missing column
    mock_schema = {
        "plot_configuration": [
            {"type": "valid_plot", "tables": [{"column_x": "time", "column_y": "amplitude"}]},
            {
                "type": "invalid_plot_all_nan",
                "tables": [{"column_x": "time", "column_y": "amplitude_low_gain"}],
            },
            {
                "type": "multiple_tables_plot",
                "tables": [
                    {"column_x": "time", "column_y": "amplitude"},
                    {
                        "column_x": "wavelength",
                        "column_y": "amplitude",
                    },
                ],
            },
        ]
    }
    mock_collect_data.return_value = mock_schema

    # Test with valid configuration
    configs, output_files = plot_tables.generate_plot_configurations(
        parameter="test_parameter",
        parameter_version="1.0.0",
        site="South",
        telescope="LSTS-01",
        output_path=tmp_test_directory,
        plot_type="all",
        db_config=db_config,
    )

    # Should only have the valid configuration
    assert len(configs) == 1
    assert configs[0]["type"] == "valid_plot"

    # Test with specific plot type
    mock_read_table.reset_mock()
    configs, _ = plot_tables.generate_plot_configurations(
        parameter="test_parameter",
        parameter_version="1.0.0",
        site="South",
        telescope="LSTS-01",
        output_path=tmp_test_directory,
        plot_type="valid_plot",
        db_config=db_config,
    )

    assert len(configs) == 1
    assert configs[0]["type"] == "valid_plot"

    # Test with invalid plot type (NaN column)
    mock_read_table.reset_mock()
    result = plot_tables.generate_plot_configurations(
        parameter="test_parameter",
        parameter_version="1.0.0",
        site="South",
        telescope="LSTS-01",
        output_path=tmp_test_directory,
        plot_type="invalid_plot_all_nan",
        db_config=db_config,
    )

    # Should return None since no valid configs were found for this plot type
    assert result is None
