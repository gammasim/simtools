#!/usr/bin/python3

from unittest import mock

import pytest

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

    mock_read_table_data.assert_called_once_with(config, None)
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
