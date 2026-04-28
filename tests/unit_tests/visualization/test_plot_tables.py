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

    mock_read_table_data.assert_called_once_with(config, None)
    mock_visualize.plot_1d.assert_called_once_with(mock_data, **config)
    mock_visualize.save_figure.assert_called_once_with(mock_fig, output_file)


@mock.patch("simtools.visualization.plot_tables.read_simtel_table")
def test_read_astropy_table_data_from_file(mock_read_simtel_table):
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "test_file.ecsv",
                "column_x": "x",
                "column_y": "y",
                "select_values": {"column_name": "x", "value": 42},
            },
        ]
    }
    mock_read_simtel_table.return_value = Table({"x": [41, 42], "y": [1.0, 2.0]})

    result = plot_tables.read_table_data(config, None)

    mock_read_simtel_table.assert_called_once_with(None, "test_file.ecsv")
    np.testing.assert_array_equal(result["test_table"]["x"], np.array([42]))
    np.testing.assert_array_equal(result["test_table"]["y"], np.array([2.0]))


def test_read_simtel_table_data_from_file():
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "spe_LST_2022-04-27_AP2.0e-4.dat",
                "parameter": "pm_photoelectron_spectrum",
                "column_x": "amplitude",
                "column_y": "response",
            },
        ]
    }

    result = plot_tables.read_table_data(config, Path("tests/resources"))

    assert len(result["test_table"]) == 2101
    assert result["test_table"].dtype.names == ("amplitude", "response")


def test_read_simtel_table_data_from_file_without_parameter_raises():
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "spe_LST_2022-04-27_AP2.0e-4.dat",
                "column_x": "amplitude",
                "column_y": "response",
            },
        ]
    }

    with pytest.raises(
        ValueError, match=r"Parameter name must be provided for sim_telarray table reading\."
    ):
        plot_tables.read_table_data(config, Path("tests/resources"))


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
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_db_handler.export_model_file.return_value = mock_table
    mock_structure_array = mock.MagicMock()
    mock_get_structure_array_from_table.return_value = mock_structure_array

    result = plot_tables.read_table_data(config)

    mock_db_handler_class.assert_called_once_with()
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

    with pytest.raises(ValueError, match=r"No table data defined in configuration."):
        plot_tables.read_table_data(config)


@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_export_model_file(mock_db_handler_class):
    table_config = {
        "site": "test_site",
        "telescope": "test_telescope",
        "model_version": "test_version",
        "parameter": "test_parameter",
    }
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_db_handler.export_model_file.return_value = mock_table

    config = {"tables": [table_config]}
    table_config["label"] = "test_label"
    table_config["column_x"] = "x"
    table_config["column_y"] = "y"

    plot_tables.read_table_data(config)

    mock_db_handler_class.assert_called_once_with()
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
    mock_db_handler = mock_db_handler_class.return_value
    mock_table = mock.MagicMock()
    mock_db_handler.export_model_file.return_value = mock_table

    config = {"tables": [table_config]}
    table_config["label"] = "test_label"
    table_config["column_x"] = "x"
    table_config["column_y"] = "y"

    plot_tables.read_table_data(config)

    mock_db_handler_class.assert_called_once_with()
    mock_db_handler.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name=None,
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
    )


@mock.patch("simtools.visualization.plot_tables.db_handler.DatabaseHandler")
def test_export_model_file_with_db_export_path(mock_db_handler_class):
    table_config = {
        "site": "test_site",
        "telescope": "test_telescope",
        "model_version": "test_version",
        "parameter": "test_parameter",
        "db_export_path": "/tmp/test_plot_tables",
    }
    mock_db_handler = mock_db_handler_class.return_value
    mock_db_handler.io_handler.output_path.get.return_value = "output/default"
    mock_db_handler.export_model_file.return_value = mock.MagicMock()

    config = {"tables": [table_config]}
    table_config["label"] = "test_label"
    table_config["column_x"] = "x"
    table_config["column_y"] = "y"

    plot_tables.read_table_data(config)

    mock_db_handler.io_handler.set_paths.assert_any_call(output_path="/tmp/test_plot_tables")
    mock_db_handler.io_handler.set_paths.assert_any_call(output_path="output/default")


def test_read_table_and_normalize():
    config = {
        "tables": [
            {
                "file_name": (
                    "tests/resources//SinglePhe_spectrum_totalfit_19pixel-average_20200601.csv"
                ),
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
    assert data["test_table"]["response"].max() == pytest.approx(1.0)


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
@mock.patch("simtools.visualization.plot_tables._read_parameter_dict_from_model_database")
@mock.patch("simtools.visualization.plot_tables.ascii_handler.collect_data_from_file")
def test_generate_plot_configurations(
    mock_collect_data, mock_read_parameter_dict, mock_read_table, tmp_test_directory
):
    # Mock the table data
    mock_table = Table()
    mock_table["time"] = [1.0, 2.0, 3.0]
    mock_table["amplitude"] = [0.1, 0.2, 0.3]
    mock_read_table.return_value = mock_table
    mock_read_parameter_dict.return_value = {"model_parameter_schema_version": "0.2.0"}

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
@mock.patch("simtools.visualization.plot_tables._read_parameter_dict_from_model_database")
def test_generate_plot_configurations_with_nan_and_missing_columns(
    mock_read_parameter_dict, mock_read_table, mock_collect_data, tmp_test_directory
):
    """Test handling of NaN values and missing columns in generate_plot_configurations."""
    # Create mock table with valid and NaN columns
    mock_table = Table()
    mock_table["time"] = [1.0, 2.0, 3.0]
    mock_table["amplitude"] = [0.1, 0.2, 0.3]
    mock_table["amplitude_low_gain"] = [np.nan, np.nan, np.nan]

    mock_read_table.return_value = mock_table
    mock_read_parameter_dict.return_value = {"model_parameter_schema_version": "0.2.0"}

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
    configs, _ = plot_tables.generate_plot_configurations(
        parameter="test_parameter",
        parameter_version="1.0.0",
        site="South",
        telescope="LSTS-01",
        output_path=tmp_test_directory,
        plot_type="all",
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
    )

    # Should return None since no valid configs were found for this plot type
    assert result is None


@mock.patch("simtools.visualization.plot_tables._read_table_from_model_database")
@mock.patch("simtools.visualization.plot_tables._read_parameter_dict_from_model_database")
@mock.patch("simtools.visualization.plot_tables.ascii_handler.collect_data_from_file")
def test_generate_plot_configurations_selects_schema_matching_parameter_version(
    mock_collect_data, mock_read_parameter_dict, mock_read_table, tmp_test_directory
):
    """Select the schema entry matching model_parameter_schema_version from a list."""
    mock_table = Table()
    mock_table["time"] = [1.0, 2.0, 3.0]
    mock_table["amplitude"] = [0.1, 0.2, 0.3]
    mock_read_table.return_value = mock_table
    mock_read_parameter_dict.return_value = {"model_parameter_schema_version": "0.1.0"}

    mock_collect_data.return_value = [
        {
            "schema_version": "0.1.0",
            "plot_configuration": [
                {"type": "legacy_plot", "tables": [{"column_x": "time", "column_y": "amplitude"}]}
            ],
        },
        {
            "schema_version": "0.2.0",
            "plot_configuration": [
                {
                    "type": "fadc_pulse_shape",
                    "tables": [{"column_x": "time", "column_y": "amplitude"}],
                }
            ],
        },
    ]

    configs, output_files = plot_tables.generate_plot_configurations(
        parameter="fadc_pulse_shape",
        parameter_version="1.0.0",
        site="South",
        telescope="SSTS-design",
        output_path=tmp_test_directory,
        plot_type="legacy_plot",
    )

    assert len(configs) == 1
    assert configs[0]["type"] == "legacy_plot"
    assert len(output_files) == 1


def test_select_schema_entry_returns_latest_when_version_not_found():
    """Fall back to newest schema entry when requested version is unavailable."""
    schema_data = [
        {"schema_version": "0.1.0", "plot_configuration": [{"type": "old"}]},
        {"schema_version": "0.3.0", "plot_configuration": [{"type": "new"}]},
    ]

    result = plot_tables._select_schema_entry(schema_data, schema_version="0.2.0")

    assert result["schema_version"] == "0.3.0"


def test_select_schema_entry_returns_empty_dict_for_empty_input():
    """Return an empty dict when schema input is empty/invalid."""
    assert plot_tables._select_schema_entry([], schema_version="0.2.0") == {}
