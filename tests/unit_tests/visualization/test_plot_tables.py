#!/usr/bin/python3

import logging
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from simtools.constants import TEST_RESOURCES_GENERATED
from simtools.reporting import docs_parameter_plots
from simtools.visualization import plot_tables

PLOT_INTERSECTION_PROTON_CURVE = (
    "simtools.visualization.plot_tables._plot_intersection_proton_curve"
)


@mock.patch("simtools.visualization.plot_tables.visualize")
@mock.patch("simtools.visualization.plot_tables.read_table_data")
def test_plot(mock_read_table_data, mock_visualize, run_context):
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

    plot_tables.plot(config, output_file, context=run_context)

    mock_read_table_data.assert_called_once_with(
        config, None, model_reader=None, context=run_context
    )
    mock_visualize.plot_1d.assert_called_once_with(mock_data, **config)
    mock_visualize.save_figure.assert_called_once_with(
        mock_fig, output_file, close=True, context=run_context
    )


def test_select_values_from_table_accepts_unitless_schema_value_for_unit_column():
    table = Table({"altitude": [15.0, 30.0] * u.km, "extinction": [0.1, 0.2]})

    selected = plot_tables._select_values_from_table(table, "altitude", 30.0)

    assert len(selected) == 1
    assert selected["altitude"].unit == u.km
    np.testing.assert_allclose(selected["altitude"][0], 30.0)


def test_read_ecsv_table_data_from_file(tmp_test_directory):
    source = Path(tmp_test_directory) / "spectrum.ecsv"
    Table({"amplitude": [1.0, 2.0, 3.0], "response": [0.1, 0.2, 0.3]}).write(
        source, format="ascii.ecsv"
    )
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": source.name,
                "parameter": "pm_photoelectron_spectrum",
                "column_x": "amplitude",
                "column_y": "response",
            },
        ]
    }

    result = plot_tables.read_table_data(config, source.parent)

    assert len(result["test_table"]) == 3
    assert result["test_table"].dtype.names == ("amplitude", "response")


def test_read_non_ecsv_table_data_from_file_raises():
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

    with pytest.raises(ValueError, match="ECSV"):
        plot_tables.read_table_data(config, Path(TEST_RESOURCES_GENERATED) / "model_parameters")


@mock.patch("simtools.visualization.plot_tables.legacy_data_handler.read_legacy_data_as_table")
def test_read_table_data_from_file(mock_read_legacy_data_as_table):
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
    mock_table = Table({"x": [1.0], "y": [2.0]})
    mock_read_legacy_data_as_table.return_value = mock_table

    result = plot_tables.read_table_data(config, None)

    mock_read_legacy_data_as_table.assert_called_once_with("test_file", "legacy_csv")
    assert result["test_table"].dtype.names == ("x", "y")
    assert result["test_table"][0].tolist() == (1.0, 2.0)


def test_read_table_data_from_model_repository(run_context):
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
    model_reader = mock.MagicMock()
    model_reader.export_model_file.return_value = Table({"x": [1.0], "y": [2.0]})

    result = plot_tables.read_table_data(config, model_reader=model_reader, context=run_context)

    model_reader.export_model_file.assert_called_once_with(
        parameter="test_parameter",
        site="test_site",
        array_element_name="test_telescope",
        parameter_version=None,
        model_version="test_version",
        export_file_as_table=True,
        dest=mock.ANY,
    )
    assert result["test_table"].dtype.names == ("x", "y")
    assert result["test_table"][0].tolist() == (1.0, 2.0)


def test_read_table_data_no_table_data_defined():
    config = {"tables": [{"label": "test_table", "column_x": "x", "column_y": "y"}]}

    with pytest.raises(ValueError, match=r"No table data defined in configuration."):
        plot_tables.read_table_data(config)


def test_export_model_file_with_model_export_path(tmp_test_directory):
    table_config = {
        "site": "test_site",
        "telescope": "test_telescope",
        "model_version": "test_version",
        "parameter": "test_parameter",
        "model_export_path": str(tmp_test_directory / "test_plot_tables"),
    }
    model_reader = mock.MagicMock()
    model_reader.export_model_file.return_value = mock.MagicMock()

    config = {"tables": [table_config]}
    table_config["label"] = "test_label"
    table_config["column_x"] = "x"
    table_config["column_y"] = "y"

    plot_tables.read_table_data(config, model_reader=model_reader)

    assert model_reader.export_model_file.call_args.kwargs["dest"] == str(
        tmp_test_directory / "test_plot_tables"
    )


def test_read_table_and_normalize(tmp_test_directory):
    test_file = Path(tmp_test_directory) / "single_pe.csv"
    test_file.write_text("0.0,0.5\n1.0,1.0\n", encoding="utf-8")
    config = {
        "tables": [
            {
                "file_name": test_file.name,
                "type": "legacy_lst_single_pe",
                "label": "test_table",
                "column_x": "amplitude",
                "column_y": "response",
                "normalize_y": True,
            }
        ]
    }
    data = plot_tables.read_table_data(config, tmp_test_directory)
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


@mock.patch("simtools.visualization.plot_tables._read_table_from_model_repository")
@mock.patch("simtools.visualization.plot_tables._read_parameter_dict_from_model_repository")
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


@mock.patch("simtools.visualization.plot_tables._read_table_from_model_repository")
@mock.patch("simtools.visualization.plot_tables._read_parameter_dict_from_model_repository")
@mock.patch("simtools.visualization.plot_tables.ascii_handler.collect_data_from_file")
def test_generate_plot_configurations_uses_and_names_selected_model_version(
    mock_collect_data, mock_read_parameter_dict, mock_read_table, tmp_test_directory
):
    mock_read_parameter_dict.return_value = {"model_parameter_schema_version": "0.3.0"}
    mock_read_table.return_value = Table({"wavelength": [300.0], "reflectivity": [0.9]})
    mock_collect_data.return_value = {
        "plot_configuration": [
            {
                "type": "mirror_reflectivity",
                "tables": [{"column_x": "wavelength", "column_y": "reflectivity"}],
            }
        ]
    }

    configs, output_files = plot_tables.generate_plot_configurations(
        parameter="mirror_reflectivity",
        parameter_version="2.0.0",
        site="North",
        telescope="LSTN-design",
        output_path=tmp_test_directory,
        plot_type="all",
        model_version="8.0.0",
    )

    assert mock_read_parameter_dict.call_args.args[0]["model_version"] == "8.0.0"
    assert mock_read_table.call_args.args[0]["model_version"] == "8.0.0"
    assert configs[0]["tables"][0]["model_version"] == "8.0.0"
    assert output_files[0].name == "mirror_reflectivity_2.0.0_North_LSTN-design.pdf"


def test_figure_cache_key_tracks_rendered_data_config_and_output_settings(run_context):
    plot_config = {
        "title": "Reflectivity",
        "xtitle": "Wavelength [nm]",
        "tables": [{"model_version": "7.0.0", "label": "Mirror", "column_y": "y"}],
    }
    data = {"Mirror": [(300.0, 0.91), (400.0, 0.87)]}
    context_dpi_120 = run_context.replace(args={"figure_dpi": 120})

    initial_key = docs_parameter_plots._get_figure_cache_key(plot_config, data, context_dpi_120)
    same_render_from_another_version = {
        **plot_config,
        "tables": [{**plot_config["tables"][0], "model_version": "8.0.0"}],
    }
    assert (
        docs_parameter_plots._get_figure_cache_key(
            same_render_from_another_version, data, context_dpi_120
        )
        == initial_key
    )

    changed_data = {"Mirror": [(300.0, 0.91), (400.0, 0.86)]}
    assert (
        docs_parameter_plots._get_figure_cache_key(plot_config, changed_data, context_dpi_120)
        != initial_key
    )
    changed_config = {**plot_config, "xtitle": "Wavelength [angstrom]"}
    assert (
        docs_parameter_plots._get_figure_cache_key(changed_config, data, context_dpi_120)
        != initial_key
    )

    context_dpi_240 = run_context.replace(args={"figure_dpi": 240})
    assert (
        docs_parameter_plots._get_figure_cache_key(plot_config, data, context_dpi_240)
        != initial_key
    )


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        (["duplicate_label"], "duplicate_label (1)"),
        (["duplicate_label", "duplicate_label (1)", "duplicate_label (2)"], "duplicate_label (3)"),
    ],
)
def test_get_plotting_label_duplicates(existing, expected):
    config = {"label": "duplicate_label", "column_x": "x", "column_y": "y"}
    data = dict.fromkeys(existing, "data")
    assert plot_tables._get_plotting_label(config, data) == expected


@mock.patch("simtools.visualization.plot_tables.ascii_handler.collect_data_from_file")
@mock.patch("simtools.visualization.plot_tables._read_table_from_model_repository")
@mock.patch("simtools.visualization.plot_tables._read_parameter_dict_from_model_repository")
def test_generate_plot_configurations_with_nan_and_missing_columns(
    mock_read_parameter_dict, mock_read_table, mock_collect_data, tmp_test_directory
):
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


@mock.patch("simtools.visualization.plot_tables._read_table_from_model_repository")
@mock.patch("simtools.visualization.plot_tables._read_parameter_dict_from_model_repository")
@mock.patch("simtools.visualization.plot_tables.ascii_handler.collect_data_from_file")
def test_generate_plot_configurations_selects_schema_matching_parameter_version(
    mock_collect_data, mock_read_parameter_dict, mock_read_table, tmp_test_directory
):
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
    schema_data = [
        {"schema_version": "0.1.0", "plot_configuration": [{"type": "old"}]},
        {"schema_version": "0.3.0", "plot_configuration": [{"type": "new"}]},
    ]

    result = plot_tables._select_schema_entry(schema_data, schema_version="0.2.0")

    assert result["schema_version"] == "0.3.0"


def test_select_schema_entry_returns_empty_dict_for_empty_input():
    assert plot_tables._select_schema_entry([], schema_version="0.2.0") == {}


def test_resolve_plot_output_path_file_and_directory(tmp_path):
    explicit_file = tmp_path / "custom_name.pdf"
    output_dir = tmp_path / "plots"

    assert plot_tables.resolve_plot_output_path(explicit_file) == explicit_file
    assert plot_tables.resolve_plot_output_path(output_dir) == output_dir / "bias_curve.png"


@mock.patch("simtools.visualization.plot_tables._configure_bias_curve_axis")
@mock.patch(PLOT_INTERSECTION_PROTON_CURVE)
@mock.patch("simtools.visualization.plot_tables._plot_scaled_proton_curve")
@mock.patch("simtools.visualization.plot_tables._plot_proton_curve")
@mock.patch("simtools.visualization.plot_tables._plot_nsb_curve")
@mock.patch("simtools.visualization.plot_tables.plt")
def test_plot_bias_curves_saves_and_closes_figure(
    mock_plt,
    mock_plot_nsb_curve,
    mock_plot_proton_curve,
    mock_plot_scaled_proton_curve,
    mock_plot_intersection_proton_curve,
    mock_configure_axis,
    tmp_path,
):
    mock_fig = mock.MagicMock()
    mock_axis = mock.MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axis)

    output_path = tmp_path / "nested" / "bias.png"
    config = {"title": "Bias", "ymin": 1, "ymax": 1e6}
    nsb_stats = {220: {"rate_hz": 10, "error_hz": 1}}
    proton_stats = {220: {"rate_hz": 5}}

    plot_tables.plot_bias_curves(nsb_stats, proton_stats, config, output_path)

    mock_plt.subplots.assert_called_once_with(figsize=(10, 7))
    mock_plot_nsb_curve.assert_called_once_with(mock_axis, nsb_stats)
    mock_plot_proton_curve.assert_called_once_with(mock_axis, proton_stats)
    mock_plot_scaled_proton_curve.assert_called_once_with(mock_axis, proton_stats, 1.35)
    mock_plot_intersection_proton_curve.assert_called_once_with(mock_axis, proton_stats, 1.35)
    mock_configure_axis.assert_called_once_with(mock_axis, config, nsb_stats, proton_stats)
    mock_fig.tight_layout.assert_called_once_with()
    mock_fig.savefig.assert_called_once_with(
        output_path, format="png", bbox_inches="tight", dpi=200
    )
    assert output_path.parent.exists()


@mock.patch("simtools.visualization.plot_tables._plot_log_linear_fit")
def test_plot_nsb_curve_draws_errorbar_and_fit(mock_plot_fit):
    axis = mock.MagicMock()
    nsb_stats = {
        240: {"rate_hz": 200.0, "error_hz": 10.0},
        220: {"rate_hz": 100.0, "error_hz": 5.0},
    }

    plot_tables._plot_nsb_curve(axis, nsb_stats)

    axis.errorbar.assert_called_once_with(
        [220, 240],
        [100.0, 200.0],
        yerr=[5.0, 10.0],
        fmt="o",
        label="NSB",
        color="tab:blue",
        capsize=3,
    )
    mock_plot_fit.assert_called_once_with(axis, [220, 240], [100.0, 200.0], color="tab:blue")


@pytest.mark.parametrize(
    ("curve_function", "rate_220", "rate_240", "err_220", "err_240", "fmt", "label", "color"),
    [
        (plot_tables._plot_nsb_curve, 100.0, 200.0, 5.0, 10.0, "o", "NSB", "tab:blue"),
        (plot_tables._plot_proton_curve, 10.0, 20.0, 1.0, 2.0, "s", "Proton", "tab:orange"),
        # Scaled proton curve multiplies rates and errors by 1.35
        (
            plot_tables._plot_scaled_proton_curve,
            10.0,
            20.0,
            1.0,
            2.0,
            "^",
            "1.35 x Proton",
            "tab:red",
        ),
    ],
)
@mock.patch("simtools.visualization.plot_tables._plot_log_linear_fit")
def test_plot_rate_curves_draw_points_and_fit(
    mock_plot_fit, curve_function, rate_220, rate_240, err_220, err_240, fmt, label, color
):
    axis = mock.MagicMock()
    stats = {
        240: {"rate_hz": rate_240, "error_hz": err_240},
        220: {"rate_hz": rate_220, "error_hz": err_220},
    }
    scale = 1.35 if curve_function is plot_tables._plot_scaled_proton_curve else 1.0
    rates = [rate_220 * scale, rate_240 * scale]
    curve_function(axis, stats)
    axis.errorbar.assert_called_once_with(
        [220, 240],
        pytest.approx(rates),
        yerr=pytest.approx([err_220 * scale, err_240 * scale]),
        fmt=fmt,
        label=label,
        color=color,
        capsize=3,
    )
    mock_plot_fit.assert_called_once_with(axis, [220, 240], pytest.approx(rates), color=color)


@pytest.mark.parametrize(
    "curve_function",
    [
        plot_tables._plot_nsb_curve,
        plot_tables._plot_proton_curve,
        plot_tables._plot_scaled_proton_curve,
    ],
)
@mock.patch("simtools.visualization.plot_tables._plot_log_linear_fit")
def test_plot_rate_curves_return_early_when_empty(mock_plot_fit, curve_function):
    axis = mock.MagicMock()
    curve_function(axis, {})
    axis.errorbar.assert_not_called()
    mock_plot_fit.assert_not_called()


def test_plot_log_linear_fit_returns_without_plot_for_insufficient_data(caplog):
    axis = mock.MagicMock()
    caplog.set_level(logging.INFO, logger="simtools.visualization.plot_tables")

    plot_tables._plot_log_linear_fit(axis, [220], [10.0], color="tab:blue")
    plot_tables._plot_log_linear_fit(axis, [220, 240], [0.0, -1.0], color="tab:blue")

    axis.plot.assert_not_called()
    assert caplog.text.count("need at least two distinct thresholds") == 2


def test_plot_log_linear_fit_plots_for_valid_positive_rates():
    axis = mock.MagicMock()

    plot_tables._plot_log_linear_fit(axis, [220, 240, 260], [100.0, 10.0, 1.0], color="tab:blue")

    axis.plot.assert_called_once()
    args, kwargs = axis.plot.call_args
    assert len(args[0]) == 100
    assert len(args[1]) == 100
    assert args[2] == "--"
    assert kwargs["color"] == "tab:blue"
    assert kwargs["linewidth"] == 1


def test_configure_bias_curve_axis_with_legend():
    axis = mock.MagicMock()
    axis.get_legend_handles_labels.return_value = ([object()], ["NSB"])
    config = {"title": "Bias", "ymin": 1, "ymax": 1e6}
    nsb_stats = {220: {"rate_hz": 10, "error_hz": 1}}
    proton_stats = {220: {"rate_hz": 5}}

    plot_tables._configure_bias_curve_axis(axis, config, nsb_stats, proton_stats)

    axis.set_title.assert_called_once_with("Bias", fontsize=14, fontweight="bold")
    axis.set_xlabel.assert_called_once_with("Threshold", fontsize=12)
    axis.set_ylabel.assert_called_once_with("Trigger Rate [Hz]", fontsize=12)
    axis.set_yscale.assert_called_once_with("log")
    axis.grid.assert_called_once_with(which="both", alpha=0.3, linestyle=":")
    axis.legend.assert_called_once_with(fontsize=11, loc="best")


@mock.patch("simtools.visualization.plot_tables._logger")
def test_configure_bias_curve_axis_without_legend_logs_warning(mock_logger):
    axis = mock.MagicMock()
    axis.get_legend_handles_labels.return_value = ([], [])
    config = {"title": "Bias", "ymin": 1, "ymax": 1e6}
    nsb_stats = {}
    proton_stats = {}

    plot_tables._configure_bias_curve_axis(axis, config, nsb_stats, proton_stats)

    axis.legend.assert_not_called()
    mock_logger.warning.assert_called_once_with(
        "No NSB or proton rates found; writing empty bias-curve plot"
    )


@pytest.mark.parametrize(
    ("trigger_threshold", "expected_line"),
    [(250.0, True), (None, False)],
)
@mock.patch("simtools.visualization.plot_tables._configure_bias_curve_axis")
@mock.patch(PLOT_INTERSECTION_PROTON_CURVE)
@mock.patch("simtools.visualization.plot_tables._plot_scaled_proton_curve")
@mock.patch("simtools.visualization.plot_tables._plot_proton_curve")
@mock.patch("simtools.visualization.plot_tables._plot_nsb_curve")
@mock.patch("simtools.visualization.plot_tables.plt")
def test_plot_bias_curves_trigger_threshold_line(
    mock_plt,
    mock_plot_nsb_curve,
    mock_plot_proton_curve,
    mock_plot_scaled_proton_curve,
    mock_plot_intersection_proton_curve,
    mock_configure_axis,
    trigger_threshold,
    expected_line,
    tmp_path,
):
    mock_axis = mock.MagicMock()
    mock_plt.subplots.return_value = (mock.MagicMock(), mock_axis)
    proton_stats = {220: {"rate_hz": 5}}

    plot_tables.plot_bias_curves(
        {220: {"rate_hz": 10, "error_hz": 1}},
        proton_stats,
        {"title": "Bias", "ymin": 1, "ymax": 1e6},
        tmp_path / "bias.png",
        trigger_threshold,
    )

    mock_plot_intersection_proton_curve.assert_called_once_with(mock_axis, proton_stats, 1.35)
    if expected_line:
        call_kwargs = mock_axis.axvline.call_args.kwargs
        assert call_kwargs["x"] == pytest.approx(trigger_threshold)
        assert call_kwargs["color"] == "grey"
        assert call_kwargs["linestyle"] == "--"
    else:
        mock_axis.axvline.assert_not_called()


@pytest.mark.parametrize(
    ("thresholds", "rates"), [([7], [10]), ([7, 7], [10, 20]), ([7, 8], [0, -1])]
)
def test_fit_bias_curve_insufficient_points(thresholds, rates):
    assert plot_tables.fit_bias_curve(thresholds, rates) is None


def test_fit_bias_curve_filters_invalid_points():
    thresholds, coefficients = plot_tables.fit_bias_curve(
        [7, 7.2, 8, 9, 10, np.inf], [100, 0, 10, np.nan, np.inf, 1]
    )
    assert thresholds == pytest.approx([7, 8])
    assert coefficients == pytest.approx([-1, 9])


def test_documented_library_plot_example(tmp_test_directory):
    data_path = Path(tmp_test_directory)
    Table({"wavelength": [300.0, 400.0], "efficiency": [0.2, 0.8]}).write(
        data_path / "response.ecsv", format="ascii.ecsv"
    )
    config = {
        "type": "example_response",
        "title": "Detector response",
        "xtitle": "Wavelength [nm]",
        "ytitle": "Response",
        "tables": [
            {
                "file_name": "response.ecsv",
                "label": "Response",
                "column_x": "wavelength",
                "column_y": "efficiency",
            }
        ],
    }
    output_file = data_path / "response.png"

    result = plot_tables.plot(config, output_file, data_path=data_path)

    assert result == output_file
    assert output_file.exists()
