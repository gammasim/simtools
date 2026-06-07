"""Tests for production grid plotting in production_configuration."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from astropy.table import Table

from simtools.visualization.plot_production_grid import (
    DEFAULT_OUTPUT_FILE_STEM,
    PLOT_VALUE_SPECS,
    ProductionGridPlotter,
)


def _write_grid_file(tmp_test_directory, file_name, grid_points):
    """Write grid points to a temporary ECSV file."""
    file_path = Path(tmp_test_directory) / file_name
    rows = []
    for point in grid_points:
        row = {}
        for key, value in point.items():
            if isinstance(value, dict) and "value" in value:
                row[key] = value["value"]
            else:
                row[key] = value
        rows.append(row)

    table = Table(rows=rows)
    if "azimuth" in table.colnames:
        table["azimuth"].unit = "deg"
    if "zenith_angle" in table.colnames:
        table["zenith_angle"].unit = "deg"
    if "ra" in table.colnames:
        table["ra"].unit = "deg"
    if "dec" in table.colnames:
        table["dec"].unit = "deg"
    if "nsb_level" in table.colnames:
        table["nsb_level"].unit = "MHz"
    if "offset" in table.colnames:
        table["offset"].unit = "deg"
    if "core_scatter_max_value" in table.colnames:
        table["core_scatter_max_value"].unit = "m"
    if "view_cone_max_value" in table.colnames:
        table["view_cone_max_value"].unit = "deg"
    table.write(file_path, format="ascii.ecsv", overwrite=True)
    return file_path


def _create_plotter(grid_file, output_path):
    """Create a plotter for file-driven plotting."""
    return ProductionGridPlotter(
        grid_points_file=grid_file,
        output_path=output_path,
    )


def test_normalize_altaz_point(tmp_test_directory):
    """Keep native Alt/Az values and no RA/Dec when absent in file."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz.ecsv",
        [{"azimuth": 180.0, "zenith_angle": 20.0, "nsb_level": 0.0}],
    )

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "altaz"
    assert normalized_points[0]["visible_in_altaz"] is True
    assert normalized_points[0]["azimuth"] == pytest.approx(180.0)
    assert normalized_points[0]["zenith"] == pytest.approx(20.0)
    assert normalized_points[0]["ra"] is None
    assert normalized_points[0]["dec"] is None


def test_normalize_flattened_job_grid_altaz_columns(tmp_test_directory):
    """Handle job-grid style flattened coordinate columns."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz_flattened.ecsv",
        [
            {
                "azimuth_angle_value": 0.0,
                "azimuth_angle_unit": "deg",
                "zenith_angle_value": 70.0,
                "zenith_angle_unit": "deg",
                "primary": "gamma",
            }
        ],
    )

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "altaz"
    assert normalized_points[0]["azimuth"] == pytest.approx(0.0)
    assert normalized_points[0]["zenith"] == pytest.approx(70.0)


def test_normalize_altaz_keeps_explicit_radec_columns(tmp_test_directory):
    """Use explicit RA/Dec columns from the same grid row when available."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz_with_radec.ecsv",
        [
            {
                "azimuth_angle_value": 10.0,
                "azimuth_angle_unit": "deg",
                "zenith_angle_value": 20.0,
                "zenith_angle_unit": "deg",
                "ra": 120.0,
                "dec": -20.0,
            }
        ],
    )

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "altaz"
    assert normalized_points[0]["ra"] == pytest.approx(120.0)
    assert normalized_points[0]["dec"] == pytest.approx(-20.0)


def test_normalize_radec_point_without_altaz_projection(tmp_test_directory):
    """Keep native RA/Dec points without deriving Alt/Az."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_radec.ecsv",
        [{"ra": 155.0, "dec": 30.0}],
    )

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "radec"
    assert normalized_points[0]["azimuth"] is None
    assert normalized_points[0]["zenith"] is None
    assert normalized_points[0]["ra"] == pytest.approx(155.0)
    assert normalized_points[0]["dec"] == pytest.approx(30.0)


def test_plot_sky_projection_creates_output_altaz_only(tmp_test_directory):
    """Write the sky projection plot with Alt/Az data only."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz_only.ecsv",
        [{"azimuth": 120.0, "zenith_angle": 35.0}],
    )
    output_path = Path(tmp_test_directory) / "output"

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=output_path,
    )

    plotter.plot_sky_projection(plot_ra_dec_tracks=True, dec_values=[20.0])

    assert (output_path / f"{DEFAULT_OUTPUT_FILE_STEM}.png").exists()


def test_plot_sky_projection_creates_output_with_radec_panel(tmp_test_directory):
    """Write the sky projection plot with both Alt/Az and RA/Dec data."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz_radec.ecsv",
        [{"azimuth": 100.0, "zenith_angle": 25.0, "ra": 180.0, "dec": -10.0}],
    )
    output_path = Path(tmp_test_directory) / "output"

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=output_path,
    )

    plotter.plot_sky_projection()

    assert (output_path / f"{DEFAULT_OUTPUT_FILE_STEM}.png").exists()


def test_plot_altaz_projection_with_limits_creates_outputs(tmp_test_directory):
    """Write Alt/Az color-scale plots and zenith profiles for all supported limits."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_energy_flattened.ecsv",
        [
            {
                "azimuth_angle_value": 0.0,
                "azimuth_angle_unit": "deg",
                "zenith_angle_value": 20.0,
                "zenith_angle_unit": "deg",
                "energy_min_value": 0.03,
                "energy_min_unit": "TeV",
                "energy_max_value": 150.0,
                "energy_max_unit": "TeV",
                "core_scatter_max_value": 1200.0,
                "core_scatter_max_unit": "m",
                "view_cone_max_value": 10.0,
                "view_cone_max_unit": "deg",
            },
            {
                "azimuth_angle_value": 180.0,
                "azimuth_angle_unit": "deg",
                "zenith_angle_value": 40.0,
                "zenith_angle_unit": "deg",
                "energy_min_value": 0.06,
                "energy_min_unit": "TeV",
                "energy_max_value": 200.0,
                "energy_max_unit": "TeV",
                "core_scatter_max_value": 1800.0,
                "core_scatter_max_unit": "m",
                "view_cone_max_value": 12.0,
                "view_cone_max_unit": "deg",
            },
        ],
    )
    output_path = Path(tmp_test_directory) / "output"

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=output_path,
    )

    for value_spec in PLOT_VALUE_SPECS:
        plotter.plot_altaz_projection_with_color_scale(
            value_key=value_spec.key,
            value_label=value_spec.value_label,
            output_file_stem=value_spec.azimuth_zenith_output_file_stem,
        )
        plotter.plot_zenith_limits_for_azimuths(
            value_key=value_spec.key,
            value_label=value_spec.value_label,
            output_file_stem=value_spec.zenith_profile_output_file_stem,
        )

    for value_spec in PLOT_VALUE_SPECS:
        assert (output_path / f"{value_spec.azimuth_zenith_output_file_stem}.png").exists()
        assert (output_path / f"{value_spec.zenith_profile_output_file_stem}.png").exists()


def test_load_grid_points_file_not_found(tmp_test_directory):
    """Raise when the grid points file does not exist."""
    missing_file = Path(tmp_test_directory) / "does_not_exist.ecsv"

    with pytest.raises(FileNotFoundError, match="Grid points file not found"):
        _create_plotter(
            grid_file=missing_file,
            output_path=Path(tmp_test_directory) / "output",
        )


def test_load_grid_points_wrong_suffix(tmp_test_directory):
    """Raise when the grid points file is not ECSV."""
    wrong_file = Path(tmp_test_directory) / "grid_points.txt"
    wrong_file.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="must be ECSV"):
        _create_plotter(
            grid_file=wrong_file,
            output_path=Path(tmp_test_directory) / "output",
        )


def test_extract_quantity_value_dict_branches():
    """Cover value/lower/None extraction paths."""
    point_with_value = {"x": {"value": 12.3, "unit": "deg"}}
    assert ProductionGridPlotter._extract_quantity_value(point_with_value, "x") == pytest.approx(
        12.3
    )

    point_with_lower = {"x": {"lower": {"value": 7.5, "unit": "deg"}}}
    assert ProductionGridPlotter._extract_quantity_value(point_with_lower, "x") == pytest.approx(
        7.5
    )

    point_without_value = {"x": {"unit": "deg"}}
    assert ProductionGridPlotter._extract_quantity_value(point_without_value, "x") is None


def test_format_value_label_with_unit_uses_available_unit():
    """Append units from *_unit keys when present in normalized points."""
    plot_points = [{"energy_min_unit": "TeV"}]

    formatted_label = ProductionGridPlotter._format_value_label_with_unit(
        plot_points,
        value_key="energy_min",
        value_label="energy_min",
    )

    assert formatted_label == "energy_min [TeV]"


def test_configure_radec_axis_expands_flat_ranges(tmp_test_directory):
    """Expand axis limits by +/-5 deg when computed min and max are equal."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_single_radec.ecsv",
        [{"ra": 40.0, "dec": 20.0}],
    )
    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    figure, axis = plt.subplots()
    try:
        plotter._configure_radec_axis(axis, [{"ra": 40.0, "dec": 20.0}])
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        assert xlim == pytest.approx((45.0, 35.0))
        assert ylim == pytest.approx((15.0, 25.0))
    finally:
        plt.close(figure)


def test_plot_frame_points_logs_no_valid_points(tmp_test_directory, caplog):
    """Log warning and return zero when no points can be plotted."""
    grid_file = _write_grid_file(tmp_test_directory, "grid_empty.ecsv", [])
    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    figure, axis = plt.subplots()
    try:
        with caplog.at_level("WARNING"):
            plotted = plotter._plot_frame_points(
                axis=axis,
                plot_points=[],
                primary_frame="altaz",
                secondary_frame="radec",
                primary_color="tab:blue",
                secondary_color="tab:orange",
                x_key="azimuth",
                y_key="zenith",
                panel_name="Alt/Az",
            )
        assert plotted == 0
        assert "No valid grid points found for Alt/Az plotting" in caplog.text
    finally:
        plt.close(figure)


def test_plot_altaz_points_logs_hidden_radec_points(tmp_test_directory, caplog):
    """Log info when RA/Dec points are not visible in Azimuth/Zenith panel."""
    grid_file = _write_grid_file(tmp_test_directory, "grid_empty_altaz.ecsv", [])
    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1, projection="polar")
    try:
        plot_points = [
            {
                "native_frame": "radec",
                "azimuth": None,
                "zenith": None,
                "ra": 10.0,
                "dec": -80.0,
                "visible_in_altaz": False,
            }
        ]
        with caplog.at_level("INFO"):
            plotter._plot_altaz_points(axis, plot_points)
        assert "Skipping 1 RA/Dec points below the horizon in Azimuth/Zenith panel" in caplog.text
    finally:
        plt.close(figure)


def test_plot_sky_projection_logs_tracks_disabled(tmp_test_directory, caplog):
    """Log that RA/Dec tracks are disabled in file-driven mode."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz_with_radec_for_tracks.ecsv",
        [{"azimuth": 10.0, "zenith_angle": 20.0, "ra": 120.0, "dec": -20.0}],
    )
    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    with caplog.at_level("INFO"):
        plotter.plot_sky_projection(plot_ra_dec_tracks=True)

    assert "RA/Dec tracks are disabled in file-driven plotting mode" in caplog.text


def test_plot_sky_projection_writes_grid_density_subtitle(tmp_test_directory, monkeypatch):
    """Render direction grid density in subtitle when present in metadata."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_with_density_meta.ecsv",
        [{"azimuth": 100.0, "zenith_angle": 25.0, "ra": 180.0, "dec": -10.0}],
    )
    table = Table.read(grid_file, format="ascii.ecsv")
    table.meta["direction_grid_density"] = 0.25
    table.meta["direction_grid_density_unit"] = "1/deg^2"
    table.write(grid_file, format="ascii.ecsv", overwrite=True)

    plotter = _create_plotter(
        grid_file=grid_file,
        output_path=Path(tmp_test_directory) / "output",
    )

    recorded_text = []
    original_text = plt.Figure.text

    def _record_text(self, x, y, s, **kwargs):
        recorded_text.append(s)
        return original_text(self, x, y, s, **kwargs)

    monkeypatch.setattr(plt.Figure, "text", _record_text)

    plotter.plot_sky_projection()

    assert any(text == "Grid density: 0.25 1/deg^2" for text in recorded_text)
