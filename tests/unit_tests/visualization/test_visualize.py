#!/usr/bin/python3

# pylint: disable=protected-access,redefined-outer-name,unused-argument

import logging
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

import simtools.utils.general as gen
from simtools.visualization import visualize

logger = logging.getLogger(__name__)


@pytest.fixture
def wavelength():
    return "Wavelength [nm]"


def test_plot_1d(db, io_handler, wavelength):
    logger.debug("Testing plot_1d")

    x_title = wavelength
    y_title = "Mirror reflectivity [%]"
    headers_type = {"names": (x_title, y_title), "formats": ("f8", "f8")}
    title = "Test 1D plot"

    test_file_name = "ref_LST1_2022_04_01.dat"
    db.export_model_files(
        db_name=None,
        dest=io_handler.get_output_directory(sub_dir="model"),
        file_names=test_file_name,
    )
    test_data_file = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(sub_dir="model"),
    )
    data_in = np.loadtxt(test_data_file, usecols=(0, 1), dtype=headers_type)

    # Change y-axis to percent
    if "%" in y_title:
        if np.max(data_in[y_title]) <= 1:
            data_in[y_title] = 100 * data_in[y_title]
    data = {}
    data["Reflectivity"] = data_in
    for i in range(5):
        new_data = np.copy(data_in)
        new_data[y_title] = new_data[y_title] * (1 - 0.1 * (i + 1))
        data[f"{100 * (1 - 0.1 * (i + 1))}%% reflectivity"] = new_data

    fig = visualize.plot_1d(data, title=title, palette="autumn")

    plot_file = io_handler.get_output_file(file_name="plot_1d.pdf", sub_dir="plots")
    if plot_file.exists():
        plot_file.unlink()
    fig.savefig(plot_file)

    logger.debug(f"Produced 1D plot ({plot_file}).")

    assert plot_file.exists()


def test_plot_table(io_handler):
    logger.debug("Testing plot_table")

    title = "Test plot table"
    table = astropy.io.ascii.read("tests/resources/Transmission_Spectrum_PlexiGlass.dat")

    fig = visualize.plot_table(table, y_title="Transmission", title=title, no_markers=True)

    plot_file = io_handler.get_output_file(file_name="plot_table.pdf", sub_dir="plots")
    if plot_file.exists():
        plot_file.unlink()
    fig.savefig(plot_file)

    logger.debug(f"Produced 1D plot ({plot_file}).")

    assert plot_file.exists()


def test_add_unit(caplog, wavelength):
    value_with_unit = [30, 40] << u.nm
    assert visualize._add_unit("Wavelength", value_with_unit) == wavelength
    value_without_unit = [30, 40]
    assert visualize._add_unit("Wavelength", value_without_unit) == "Wavelength"

    with caplog.at_level(logging.WARNING):
        assert visualize._add_unit(wavelength, value_with_unit)
    assert "Tried to add a unit from astropy.unit" in caplog.text

    value_with_unit = [30, 40] * u.cm**2
    assert visualize._add_unit("Area", value_with_unit) == "Area [$cm^2$]"


def test_save_figure(io_handler):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("Test Figure")

    output_file = io_handler.get_output_file(file_name="test_save_figure", sub_dir="plots")
    figure_formats = ["pdf", "png"]

    visualize.save_figure(fig, output_file, figure_format=figure_formats, log_title="Test Figure")

    for fmt in figure_formats:
        file_path = Path(output_file).with_suffix(f".{fmt}")
        assert file_path.exists()
        logger.debug(f"Saved plot to {file_path}")

    plt.close(fig)


def test_plot_error_plots():
    """Test the _plot_error_plots function for both error types."""
    x = np.array([1, 2, 3])
    y = np.array([10, 20, 30])
    y_err = np.array([1, 2, 1])
    x_err = np.array([0.1, 0.2, 0.1])

    data_y_err = np.zeros(3, dtype=[("x", float), ("y", float), ("y_err", float)])
    data_y_err["x"] = x
    data_y_err["y"] = y
    data_y_err["y_err"] = y_err

    data_xy_err = np.zeros(
        3, dtype=[("x", float), ("y", float), ("x_err", float), ("y_err", float)]
    )
    data_xy_err["x"] = x
    data_xy_err["y"] = y
    data_xy_err["x_err"] = x_err
    data_xy_err["y_err"] = y_err

    fig1, ax1 = plt.subplots()
    kwargs_fill = {"error_type": "fill_between"}
    visualize._plot_error_plots(kwargs_fill, data_y_err, "x", "y", None, "y_err", "blue")
    assert len(ax1.collections) > 0
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    kwargs_errorbar = {"error_type": "errorbar"}
    visualize._plot_error_plots(kwargs_errorbar, data_xy_err, "x", "y", "x_err", "y_err", "red")
    assert len(ax2.containers) > 0
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    kwargs_none = {}
    visualize._plot_error_plots(kwargs_none, data_y_err, "x", "y", None, "y_err", "green")
    assert len(ax3.collections) == 0
    assert len(ax3.containers) == 0
    plt.close(fig3)


def test_get_data_columns():
    """Test the _get_data_columns function with different column configurations."""
    # Test with 2 columns
    data_2col = np.zeros(3, dtype=[("x", float), ("y", float)])
    x_col, y_col, x_err_col, y_err_col = visualize._get_data_columns(data_2col)
    assert x_col == "x"
    assert y_col == "y"
    assert x_err_col is None
    assert y_err_col is None

    # Test with 3 columns (y error)
    data_3col = np.zeros(3, dtype=[("x", float), ("y", float), ("y_err", float)])
    x_col, y_col, x_err_col, y_err_col = visualize._get_data_columns(data_3col)
    assert x_col == "x"
    assert y_col == "y"
    assert x_err_col is None
    assert y_err_col == "y_err"

    # Test with 4 columns (x and y errors)
    data_4col = np.zeros(3, dtype=[("x", float), ("y", float), ("x_err", float), ("y_err", float)])
    x_col, y_col, x_err_col, y_err_col = visualize._get_data_columns(data_4col)
    assert x_col == "x"
    assert y_col == "y"
    assert x_err_col == "x_err"
    assert y_err_col == "y_err"

    # Test the assertion for minimum columns
    data_1col = np.zeros(3, dtype=[("x", float)])
    with pytest.raises(
        AssertionError, match="Input array must have at least two columns with titles."
    ):
        visualize._get_data_columns(data_1col)


def test_plot_ratio_difference():
    """Test the plot_ratio_difference function for both ratio and difference plots."""
    # Create test data
    x = np.array([1, 2, 3])
    y1 = np.array([10, 20, 30])
    y2 = np.array([15, 25, 35])
    y3 = np.array([12, 22, 32])

    # Create structured arrays
    dtype = [("x", float), ("y", float)]
    data1 = np.zeros(3, dtype=dtype)
    data1["x"] = x
    data1["y"] = y1

    data2 = np.zeros(3, dtype=dtype)
    data2["x"] = x
    data2["y"] = y2

    data3 = np.zeros(3, dtype=dtype)
    data3["x"] = x
    data3["y"] = y3

    data_dict = {"reference": data1, "test1": data2, "test2": data3}

    # Test ratio plot
    fig1 = plt.figure()
    gs1 = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs1[0])
    plot_args = {"marker": "o"}

    visualize.plot_ratio_difference(ax1, data_dict, True, gs1, plot_args)

    ratio_ax = plt.gcf().axes[-1]
    assert len(ratio_ax.lines) == 3  # One empty line for cycler + two ratio lines
    # Check ratio values for first dataset
    expected_ratio = y2 / y1
    np.testing.assert_array_almost_equal(ratio_ax.lines[1].get_ydata(), expected_ratio)
    plt.close(fig1)

    # Test difference plot
    fig2 = plt.figure()
    gs2 = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax2 = plt.subplot(gs2[0])

    visualize.plot_ratio_difference(ax2, data_dict, False, gs2, plot_args)

    diff_ax = plt.gcf().axes[-1]
    assert len(diff_ax.lines) == 3  # One empty line for cycler + two difference lines
    # Check difference values for first dataset
    expected_diff = y2 - y1
    np.testing.assert_array_almost_equal(diff_ax.lines[1].get_ydata(), expected_diff)
    plt.close(fig2)

    # Test long reference name handling
    data_dict_long = {"very_long_reference_name_that_exceeds_twenty_chars": data1, "test1": data2}

    fig3 = plt.figure()
    gs3 = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax3 = plt.subplot(gs3[0])

    visualize.plot_ratio_difference(ax3, data_dict_long, True, gs3, plot_args)

    ratio_ax = plt.gcf().axes[-1]
    assert ratio_ax.get_ylabel() == "Ratio"  # Should be shortened
    plt.close(fig3)

    # Test y-axis bins
    fig4 = plt.figure()
    gs4 = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax4 = plt.subplot(gs4[0])

    visualize.plot_ratio_difference(ax4, data_dict, True, gs4, plot_args)

    ratio_ax = plt.gcf().axes[-1]
    yticks = len(ratio_ax.get_yticks())
    assert yticks <= 7
    plt.close(fig4)


def test__histogram_edges_default_and_binned():
    edges_default = visualize._histogram_edges(10, timing_bins=None)
    assert np.allclose(edges_default[:3], [-0.5, 0.5, 1.5])
    # For n_samp=10, edges go from -0.5 to 9.5 in steps of 1.0 -> 11 edges
    assert edges_default.size == 11

    edges_binned = visualize._histogram_edges(10, timing_bins=5)
    # 5 bins -> 6 edges spanning -0.5 .. 9.5
    assert np.isclose(edges_binned[0], -0.5)
    assert np.isclose(edges_binned[-1], 9.5)
    assert edges_binned.size == 6


def test__draw_peak_hist_basic():
    fig, ax = plt.subplots()
    peak_samples = np.array([1, 2, 2, 3, 4, 4, 4])
    edges = np.arange(-0.5, 6.5, 1.0)
    visualize._draw_peak_hist(
        ax,
        peak_samples,
        edges,
        mean_sample=3.0,
        std_sample=1.0,
        tel_label="CT1",
        et_name="flasher",
        considered=7,
        found_count=6,
    )
    # Bars added
    assert len(ax.containers) >= 1
    # Limits set to edge bounds
    x0, x1 = ax.get_xlim()
    assert np.isclose(x0, edges[0])
    assert np.isclose(x1, edges[-1])
    plt.close(fig)
