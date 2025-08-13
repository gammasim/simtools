from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from simtools.visualization import plot_simtel_event_histograms
from simtools.visualization.plot_simtel_event_histograms import (
    _build_plot_filename,
    _create_2d_histogram_plot,
    _create_plot,
    _create_rebinned_plot,
    _execute_plotting_loop,
    _get_limits,
    plot,
)

# Common constants
MOD = "simtools.visualization.plot_simtel_event_histograms"
PATCH_SUBPLOTS = f"{MOD}.plt.subplots"
PATCH_SHOW = f"{MOD}.plt.show"
PATCH_CLOSE = f"{MOD}.plt.close"
PATCH_PCOLOR = f"{MOD}.plt.pcolormesh"
PATCH_CONTOUR = f"{MOD}.plt.contour"
PATCH_COLORBAR = f"{MOD}.plt.colorbar"
PATCH_CREATE_PLOT = f"{MOD}._create_plot"
PATCH_CREATE_REBINNED = f"{MOD}._create_rebinned_plot"
PATCH_REBIN = f"{MOD}.SimtelIOEventHistograms.rebin_2d_histogram"
PATCH_HAS_DATA = f"{MOD}._has_data"
PATCH_BUILD_FILENAME = f"{MOD}._build_plot_filename"

EVENT_COUNT = "Event Count"
TRIGGERED = "Triggered events"
ENERGY_LABEL = "Energy [TeV]"
CORE_DIST_LABEL = "Core Distance [m]"
CORE_X_LABEL = "Core X [m]"
CORE_Y_LABEL = "Core Y [m]"
POINTING_LABEL = "Distance to pointing direction [deg]"


@pytest.fixture
def sample_data():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bins_x = np.array([0, 1, 2, 3])
    bins_y = np.array([0, 1, 2, 3])
    return data, (bins_x, bins_y)


def test_create_2d_histogram_plot_linear(sample_data):
    data, bins = sample_data
    plot_params = {"norm": "linear", "cmap": "viridis", "show_contour": True}

    fig, ax = plt.subplots()
    pcm = _create_2d_histogram_plot(data, bins, plot_params)

    assert pcm is not None
    assert pcm.get_cmap().name == "viridis"
    plt.close(fig)


def test_create_2d_histogram_plot_log(sample_data):
    data, bins = sample_data
    plot_params = {"norm": "log", "cmap": "viridis"}

    fig, ax = plt.subplots()
    pcm = _create_2d_histogram_plot(data, bins, plot_params)

    assert pcm is not None
    assert isinstance(pcm.norm, LogNorm)
    plt.close(fig)


def test_create_2d_histogram_plot_no_positive_data():
    data = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    bins_x = np.array([0, 1, 2, 3])
    bins_y = np.array([0, 1, 2, 3])
    bins = (bins_x, bins_y)
    plot_params = {"norm": "log", "cmap": "viridis"}

    fig, ax = plt.subplots()
    pcm = _create_2d_histogram_plot(data, bins, plot_params)

    assert pcm is not None
    assert pcm.get_clim() == (0, 1)
    plt.close(fig)


@pytest.mark.parametrize(
    ("lines", "expect_lines", "expect_circles"),
    [
        ({"x": 1, "y": 2}, True, 0),
        ({"r": 3}, False, 1),
        ({}, False, 0),
    ],
)
def test_add_lines(lines, expect_lines, expect_circles):
    fig, ax = plt.subplots()
    plot_simtel_event_histograms._add_lines(ax, lines)
    if expect_lines:
        if "x" in lines:
            assert any(line.get_xdata() == [lines["x"], lines["x"]] for line in ax.get_lines())
        if "y" in lines:
            assert any(line.get_ydata() == [lines["y"], lines["y"]] for line in ax.get_lines())
    else:
        if not lines or ("x" not in lines and "y" not in lines):
            remaining = [ln for ln in ax.get_lines() if ln.get_label() == "_nolegend_"]
            assert len(ax.get_lines()) in (0, len(remaining))
    circles = [artist for artist in ax.get_children() if isinstance(artist, plt.Circle)]
    if expect_circles:
        assert len(circles) == expect_circles
        assert circles[0].get_radius() == lines["r"]
    else:
        assert len(circles) == 0
    plt.close(fig)


def test_plot_data_histogram():
    from unittest.mock import MagicMock

    data = np.array([1, 2, 3])
    bins = np.array([0, 1, 2, 3])
    plot_params = {"color": "blue"}
    colorbar_label = None

    fig, ax = plt.subplots()
    ax.bar = MagicMock()

    plot_simtel_event_histograms._plot_data(
        ax, data, bins, "histogram", plot_params, colorbar_label
    )

    assert ax.bar.call_count == 1
    call_args, call_kwargs = ax.bar.call_args
    # First positional arg: left edges; second positional: heights
    np.testing.assert_array_equal(call_args[0], bins[:-1])
    np.testing.assert_array_equal(call_args[1], data)
    # Width passed as keyword via numpy diff
    np.testing.assert_array_equal(call_kwargs["width"], np.diff(bins))
    assert call_kwargs["color"] == "blue"
    plt.close(fig)


def test_plot_data_histogram2d(sample_data):
    from unittest.mock import MagicMock

    data, bins = sample_data
    plot_params = {"norm": "linear", "cmap": "viridis", "show_contour": True}
    colorbar_label = "Event Count"

    fig, ax = plt.subplots()
    fake_pcm = object()

    # Patch internal 2D creator and colorbar
    original_create = plot_simtel_event_histograms._create_2d_histogram_plot
    plot_simtel_event_histograms._create_2d_histogram_plot = MagicMock(return_value=fake_pcm)
    original_colorbar = plt.colorbar
    plt.colorbar = MagicMock()

    try:
        plot_simtel_event_histograms._plot_data(
            ax, data, bins, "histogram2d", plot_params, colorbar_label
        )
        # Assert internal helper called
        plot_simtel_event_histograms._create_2d_histogram_plot.assert_called_once()
        # Assert colorbar called with returned pcm and label
        plt.colorbar.assert_called_once()
        cb_args, cb_kwargs = plt.colorbar.call_args
        assert cb_args[0] is fake_pcm
        assert cb_kwargs["label"] == colorbar_label
    finally:
        plot_simtel_event_histograms._create_2d_histogram_plot = original_create
        plt.colorbar = original_colorbar
        plt.close(fig)


def test_plot_data_invalid_plot_type():
    from unittest.mock import MagicMock

    data = np.array([1, 2, 3])
    bins = np.array([0, 1, 2, 3])
    plot_params = {"color": "blue"}
    colorbar_label = None

    fig, ax = plt.subplots()
    ax.bar = MagicMock()
    original_create = plot_simtel_event_histograms._create_2d_histogram_plot
    plot_simtel_event_histograms._create_2d_histogram_plot = MagicMock()
    original_colorbar = plt.colorbar
    plt.colorbar = MagicMock()

    try:
        plot_simtel_event_histograms._plot_data(
            ax, data, bins, "invalid_type", plot_params, colorbar_label
        )
        # No supported plot type: ensure no plotting helpers called
        ax.bar.assert_not_called()
        plot_simtel_event_histograms._create_2d_histogram_plot.assert_not_called()
        plt.colorbar.assert_not_called()
    finally:
        plot_simtel_event_histograms._create_2d_histogram_plot = original_create
        plt.colorbar = original_colorbar
        plt.close(fig)


def test_has_data():
    # Test with None data
    assert not plot_simtel_event_histograms._has_data(None)

    # Test with empty numpy array
    empty_array = np.array([])
    assert not plot_simtel_event_histograms._has_data(empty_array)

    # Test with non-empty numpy array
    non_empty_array = np.array([1, 2, 3])
    assert plot_simtel_event_histograms._has_data(non_empty_array)


def test_create_plot():
    data = np.array([1, 2, 3])
    bins = np.array([0, 1, 2, 3])
    plot_params = {"color": "blue"}
    test_plot_title = "Test Plot"
    labels = {"x": "X-axis", "y": "Y-axis", "title": test_plot_title}
    scales = {"x": "linear", "y": "log"}
    colorbar_label = "Colorbar"
    output_file = None
    lines = {"x": 1, "y": 2}

    with (
        patch(PATCH_SUBPLOTS) as mock_subplots,
        patch(PATCH_SHOW) as mock_show,
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = _create_plot(
            data=data,
            bins=bins,
            plot_type="histogram",
            plot_params=plot_params,
            labels=labels,
            scales=scales,
            colorbar_label=colorbar_label,
            output_file=output_file,
            lines=lines,
        )

        assert result is not None
        mock_subplots.assert_called_once()
        # Manually check bar call with numpy arrays
        assert mock_ax.bar.call_count == 1
        bar_args, bar_kwargs = mock_ax.bar.call_args
        np.testing.assert_array_equal(bar_args[0], bins[:-1])
        np.testing.assert_array_equal(bar_args[1], data)
        np.testing.assert_array_equal(bar_kwargs["width"], np.diff(bins))
        assert bar_kwargs["color"] == plot_params["color"]
        mock_ax.axvline.assert_called_once_with(1, color="r", linestyle="--", linewidth=0.5)
        mock_ax.axhline.assert_called_once_with(2, color="r", linestyle="--", linewidth=0.5)
        mock_ax.set.assert_called_once_with(
            xlabel="X-axis",
            ylabel="Y-axis",
            title=test_plot_title,
            xscale="linear",
            yscale="log",
        )
        mock_show.assert_called_once()


def test_create_plot_with_output_file(tmp_path):
    data = np.array([1, 2, 3])
    bins = np.array([0, 1, 2, 3])
    plot_params = {"color": "blue"}
    labels = {"x": "X-axis", "y": "Y-axis", "title": "Save Plot"}
    scales = {"x": "linear", "y": "linear"}
    output_file = tmp_path / "saved_plot.png"

    with (
        patch(PATCH_SUBPLOTS) as mock_subplots,
        patch(PATCH_SHOW) as mock_show,
        patch(PATCH_CLOSE) as mock_close,
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = _create_plot(
            data=data,
            bins=bins,
            plot_type="histogram",
            plot_params=plot_params,
            labels=labels,
            scales=scales,
            colorbar_label=None,
            output_file=output_file,
            lines={},
        )

        assert result is not None
        mock_fig.savefig.assert_called_once()
        # Ensure file path argument matches expected name
        save_args, save_kwargs = mock_fig.savefig.call_args
        assert str(output_file) in str(save_args[0])
        mock_close.assert_called_once_with(mock_fig)
        mock_show.assert_not_called()


def test_create_plot_histogram2d_colorbar():
    data = np.array([[1, 2], [3, 4]])
    bins = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    plot_params = {"norm": "linear", "cmap": "viridis", "show_contour": True}
    labels = {"x": "X", "y": "Y", "title": "2D"}

    with (
        patch(PATCH_SUBPLOTS) as mock_subplots,
        patch(PATCH_PCOLOR) as mock_pcolor,
        patch(PATCH_CONTOUR) as mock_contour,
        patch(PATCH_COLORBAR) as mock_colorbar,
        patch(PATCH_SHOW) as mock_show,
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = _create_plot(
            data=data,
            bins=bins,
            plot_type="histogram2d",
            plot_params=plot_params,
            labels=labels,
            scales=None,
            colorbar_label="Count",
            output_file=None,
            lines=None,
        )

    assert result is mock_fig
    # Ensure internal 2d creation helpers used
    mock_pcolor.assert_called_once()
    mock_contour.assert_called_once()
    mock_colorbar.assert_called_once()
    mock_show.assert_called_once()
    mock_fig.savefig.assert_not_called()


def test_create_plot_early_return_when_no_data():
    # Simple: force _has_data to return None (falsy) so _create_plot exits early
    with (
        patch(PATCH_HAS_DATA, return_value=None) as mock_has_data,
        patch(PATCH_SUBPLOTS) as mock_subplots,
    ):
        from simtools.visualization.plot_simtel_event_histograms import _create_plot

        result = _create_plot(data=None, bins=None, plot_type="histogram")
        assert result is None
        mock_has_data.assert_called_once()
        mock_subplots.assert_not_called()


@pytest.mark.parametrize("output_path_is_none", [True, False])
def test_create_rebinned_plot(output_path_is_none, tmp_path):
    data = np.array([[1, 2], [3, 4]])
    bins = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    rebin_factor = 2
    plot_args = {
        "data": data,
        "bins": bins,
        "plot_type": "histogram2d",
        "plot_params": {"norm": "linear", "cmap": "viridis"},
        "labels": {"title": "Original Plot"},
    }
    filename = "test_plot.png"
    output_path = None if output_path_is_none else tmp_path

    rebinned_data = np.array([[10]])
    rebinned_x_bins = np.array([0, 2])
    rebinned_y_bins = np.array([0, 2])

    with (
        patch(
            PATCH_REBIN,
            return_value=(rebinned_data, rebinned_x_bins, rebinned_y_bins),
        ) as mock_rebin,
        patch(PATCH_CREATE_PLOT) as mock_create_plot,
    ):
        _create_rebinned_plot(plot_args, filename, output_path, rebin_factor)

        mock_rebin.assert_called_once_with(data, bins[0], bins[1], rebin_factor)
        mock_create_plot.assert_called_once()
        rebinned_plot_args = mock_create_plot.call_args[1]
        np.testing.assert_array_equal(rebinned_plot_args["data"], rebinned_data)
        if output_path is None:
            assert rebinned_plot_args["output_file"] is None
        else:
            assert rebinned_plot_args["output_file"].name == "test_plot_rebinned.png"


def test_should_create_rebinned_plot():
    plot_args = {
        "plot_type": "histogram2d",
        "plot_params": {"norm": "linear"},
    }

    # Test case: rebin_factor > 1, plot_type is histogram2d, ends with _cumulative, norm is linear
    assert plot_simtel_event_histograms._should_create_rebinned_plot(
        rebin_factor=2, plot_args=plot_args, plot_key="test_cumulative"
    )

    # Test case: rebin_factor <= 1
    assert not plot_simtel_event_histograms._should_create_rebinned_plot(
        rebin_factor=1, plot_args=plot_args, plot_key="test_cumulative"
    )

    # Test case: plot_type is not histogram2d
    plot_args["plot_type"] = "histogram"
    assert not plot_simtel_event_histograms._should_create_rebinned_plot(
        rebin_factor=2, plot_args=plot_args, plot_key="test_cumulative"
    )

    # Test case: plot_key does not end with _cumulative
    plot_args["plot_type"] = "histogram2d"
    assert not plot_simtel_event_histograms._should_create_rebinned_plot(
        rebin_factor=2, plot_args=plot_args, plot_key="test"
    )

    # Test case: norm is not linear
    plot_args["plot_params"]["norm"] = "log"
    assert not plot_simtel_event_histograms._should_create_rebinned_plot(
        rebin_factor=2, plot_args=plot_args, plot_key="test_cumulative"
    )


def test_build_plot_filename():
    # Test without array_name
    base_filename = "test_plot"
    result = _build_plot_filename(base_filename)
    assert result == "test_plot.png"

    # Test with array_name
    array_name = "array1"
    result = _build_plot_filename(base_filename, array_name)
    assert result == "test_plot_array1.png"


def test_execute_plotting_loop():
    plots = {
        "plot1": {
            "data": np.array([1, 2, 3]),
            "bins": np.array([0, 1, 2, 3]),
            "plot_type": "histogram",
            "plot_params": {"color": "blue"},
            "labels": {"title": "Test Plot"},
            "scales": {"x": "linear", "y": "log"},
            "filename": "test_plot",
        },
        "plot2": {
            "data": None,  # This plot should be skipped
            "bins": np.array([0, 1, 2, 3]),
            "plot_type": "histogram",
            "plot_params": {"color": "red"},
            "labels": {"title": "Skipped Plot"},
            "scales": {"x": "linear", "y": "log"},
            "filename": "skipped_plot",
        },
    }
    output_path = MagicMock()
    rebin_factor = 2
    array_name = "test_array"

    with (
        patch(PATCH_CREATE_PLOT) as mock_create_plot,
        patch(PATCH_CREATE_REBINNED) as mock_create_rebinned_plot,
        patch(PATCH_BUILD_FILENAME, side_effect=lambda base, array: f"{base}_{array}.png"),
    ):
        mock_create_plot.return_value = MagicMock()  # Simulate successful plot creation

        _execute_plotting_loop(plots, output_path, rebin_factor, array_name)

        # Ensure exactly one plot was created
        assert mock_create_plot.call_count == 1
        call_args, call_kwargs = mock_create_plot.call_args
    # All params passed as kwargs
    assert call_args == ()
    np.testing.assert_array_equal(call_kwargs["data"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(call_kwargs["bins"], np.array([0, 1, 2, 3]))
    assert call_kwargs["plot_type"] == "histogram"
    assert call_kwargs["plot_params"] == {"color": "blue"}
    assert call_kwargs["labels"]["title"] == "Test Plot (test_array array)"
    assert call_kwargs["scales"] == {"x": "linear", "y": "log"}
    # Optional parameters not provided should not appear explicitly
    assert "colorbar_label" not in call_kwargs
    assert "lines" not in call_kwargs
    # output_file is a MagicMock path object; just ensure it's not None
    assert call_kwargs["output_file"] is not None

    # Ensure the second plot was skipped (still only one call)
    mock_create_plot.assert_called_once()

    # For histogram (not 2D cumulative), no rebinned plot expected
    mock_create_rebinned_plot.assert_not_called()


def test_execute_plotting_loop_rebin_and_failed_plot():
    """Cover branches where a plot returns None and where a rebinned plot is created."""
    plots = {
        # Meets all conditions for rebin (2D cumulative, linear norm, rebin_factor > 1)
        "plotA_cumulative": {
            "data": np.array([[1, 2], [3, 4]]),
            "bins": (np.array([0, 1, 2]), np.array([0, 1, 2])),
            "plot_type": "histogram2d",
            "plot_params": {"norm": "linear"},
            "labels": {"title": "Rebin Test"},
            "scales": {},
            "filename": "plotA_cumulative",
        },
        # This plot will return None from _create_plot to exercise the early-continue branch
        "plotB": {
            "data": np.array([1, 2, 3]),
            "bins": np.array([0, 1, 2, 3]),
            "plot_type": "histogram",
            "plot_params": {"color": "blue"},
            "labels": {"title": "Should Skip"},
            "scales": {},
            "filename": "plotB",
        },
    }
    output_path = MagicMock()
    rebin_factor = 2
    array_name = None

    with (
        patch(PATCH_CREATE_PLOT, side_effect=[MagicMock(), None]) as mock_create_plot,
        patch(PATCH_CREATE_REBINNED) as mock_create_rebinned_plot,
        patch(PATCH_BUILD_FILENAME, side_effect=lambda base, array: f"{base}.png"),
    ):
        _execute_plotting_loop(plots, output_path, rebin_factor, array_name)

        # First call produced a figure, second returned None
        assert mock_create_plot.call_count == 2
        # Rebinned plot should be created exactly once for plotA_cumulative
        mock_create_rebinned_plot.assert_called_once()
        args, kwargs = mock_create_rebinned_plot.call_args
        # args: (plot_args_dict, filename, output_path, rebin_factor)
        assert args[1] == "plotA_cumulative.png"
        assert args[2] is output_path
        assert args[3] == rebin_factor


def test_create_2d_cumulative_plot_config():
    cumulative_data = {"normalized_cumulative_core_vs_energy": np.array([[0.1, 0.2], [0.3, 0.4]])}
    histograms = MagicMock()
    histograms.get.side_effect = lambda key: {
        "core_vs_energy_bin_x_edges": np.array([0, 1, 2]),
        "core_vs_energy_bin_y_edges": np.array([0, 1, 2]),
    }.get(key)

    config = {
        "base_key": "core_vs_energy",
        "x_label": "Core Distance [m]",
        "y_label": "Energy [TeV]",
        "plot_params": {"norm": "linear", "cmap": "viridis", "show_contour": True},
        "lines": {"x": 1, "y": 0.5},
        "colorbar_label": "Fraction of events",
        "event_type": "Triggered events",
    }

    result = plot_simtel_event_histograms._create_2d_cumulative_plot_config(
        histograms, config, cumulative_data
    )

    assert result["data"] is cumulative_data["normalized_cumulative_core_vs_energy"]
    np.testing.assert_array_equal(result["bins"][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(result["bins"][1], np.array([0, 1, 2]))
    assert result["plot_type"] == "histogram2d"
    assert result["plot_params"] == config["plot_params"]
    assert result["labels"]["x"] == config["x_label"]
    assert result["labels"]["y"] == config["y_label"]
    assert result["labels"]["title"] == "Triggered events: fraction of events by core vs energy"
    assert result["lines"] == config["lines"]
    assert result["scales"] == {}
    assert result["colorbar_label"] == config["colorbar_label"]
    assert result["filename"] == "core_vs_energy_cumulative_distribution"


def test_create_2d_plot_config():
    histograms = MagicMock()
    histograms.get.side_effect = lambda key: {
        "core_vs_energy": np.array([[1, 2], [3, 4]]),
        "core_vs_energy_bin_x_edges": np.array([0, 1, 2]),
        "core_vs_energy_bin_y_edges": np.array([0, 1, 2]),
    }.get(key)

    config = {
        "base_key": "core_vs_energy",
        "x_label": "Core Distance [m]",
        "y_label": "Energy [TeV]",
        "plot_params": {"norm": "log", "cmap": "viridis"},
        "lines": {"x": 1, "y": 0.5},
        "scales": {"y": "log"},
        "colorbar_label": "Event Count",
        "event_type": "Triggered events",
    }

    result = plot_simtel_event_histograms._create_2d_plot_config(histograms, config, is_mc=False)

    # Compare array contents (histograms.get creates new arrays each call)
    np.testing.assert_array_equal(result["data"], np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(result["bins"][0], histograms.get("core_vs_energy_bin_x_edges"))
    np.testing.assert_array_equal(result["bins"][1], histograms.get("core_vs_energy_bin_y_edges"))
    assert result["plot_type"] == "histogram2d"
    assert result["plot_params"] == config["plot_params"]
    assert result["labels"]["x"] == config["x_label"]
    assert result["labels"]["y"] == config["y_label"]
    assert result["labels"]["title"] == "Triggered events: core vs energy"
    assert result["lines"] == config["lines"]
    assert result["scales"] == config["scales"]
    assert result["colorbar_label"] == config["colorbar_label"]
    assert result["filename"] == "core_vs_energy_distribution"


def test_create_2d_plot_config_core_xy():
    """Cover the core_xy special-case branch in _create_2d_plot_config (line 270)."""
    histograms = MagicMock()
    histograms.get.side_effect = lambda key: {
        "core_xy": np.array([[1, 0], [0, 1]]),
        "core_xy_bin_x_edges": np.array([0, 1, 2]),
        "core_xy_bin_y_edges": np.array([0, 1, 2]),
    }.get(key)

    config = {
        "base_key": "core_xy",
        "x_label": "Core X [m]",
        "y_label": "Core Y [m]",
        "plot_params": {"norm": "log", "cmap": "viridis"},
        "lines": {},
        "scales": {},
        "colorbar_label": "Event Count",
        "event_type": "Triggered events",
    }

    result = plot_simtel_event_histograms._create_2d_plot_config(histograms, config, is_mc=False)

    # Branch-specific expectation
    assert result["labels"]["title"] == "Triggered events: core x vs core y"
    assert result["data"] is not None
    np.testing.assert_array_equal(result["bins"][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(result["bins"][1], np.array([0, 1, 2]))


def test_generate_2d_plots():
    histograms = MagicMock()
    cumulative_data = {
        "normalized_cumulative_core_vs_energy": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "normalized_cumulative_angular_distance_vs_energy": np.array([[0.5, 0.6], [0.7, 0.8]]),
    }
    labels = {
        "core_distance": CORE_DIST_LABEL,
        "energy": ENERGY_LABEL,
        "pointing_direction": POINTING_LABEL,
        "event_count": EVENT_COUNT,
        "core_x": CORE_X_LABEL,
        "core_y": CORE_Y_LABEL,
    }
    limits = {
        "upper_radius_limit": MagicMock(value=100),
        "lower_energy_limit": MagicMock(value=0.1),
        "viewcone_radius": MagicMock(value=5),
    }

    plots = plot_simtel_event_histograms._generate_2d_plots(
        histograms, cumulative_data, labels, limits
    )

    # Check that the expected keys are in the plots dictionary
    expected_keys = [
        "core_vs_energy",
        "core_vs_energy_mc",
        "core_vs_energy_cumulative",
        "angular_distance_vs_energy",
        "angular_distance_vs_energy_mc",
        "angular_distance_vs_energy_cumulative",
        "x_core_shower_vs_y_core_shower",
        "x_core_shower_vs_y_core_shower_mc",
    ]
    assert all(key in plots for key in expected_keys)

    # Check one of the plot configurations
    core_vs_energy_plot = plots["core_vs_energy"]
    assert core_vs_energy_plot["plot_type"] == "histogram2d"
    assert core_vs_energy_plot["labels"]["x"] == CORE_DIST_LABEL
    assert core_vs_energy_plot["labels"]["y"] == ENERGY_LABEL
    assert core_vs_energy_plot["lines"]["x"] == 100
    assert core_vs_energy_plot["lines"]["y"] == 0.1


def test_create_1d_cumulative_plot_config():
    histograms = MagicMock()
    cumulative_data = {"cumulative_energy": np.array([0.1, 0.3, 0.6, 1.0])}
    histograms.histograms.get.side_effect = lambda key: {
        "energy_bin_edges": np.array([0.1, 1, 10, 100, 1000])
    }.get(key)

    config = {
        "base_key": "energy_distribution",
        "cumulative_key": "cumulative_energy",
        "bin_key": "energy_bin_edges",
        "x_label": "Energy [TeV]",
        "title_base": "energy distribution",
        "scales": {"x": "log", "y": "log"},
        "line_key": "x",
        "line_value": 1.0,
    }
    plot_params = {"color": "blue"}
    event_count_label = EVENT_COUNT
    cumulative_prefix = "Cumulative "

    result = plot_simtel_event_histograms._create_1d_cumulative_plot_config(
        histograms, config, plot_params, cumulative_data, event_count_label, cumulative_prefix
    )

    assert np.array_equal(result["data"], cumulative_data["cumulative_energy"])
    assert np.array_equal(result["bins"], histograms.histograms.get("energy_bin_edges"))
    assert result["plot_type"] == "histogram"
    assert result["plot_params"] == plot_params
    assert result["labels"]["x"] == ENERGY_LABEL
    assert result["labels"]["y"] == "Cumulative Event Count"
    assert result["labels"]["title"] == "Triggered events: cumulative energy distribution"
    assert result["scales"] == {"x": "log", "y": "log"}
    assert result["lines"] == {"x": 1.0}
    assert result["filename"] == "energy_distribution_cumulative"


def test_create_1d_plot_config():
    histograms = MagicMock()
    histograms.get.side_effect = lambda key: {
        "energy": np.array([10, 20, 30]),
        "energy_bin_edges": np.array([0, 1, 2, 3]),
    }.get(key)

    config = {
        "base_key": "energy_distribution",
        "histogram_key": "energy",
        "bin_key": "energy_bin_edges",
        "x_label": "Energy [TeV]",
        "title_base": "energy distribution",
        "scales": {"x": "log", "y": "log"},
        "line_key": "x",
        "line_value": 1.0,
    }
    plot_params = {"color": "tab:green", "edgecolor": "tab:green", "lw": 1}
    event_count_label = EVENT_COUNT

    result = plot_simtel_event_histograms._create_1d_plot_config(
        histograms, config, plot_params, event_count_label, is_mc=False
    )

    assert np.array_equal(result["data"], np.array([10, 20, 30]))
    assert np.array_equal(result["bins"], np.array([0, 1, 2, 3]))
    assert result["plot_type"] == "histogram"
    assert result["plot_params"] == plot_params
    assert result["labels"]["x"] == ENERGY_LABEL
    assert result["labels"]["y"] == "Event Count"
    assert result["labels"]["title"] == "Triggered events: energy distribution"
    assert result["scales"] == {"x": "log", "y": "log"}
    assert result["lines"] == {"x": 1.0}
    assert result["filename"] == "energy_distribution"


def test_generate_plot_configurations():
    histograms = MagicMock()
    cumulative_data = {
        "cumulative_energy": np.array([0.1, 0.2, 0.3]),
        "cumulative_core_distance": np.array([0.4, 0.5, 0.6]),
        "cumulative_angular_distance": np.array([0.7, 0.8, 0.9]),
        "normalized_cumulative_core_vs_energy": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "normalized_cumulative_angular_distance_vs_energy": np.array([[0.2, 0.3], [0.4, 0.5]]),
    }
    histograms.get.side_effect = lambda key: {
        "energy": np.array([1, 2, 3]),
        "energy_bin_edges": np.array([0, 1, 2, 3]),
        "core_distance": np.array([4, 5, 6]),
        "core_distance_bin_edges": np.array([0, 1, 2, 3]),
        "angular_distance": np.array([7, 8, 9]),
        "angular_distance_bin_edges": np.array([0, 1, 2, 3]),
        "core_vs_energy": np.array([[1, 2], [3, 4]]),
        "core_vs_energy_bin_x_edges": np.array([0, 1, 2]),
        "core_vs_energy_bin_y_edges": np.array([0, 1, 2]),
        "angular_distance_vs_energy": np.array([[5, 6], [7, 8]]),
        "angular_distance_vs_energy_bin_x_edges": np.array([0, 1, 2]),
        "angular_distance_vs_energy_bin_y_edges": np.array([0, 1, 2]),
    }.get(key)

    limits = {
        "upper_radius_limit": MagicMock(value=100),
        "lower_energy_limit": MagicMock(value=0.1),
        "viewcone_radius": MagicMock(value=5),
    }

    plots = plot_simtel_event_histograms._generate_plot_configurations(
        histograms, cumulative_data, limits
    )

    assert "energy_distribution" in plots
    assert "core_distance" in plots
    assert "angular_distance" in plots
    assert "core_vs_energy" in plots
    assert "core_vs_energy_cumulative" in plots

    energy_plot = plots["energy_distribution"]
    assert energy_plot["data"] is not None
    assert energy_plot["bins"] is not None
    assert energy_plot["labels"]["x"] == "Energy [TeV]"
    assert energy_plot["labels"]["y"] == "Event Count"
    assert energy_plot["scales"]["x"] == "log"
    assert energy_plot["scales"]["y"] == "log"

    core_vs_energy_plot = plots["core_vs_energy"]
    assert core_vs_energy_plot["data"] is not None
    assert core_vs_energy_plot["bins"][0] is not None
    assert core_vs_energy_plot["bins"][1] is not None
    assert core_vs_energy_plot["labels"]["x"] == "Core Distance [m]"
    assert core_vs_energy_plot["labels"]["y"] == "Energy [TeV]"
    assert core_vs_energy_plot["scales"]["y"] == "log"


def test_get_limits():
    # Test with no limits provided
    limits = None
    result = _get_limits(limits)
    assert result == (None, None, None)

    # Test with partial limits
    limits = {"upper_radius_limit": MagicMock(value=100)}
    result = _get_limits(limits)
    assert result == (100, None, None)

    # Test with all limits provided
    limits = {
        "upper_radius_limit": MagicMock(value=100),
        "lower_energy_limit": MagicMock(value=0.1),
        "viewcone_radius": MagicMock(value=5),
    }
    result = _get_limits(limits)
    assert result == (100, 0.1, 5)


@pytest.fixture
def mock_histograms():
    histograms = MagicMock()
    histograms.calculate_cumulative_data.return_value = {"mock_key": "mock_value"}
    return histograms


def test_plot_with_output_path(mock_histograms):
    output_path = Path("/mock/output/path")
    limits = {"upper_radius_limit": MagicMock(value=100)}
    rebin_factor = 2
    array_name = "test_array"

    with (
        patch(f"{MOD}._generate_plot_configurations") as mock_generate_configs,
        patch(f"{MOD}._execute_plotting_loop") as mock_execute_loop,
    ):
        mock_generate_configs.return_value = {"mock_plot": "mock_config"}

        plot(
            histograms=mock_histograms,
            output_path=output_path,
            limits=limits,
            rebin_factor=rebin_factor,
            array_name=array_name,
        )

        mock_generate_configs.assert_called_once_with(
            mock_histograms, {"mock_key": "mock_value"}, limits
        )
        mock_execute_loop.assert_called_once_with(
            {"mock_plot": "mock_config"}, output_path, rebin_factor, array_name
        )


def test_plot_without_output_path(mock_histograms):
    limits = None
    rebin_factor = 1
    array_name = None

    with (
        patch(f"{MOD}._generate_plot_configurations") as mock_generate_configs,
        patch(f"{MOD}._execute_plotting_loop") as mock_execute_loop,
    ):
        mock_generate_configs.return_value = {"mock_plot": "mock_config"}

        plot(
            histograms=mock_histograms,
            output_path=None,
            limits=limits,
            rebin_factor=rebin_factor,
            array_name=array_name,
        )

        mock_generate_configs.assert_called_once_with(
            mock_histograms, {"mock_key": "mock_value"}, limits
        )
        mock_execute_loop.assert_called_once_with(
            {"mock_plot": "mock_config"}, None, rebin_factor, array_name
        )
