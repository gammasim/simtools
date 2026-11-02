from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from simtools.histogram.histogram import Histogram
from simtools.visualization import plot_simtel_event_histograms
from simtools.visualization.plot_simtel_event_histograms import (
    _add_lines,
    _build_plot_config,
    _build_plot_filename,
    _create_2d_histogram_plot,
    _create_plot,
    _format_file_info_suffix,
    _get_broad_range_axis_limits,
    _get_limits,
    _slice_values,
    plot,
    plot_monte_carlo_statistics_diagnostics,
    plot_trigger_histograms,
)

# Suppress tight layout warnings for all tests in this file
pytestmark = pytest.mark.filterwarnings("ignore:Tight layout not applied.*:UserWarning")

# Common constants
MOD = "simtools.visualization.plot_simtel_event_histograms"
PATCH_SUBPLOTS = f"{MOD}.plt.subplots"
PATCH_SHOW = f"{MOD}.plt.show"
PATCH_CLOSE = f"{MOD}.plt.close"
PATCH_PCOLOR = f"{MOD}.plt.pcolormesh"
PATCH_CONTOUR = f"{MOD}.plt.contour"
PATCH_COLORBAR = f"{MOD}.plt.colorbar"
PATCH_CREATE_PLOT = f"{MOD}._create_plot"
PATCH_BUILD_FILENAME = f"{MOD}._build_plot_filename"

EVENT_COUNT = "Event Count"
TRIGGERED = "Triggered events"
ENERGY_LABEL = "Energy [TeV]"
CORE_DIST_LABEL = "Core Distance [m]"
CORE_X_LABEL = "Core X [m]"
CORE_Y_LABEL = "Core Y [m]"
POINTING_LABEL = "Distance to pointing direction [deg]"


@pytest.fixture(autouse=True)
def suppress_plot_window(mocker):
    """Prevent interactive windows when tests exercise the display-only path."""
    return mocker.patch(PATCH_SHOW)


@pytest.fixture
def sample_data():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bins_x = np.array([0, 1, 2, 3])
    bins_y = np.array([0, 1, 2, 3])
    return data, (bins_x, bins_y)


def test_create_2d_histogram_plot_linear(sample_data):
    data, bins = sample_data
    plot_params = {"norm": "linear", "cmap": "viridis", "show_contour": True}

    fig, _ = plt.subplots()
    pcm = _create_2d_histogram_plot(data, bins, plot_params)

    assert pcm is not None
    assert pcm.get_cmap().name == "viridis"
    assert pcm.get_clim() == pytest.approx((1.0, 9.0))
    plt.close(fig)


def test_plot_monte_carlo_statistics_diagnostics_writes_expected_plots(tmp_test_directory, mocker):
    mock_save = mocker.patch(f"{MOD}.save_figure")

    plot_monte_carlo_statistics_diagnostics(
        tmp_test_directory,
        "MSTS-01",
        {"zenith": 20.0 * u.deg, "azimuth": 180.0 * u.deg, "nsb_level": 1.0},
        np.array([0.1, 1.0, 10.0]),
        np.array([0.0, 0.5]),
        np.array([[10.0, 20.0]]),
        np.array([[0.3, 0.2]]),
    )

    assert mock_save.call_count == 2
    output_files = [call.args[1] for call in mock_save.call_args_list]
    assert tmp_test_directory / "MSTS-01_z20_az180_nsb1_expected_events" in output_files
    assert tmp_test_directory / "MSTS-01_z20_az180_nsb1_relative_uncertainty" in output_files


def test_plot_monte_carlo_statistics_diagnostics_can_mask_zero_bins(tmp_test_directory, mocker):
    mock_mesh = mocker.MagicMock()
    fig, ax = plt.subplots()
    ax.pcolormesh = mocker.MagicMock(return_value=mock_mesh)
    mocker.patch(f"{MOD}.plt.subplots", return_value=(fig, ax))

    plot_monte_carlo_statistics_diagnostics(
        tmp_test_directory,
        "test",
        None,
        np.array([0.1, 1.0, 10.0]),
        np.array([0.0, 0.5]),
        np.array([[0.0, 2.0]]),
        np.array([[0.3, 0.2]]),
    )

    plotted_array = ax.pcolormesh.call_args_list[0].args[2]
    cmap = ax.pcolormesh.call_args_list[0].kwargs["cmap"]
    assert np.ma.isMaskedArray(plotted_array)
    assert np.any(np.ma.getmaskarray(plotted_array))
    assert cmap.get_bad()[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert (tmp_test_directory / "test_expected_events.png").exists()


def test_create_2d_histogram_plot_log(sample_data):
    data, bins = sample_data
    plot_params = {"norm": "log", "cmap": "viridis"}

    fig, _ = plt.subplots()
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

    fig, _ = plt.subplots()
    pcm = _create_2d_histogram_plot(data, bins, plot_params)

    assert pcm is not None
    assert pcm.get_clim() == (0, 1)
    plt.close(fig)


@pytest.mark.parametrize(
    ("lines", "expect_lines", "expect_circles"),
    [
        ({"x": 1, "y": 2}, False, 0),
        ({"r": 3}, False, 1),
        ({"curve": {"x": [1, 2], "y": [3, 4]}}, True, 0),
        ({}, False, 0),
    ],
)
def test_add_lines(lines, expect_lines, expect_circles):
    fig, ax = plt.subplots()
    plot_simtel_event_histograms._add_lines(ax, lines)
    if expect_lines:
        if "curve" in lines:
            plotted = ax.get_lines()[-1]
            np.testing.assert_array_equal(plotted.get_xdata(), np.array([1, 2]))
            np.testing.assert_array_equal(plotted.get_ydata(), np.array([3, 4]))
            assert plotted.get_color() == "r"
            assert plotted.get_linestyle() == "--"
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


def test_create_plot_with_output_file(tmp_test_directory):
    data = np.array([1, 2, 3])
    bins = np.array([0, 1, 2, 3])
    plot_params = {"color": "blue"}
    labels = {"x": "X-axis", "y": "Y-axis", "title": "Save Plot"}
    scales = {"x": "linear", "y": "linear"}
    output_file = Path(tmp_test_directory) / "saved_plot.png"

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
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_create_plot_early_return_when_no_data():
    with patch(PATCH_SUBPLOTS) as mock_subplots:
        result = _create_plot(data=None, bins=None, plot_type="histogram")
        assert result is None
        mock_subplots.assert_not_called()


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (("test_plot",), {}, "test_plot.png"),
        (("test_plot", "array1"), {}, "test_plot_array1.png"),
        (
            (
                "angular_distance",
                "MyArray",
                {"zenith": 20.0 * u.deg, "azimuth": 0.0 * u.deg, "nsb_level": 0.3},
            ),
            {},
            "angular_distance_MyArray_z20_az0_nsb0.3.png",
        ),
        (("plot",), {"file_info": {"zenith": 52.0 * u.deg}}, "plot_z52.png"),
    ],
)
def test_build_plot_filename(args, kwargs, expected):
    assert _build_plot_filename(*args, **kwargs) == expected


def test_format_file_info_suffix_empty():
    assert _format_file_info_suffix({}) == ""


def test_build_plot_config_2d():
    histogram = Histogram.from_edges(
        [np.array([0, 1, 2]), np.array([0, 1, 2])],
        values=np.array([[1, 2], [3, 4]]),
        axis_names=["core_distance", "energy"],
        axis_units=["m", "TeV"],
    )
    result, filename = _build_plot_config("core_distance_vs_energy", histogram, None)

    np.testing.assert_array_equal(result["data"], np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(result["bins"][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(result["bins"][1], np.array([0, 1, 2]))
    assert result["plot_type"] == "histogram2d"
    assert result["plot_params"] == {"norm": "log", "cmap": "viridis"}
    assert result["labels"]["x"] == "Core Distance (m)"
    assert result["labels"]["y"] == "Energy (TeV)"
    assert result["labels"]["title"] == "Triggered Events: core distance vs energy"
    assert result["colorbar_label"] == EVENT_COUNT
    assert result["lines"] == {}
    assert filename == "core_distance_vs_energy_triggered"


def test_build_plot_config_1d():
    histogram = Histogram.from_edges(
        [np.array([0, 1, 2, 3])],
        values=np.array([10, 20, 30]),
        axis_names=["energy"],
        axis_units=["TeV"],
    )
    result, filename = _build_plot_config("energy", histogram, None)

    assert np.array_equal(result["data"], np.array([10, 20, 30]))
    assert np.array_equal(result["bins"], np.array([0, 1, 2, 3]))
    assert result["plot_type"] == "histogram"
    assert result["labels"]["x"] == "Energy (TeV)"
    assert result["labels"]["y"] == EVENT_COUNT
    assert result["labels"]["title"] == "Triggered Events: energy"
    assert result["scales"] == {"x": "log", "y": "log"}
    assert result["lines"] == {}
    assert filename == "energy_triggered"


def test_plot_with_output_path(tmp_test_directory, mocker):
    mock_create_plot = mocker.patch(f"{MOD}._create_plot")
    histograms = {
        "energy": Histogram.from_edges(
            [np.array([0.1, 1.0, 10.0])], values=[1, 2], axis_names=["energy"], axis_units=["TeV"]
        )
    }

    plot(histograms, output_path=Path(tmp_test_directory), array_name="test_array")

    assert mock_create_plot.call_args.kwargs["output_file"] == (
        Path(tmp_test_directory) / "energy_triggered_test_array.png"
    )


def test_build_plot_config_norm_selection():
    histogram_2d = Histogram.from_edges([[0, 1], [0, 1]])
    expected_norms = {
        "core_distance_vs_energy": "log",
        "energy_cumulative": "linear",
        "core_distance_vs_energy_eff": "linear",
        "reuse_mean_vs_core_distance_vs_energy": "linear",
    }
    for name, expected_norm in expected_norms.items():
        config, _ = _build_plot_config(name, histogram_2d, None)
        assert config["plot_params"]["norm"] == expected_norm

    histogram_1d = Histogram.from_edges(
        [np.array([0.1, 1.0, 10.0])], axis_names=["energy"], axis_units=["TeV"]
    )
    config, _ = _build_plot_config("reuse_mean_vs_energy", histogram_1d, None)
    assert config["scales"]["y"] == "linear"


def test_histogram_style_table_preserves_plot_configuration():
    axis = np.array([0.1, 1.0, 10.0])
    histograms = {
        "energy": Histogram.from_edges(
            [axis], values=[1, 2], axis_names=["energy"], axis_units=["TeV"]
        ),
        "energy_mc": Histogram.from_edges(
            [axis], values=[2, 4], axis_names=["energy"], axis_units=["TeV"]
        ),
        "energy_eff": Histogram.from_edges(
            [axis], values=[0.5, 0.5], axis_names=["energy"], axis_units=["TeV"]
        ),
        "energy_cumulative": Histogram.from_edges(
            [axis], values=[0.5, 1.0], axis_names=["energy"], axis_units=["TeV"]
        ),
        "reuse_mean_vs_energy": Histogram.from_edges(
            [axis], values=[2.0, 1.0], axis_names=["energy"], axis_units=["TeV"]
        ),
        "reuse_mean_vs_core_distance_vs_energy": Histogram.from_edges(
            [np.array([0, 1, 2]), axis],
            values=np.ones((2, 2)),
            axis_names=["core_distance", "energy"],
            axis_units=["m", "TeV"],
        ),
    }

    plots = {name: _build_plot_config(name, h, None)[0] for name, h in histograms.items()}

    assert plots["energy"]["labels"] == {
        "x": "Energy (TeV)",
        "y": "Event Count",
        "title": "Triggered Events: energy",
    }
    assert plots["energy"]["scales"] == {"x": "log", "y": "log"}
    assert plots["energy_mc"]["labels"]["title"] == "Simulated Events: energy mc"
    assert plots["energy_eff"]["labels"]["y"] == "Efficiency"
    assert plots["energy_cumulative"]["labels"]["y"] == "Fraction of Events"
    assert plots["reuse_mean_vs_energy"]["labels"]["y"] == "Mean Reuse"
    assert plots["reuse_mean_vs_energy"]["scales"] == {"x": "log", "y": "linear"}
    assert plots["reuse_mean_vs_core_distance_vs_energy"]["scales"] == {
        "x": "linear",
        "y": "log",
    }


def test_broad_range_axis_limits_do_not_replace_derived_limits():
    limits = {
        "lower_energy_limit": 0.1 * u.TeV,
        "upper_radius_limit": 1200.0 * u.m,
        "viewcone_radius": 2.5 * u.deg,
        "br_energy_min": 0.02 * u.TeV,
        "br_energy_max": 200.0 * u.TeV,
        "br_core_scatter_max": 1800.0 * u.m,
        "br_viewcone_max": 4.0 * u.deg,
    }

    axis_limits = _get_broad_range_axis_limits("core_distance_vs_energy", limits)

    assert axis_limits == {"x": (0.0, 1800.0), "y": (0.02, 200.0)}
    assert _get_limits("core_distance_vs_energy", limits) == {
        "x": 1200.0,
        "y": 0.1,
        "curve": None,
    }


def test_broad_range_axis_limits_expand_for_coincident_energy_limit(tmp_test_directory):
    limits = {
        "lower_energy_limit": 0.02 * u.TeV,
        "br_energy_min": 0.02 * u.TeV,
        "br_energy_max": 200.0 * u.TeV,
        "br_core_scatter_max": 1800.0 * u.m,
        "br_viewcone_max": 4.0 * u.deg,
    }

    axis_limits = plot_simtel_event_histograms._get_broad_range_axis_limits(
        "core_distance_vs_energy", limits
    )

    assert axis_limits["x"] == pytest.approx((0.0, 1800.0))
    assert axis_limits["y"] == pytest.approx((0.018, 200.0))
    fig = _create_plot(
        data=np.ones((2, 2)),
        bins=[np.array([0.0, 100.0, 200.0]), np.array([0.01, 1.0, 10.0])],
        plot_type="histogram2d",
        plot_params={"norm": "log", "cmap": "viridis"},
        labels={"title": "Triggered"},
        scales={"y": "log"},
        output_file=None,
        lines={"y": 0.02},
        axis_limits=axis_limits,
        projection_kind="core_distance",
    )

    assert fig.axes[0].get_ylim()[0] == pytest.approx(0.018)
    assert fig.axes[2].get_xlim()[0] == pytest.approx(0.018)


def test_projection_slice_coordinates():
    energy_slices = _slice_values(np.logspace(-2, 3, 6), (0.02, 200.0))
    core_slices = _slice_values(np.linspace(0.0, 2000.0, 5), (0.0, 1800.0), "core_distance")
    angular_slices = _slice_values(np.linspace(0.0, 5.0, 6), (0.0, 4.0), "angular_distance")

    assert [np.log10(value) for value, _ in energy_slices] == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert [value for value, _ in core_slices] == [0.0, 500.0, 1000.0, 1500.0]
    assert [value for value, _ in angular_slices] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_create_2d_plot_with_distance_projections():
    data = np.arange(1.0, 21.0).reshape(4, 5)
    bins = [np.linspace(0.0, 2000.0, 5), np.logspace(-2, 3, 6)]

    fig = _create_plot(
        data=data,
        bins=bins,
        plot_type="histogram2d",
        plot_params={"norm": "log", "cmap": "viridis"},
        labels={"x": "Core distance (m)", "y": "Energy (TeV)", "title": "Triggered"},
        scales={"y": "log"},
        colorbar_label="Event count",
        output_file=None,
        lines={"x": 1200.0, "y": 0.1},
        axis_limits={"x": (0.0, 1800.0), "y": (0.02, 200.0)},
        projection_kind="core_distance",
    )

    assert len(fig.axes) >= 3
    assert fig.axes[0].get_xlim() == pytest.approx((0.0, 1800.0))
    assert fig.axes[0].get_ylim() == pytest.approx((0.02, 200.0))
    assert [line.get_label() for line in fig.axes[1].lines] == [
        "overall",
        "log10(E/TeV)=-1.5",
        "log10(E/TeV)=-0.5",
        "log10(E/TeV)=0.5",
        "log10(E/TeV)=1.5",
        "_child5",
    ]
    assert [line.get_label() for line in fig.axes[2].lines] == [
        "overall",
        "0 m",
        "500 m",
        "1000 m",
        "1500 m",
        "_child5",
    ]
    assert fig.axes[1].lines[-1].get_xdata() == pytest.approx([1200.0, 1200.0])
    assert fig.axes[1].lines[-1].get_color() == "r"
    assert fig.axes[1].lines[-1].get_linestyle() == "--"
    assert fig.axes[2].lines[-1].get_xdata() == pytest.approx([0.1, 0.1])
    assert fig.axes[2].lines[-1].get_color() == "r"
    assert fig.axes[2].lines[-1].get_linestyle() == "--"
    assert fig.axes[1].get_legend()._loc == 1
    assert fig.axes[2].get_legend()._loc == 1


def test_create_2d_plot_with_angular_distance_projection_limits():
    data = np.arange(1.0, 21.0).reshape(4, 5)
    bins = [np.linspace(0.0, 4.0, 5), np.logspace(-2, 3, 6)]

    fig = _create_plot(
        data=data,
        bins=bins,
        plot_type="histogram2d",
        plot_params={"norm": "log", "cmap": "viridis"},
        labels={"x": "Angular distance (deg)", "y": "Energy (TeV)", "title": "Triggered"},
        scales={"y": "log"},
        colorbar_label="Event count",
        output_file=None,
        lines={"x": 2.5, "y": 0.1},
        axis_limits={"x": (0.0, 4.0), "y": (0.02, 200.0)},
        projection_kind="angular_distance",
    )

    assert fig.axes[1].lines[-1].get_xdata() == pytest.approx([2.5, 2.5])
    assert fig.axes[1].lines[-1].get_color() == "r"
    assert fig.axes[1].lines[-1].get_linestyle() == "--"
    assert fig.axes[2].lines[-1].get_xdata() == pytest.approx([0.1, 0.1])
    assert fig.axes[2].lines[-1].get_color() == "r"
    assert fig.axes[2].lines[-1].get_linestyle() == "--"


def test_create_2d_plot_renders_single_point_limit_curve():
    data = np.ones((2, 2))
    bins = [np.array([0.0, 100.0, 200.0]), np.array([0.1, 1.0, 10.0])]

    fig = _create_plot(
        data=data,
        bins=bins,
        plot_type="histogram2d",
        plot_params={"norm": "log", "cmap": "viridis"},
        labels={"title": "Triggered"},
        scales={"y": "log"},
        output_file=None,
        lines={"curve": {"x": [120.0], "y": [1.0]}},
    )

    curve = fig.axes[0].lines[-1]
    assert curve.get_color() == "r"
    assert curve.get_linestyle() == "--"
    assert curve.get_marker() == "o"
    assert curve.get_xdata() == pytest.approx([120.0])
    assert curve.get_ydata() == pytest.approx([1.0])


def test_add_lines_handles_multi_point_and_malformed_limit_curves():
    fig, ax = plt.subplots()

    with plt.rc_context({"lines.marker": "o"}):
        _add_lines(ax, {"curve": {"x": [100.0, 200.0], "y": [1.0, 2.0]}})
        _add_lines(ax, {"curve": {"x": [100.0], "y": [1.0, 2.0]}})

    assert len(ax.lines) == 1
    assert ax.lines[0].get_marker() == "None"
    plt.close(fig)


def test_plot_uses_array_name_in_title():
    histogram = Histogram.from_edges([[0, 1]])
    histogram.fill(np.array([0.5]))

    with patch(f"{MOD}._create_plot") as mock_create_plot:
        plot({"energy": histogram}, array_name="CTAO-Test")

    assert mock_create_plot.call_args.kwargs["labels"]["title"] == (
        "Triggered Events: energy (CTAO-Test)"
    )


def test_plot_trigger_histograms_renders_real_histogram(tmp_test_directory):
    histogram = Histogram.from_edges([[0, 1]])
    histogram.fill(np.array([0.5]))
    loaded = [(None, SimpleNamespace(array_name="alpha", histograms={"energy": histogram}))]
    output_directory = Path(tmp_test_directory)

    plot_trigger_histograms(loaded, output_directory)

    output_files = list((output_directory / "alpha").glob("*.png"))
    assert output_files
