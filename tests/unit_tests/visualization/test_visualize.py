#!/usr/bin/python3

# pylint: disable=protected-access,redefined-outer-name,unused-argument

import logging
from pathlib import Path
from types import SimpleNamespace

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


# Tests for event selection and event image plotting


def _fake_event(dl1_image=None, r1_waveforms=None):
    tel_id = 1
    ev = SimpleNamespace()
    ev.index = SimpleNamespace(obs_id=1, event_id=42)
    ev.trigger = SimpleNamespace(event_type=SimpleNamespace(name="flasher"))
    ev.dl1 = SimpleNamespace(tel={})
    ev.r1 = SimpleNamespace(tel={})
    if dl1_image is not None:
        ev.dl1.tel[tel_id] = SimpleNamespace(image=np.asarray(dl1_image))
    if r1_waveforms is not None:
        ev.r1.tel[tel_id] = SimpleNamespace(waveform=np.asarray(r1_waveforms))
    return ev, tel_id


def _fake_source_with_event(ev, tel_id):
    class _Sub:
        def __init__(self):
            # minimal camera/readout/geometry stubs
            self.tel = {
                tel_id: SimpleNamespace(
                    type=SimpleNamespace(name="LST"),
                    optics=SimpleNamespace(name="LST-Optics"),
                    camera_name="LSTCam",
                    camera=SimpleNamespace(
                        geometry=SimpleNamespace(name="LSTCam"),
                        readout=SimpleNamespace(sampling_rate=None),
                    ),
                )
            }

    class _Src:
        def __init__(self):
            self.subarray = _Sub()
            self._ev = [ev]

        def __iter__(self):
            return iter(self._ev)

    return _Src()


def test__select_event_by_type_returns_first(caplog):
    ev, _ = _fake_event([1, 2, 3])
    src = _fake_source_with_event(ev, 1)
    caplog.clear()
    with caplog.at_level("INFO", logger=visualize._logger.name):  # pylint:disable=protected-access
        out = visualize._select_event_by_type(src, preferred="flasher")  # pylint:disable=protected-access
    assert out is ev
    assert any("filtering ('flasher') not applied" in r.message for r in caplog.records)


def test__select_event_by_type_none_warns(caplog):
    class _Empty:
        def __iter__(self):
            return iter(())

    caplog.clear()
    with caplog.at_level("WARNING", logger=visualize._logger.name):  # pylint:disable=protected-access
        out = visualize._select_event_by_type(_Empty(), preferred=None)  # pylint:disable=protected-access
    assert out is None
    assert any("No events available" in r.message for r in caplog.records)


# Helpers for ctapipe stubs and waveform generation


def _make_waveforms(n_pix=4, n_samp=16):
    w = np.tile(np.arange(n_samp, dtype=float), (n_pix, 1))
    w += np.arange(n_pix)[:, None]
    return w


def _install_fake_ctapipe(monkeypatch, source_obj):
    import sys
    from types import ModuleType

    ctapipe_mod = ModuleType("ctapipe")
    io_mod = ModuleType("io")
    calib_mod = ModuleType("calib")
    vis_mod = ModuleType("visualization")
    image_mod = ModuleType("image")

    class _EventSource:
        def __init__(self, *a, **k):
            self._src = source_obj
            self.subarray = getattr(source_obj, "subarray", None)

        def __iter__(self):
            return iter(self._src)

    class _CameraCalibrator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return None

    class _CameraDisplay:
        def __init__(self, *a, **k):
            self.cmap = None

        def add_colorbar(self, *a, **k):
            pass

        def set_limits_percent(self, *a, **k):
            pass

    def _tailcuts_clean(*a, **k):
        img = a[1]
        return np.ones_like(img, dtype=bool)

    io_mod.EventSource = _EventSource
    calib_mod.CameraCalibrator = _CameraCalibrator
    vis_mod.CameraDisplay = _CameraDisplay
    image_mod.tailcuts_clean = _tailcuts_clean

    ctapipe_mod.io = io_mod
    ctapipe_mod.calib = calib_mod
    ctapipe_mod.visualization = vis_mod
    ctapipe_mod.image = image_mod

    monkeypatch.setitem(sys.modules, "ctapipe", ctapipe_mod)
    monkeypatch.setitem(sys.modules, "ctapipe.io", io_mod)
    monkeypatch.setitem(sys.modules, "ctapipe.calib", calib_mod)
    monkeypatch.setitem(sys.modules, "ctapipe.visualization", vis_mod)
    monkeypatch.setitem(sys.modules, "ctapipe.image", image_mod)


def test_plot_simtel_event_image_returns_figure(monkeypatch):
    ev, tel_id = _fake_event(dl1_image=np.array([1.0, 2.0, 3.0]))
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = visualize.plot_simtel_event_image("dummy.simtel.gz", return_cleaned=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_event_image_no_event(monkeypatch, caplog):
    class _EmptySrc:
        def __iter__(self):
            return iter(())

    # Attach minimal subarray to avoid attribute errors when accessing subarray
    _EmptySrc.subarray = SimpleNamespace(tel={})

    _install_fake_ctapipe(monkeypatch, _EmptySrc())

    caplog.clear()
    with caplog.at_level("WARNING", logger=visualize._logger.name):  # pylint:disable=protected-access
        fig = visualize.plot_simtel_event_image("dummy.simtel.gz")
    assert fig is None
    assert any("No event found" in r.message for r in caplog.records)


def test_plot_simtel_time_traces_returns_figure(monkeypatch):
    w = _make_waveforms(5, 20)
    ev, tel_id = _fake_event(dl1_image=np.arange(w.shape[0]), r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = visualize.plot_simtel_time_traces("dummy.simtel.gz", n_pixels=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_waveform_pcolormesh_returns_figure(monkeypatch):
    w = _make_waveforms(8, 32)
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = visualize.plot_simtel_waveform_pcolormesh("dummy.simtel.gz", pixel_step=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_step_traces_returns_figure(monkeypatch):
    w = _make_waveforms(12, 10)
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = visualize.plot_simtel_step_traces("dummy.simtel.gz", pixel_step=5, max_pixels=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_time_traces_no_waveforms(monkeypatch, caplog):
    ev, tel_id = _fake_event(dl1_image=np.array([0.0, 1.0, 2.0]), r1_waveforms=None)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=visualize._logger.name):  # pylint:disable=protected-access
        fig = visualize.plot_simtel_time_traces("dummy.simtel.gz")
    assert fig is None
    assert any("No R1 waveforms available" in r.message for r in caplog.records)


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
        tel_id=1,
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


def test_plot_simtel_peak_timing_returns_stats(monkeypatch):
    # Build fake scipy.signal
    import sys
    from types import ModuleType

    scipy_mod = ModuleType("scipy")
    signal_mod = ModuleType("signal")

    def _find_peaks(trace, prominence=None):  # pylint:disable=unused-argument
        # return the argmax as the only peak
        peak = int(np.argmax(trace))
        return np.array([peak]), {}

    def _find_peaks_cwt(trace, widths):  # pylint:disable=unused-argument
        # no cwt peaks -> force fallback
        return []

    signal_mod.find_peaks = _find_peaks
    signal_mod.find_peaks_cwt = _find_peaks_cwt
    scipy_mod.signal = signal_mod

    monkeypatch.setitem(sys.modules, "scipy", scipy_mod)
    monkeypatch.setitem(sys.modules, "scipy.signal", signal_mod)

    # Fake event with simple peaked waveforms
    n_pix, n_samp, peak_idx = 6, 20, 7
    w = np.zeros((n_pix, n_samp), dtype=float)
    for i in range(n_pix):
        w[i, peak_idx] = 10 + i
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    monkeypatch.setattr(visualize, "EventSource", lambda *a, **k: src)
