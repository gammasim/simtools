#!/usr/bin/python3

# pylint: disable=protected-access,redefined-outer-name,unused-argument

import importlib
import logging
from types import SimpleNamespace

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from simtools.visualization import simtel_event_plots as sep

logger = logging.getLogger(__name__)
DUMMY_SIMTEL = "DUMMY_SIMTEL"


# Helpers for ctapipe stubs and waveform generation


def _make_waveforms(n_pix=4, n_samp=16):
    w = np.tile(np.arange(n_samp, dtype=float), (n_pix, 1))
    w += np.arange(n_pix)[:, None]
    return w


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
            # Minimal stub for ctapipe.calib.CameraCalibrator.
            pass

        def __call__(self, *a, **k):
            # No-op: simulate a callable calibrator without modifying the event.
            return None

    class _CameraDisplay:
        def __init__(self, *a, **k):
            self.cmap = None

        def add_colorbar(self, *a, **k):
            # Minimal stub: tests don't assert on colorbars
            pass

        def set_limits_percent(self, *a, **k):
            # Minimal stub: emulate API without scaling
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

    # Ensure the production module re-imports ctapipe symbols from our fakes
    importlib.reload(sep)


# Tests migrated from test_visualize to target the new module


def test_plot_simtel_event_image_returns_figure(monkeypatch):
    ev, tel_id = _fake_event(dl1_image=np.array([1.0, 2.0, 3.0]), r1_waveforms=_make_waveforms())
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_event_image(DUMMY_SIMTEL)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    plt.close(fig)


def test_plot_simtel_event_image_missing_r1(monkeypatch, caplog):
    ev, tel_id = _fake_event(dl1_image=np.array([1.0, 2.0, 3.0]), r1_waveforms=None)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):  # pylint:disable=protected-access
        fig = sep.plot_simtel_event_image(DUMMY_SIMTEL)
    assert fig is None
    assert any("First event has no R1 telescope data" in r.message for r in caplog.records)


def test_plot_simtel_event_image_annotations(monkeypatch):
    ev, tel_id = _fake_event(dl1_image=np.array([1.0, 2.0, 3.0]), r1_waveforms=_make_waveforms())
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_event_image(DUMMY_SIMTEL, distance=100 * u.m)
    assert isinstance(fig, plt.Figure)
    assert any("distance: 100.0 m" in a.get_text() for a in fig.axes[0].texts)
    plt.close(fig)


def test_plot_simtel_event_image_no_event(monkeypatch, caplog):
    src = _fake_source_with_event(None, None)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):  # pylint:disable=protected-access
        fig = sep.plot_simtel_event_image(DUMMY_SIMTEL)
    assert fig is None
    assert any("No event found in the file." in r.message for r in caplog.records)


def test_plot_simtel_time_traces_returns_figure(monkeypatch):
    w = _make_waveforms(5, 20)
    ev, tel_id = _fake_event(dl1_image=np.arange(w.shape[0]), r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_time_traces(DUMMY_SIMTEL, n_pixels=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_time_traces_pixel_selection(monkeypatch):
    w = _make_waveforms(10, 20)
    ev, tel_id = _fake_event(dl1_image=np.arange(w.shape[0]), r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_time_traces(DUMMY_SIMTEL, n_pixels=5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_time_traces_with_tel_id(monkeypatch):
    w = _make_waveforms(5, 20)
    ev, tel_id = _fake_event(dl1_image=np.arange(w.shape[0]), r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_time_traces(DUMMY_SIMTEL, tel_id=tel_id, n_pixels=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_time_traces_invalid_tel_id(monkeypatch, caplog):
    w = _make_waveforms(5, 20)
    ev, tel_id = _fake_event(dl1_image=np.arange(w.shape[0]), r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):  # pylint:disable=protected-access
        fig = sep.plot_simtel_time_traces(DUMMY_SIMTEL, tel_id=9999)
    assert fig is None
    assert any("No R1 waveforms available" in r.message for r in caplog.records)


def test_plot_simtel_waveform_matrix_returns_figure(monkeypatch):
    w = _make_waveforms(8, 32)
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_waveform_matrix(DUMMY_SIMTEL, pixel_step=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_step_traces_returns_figure(monkeypatch):
    w = _make_waveforms(12, 10)
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_step_traces(DUMMY_SIMTEL, pixel_step=5, max_pixels=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_time_traces_no_waveforms(monkeypatch, caplog):
    ev, tel_id = _fake_event(dl1_image=np.array([0.0, 1.0, 2.0]), r1_waveforms=None)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):  # pylint:disable=protected-access
        fig = sep.plot_simtel_time_traces(DUMMY_SIMTEL)
    assert fig is None
    assert any("No R1 waveforms available" in r.message for r in caplog.records)


def test__histogram_edges_default_and_binned():
    edges_default = sep._histogram_edges(10, timing_bins=None)
    assert np.allclose(edges_default[:3], [-0.5, 0.5, 1.5])
    assert edges_default.size == 11

    edges_binned = sep._histogram_edges(10, timing_bins=5)
    assert np.isclose(edges_binned[0], -0.5)
    assert np.isclose(edges_binned[-1], 9.5)
    assert edges_binned.size == 6


def test__draw_peak_hist_basic():
    fig, ax = plt.subplots()
    peak_samples = np.array([1, 2, 2, 3, 4, 4, 4])
    edges = np.arange(-0.5, 6.5, 1.0)
    sep._draw_peak_hist(
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
    assert len(ax.containers) >= 1
    x0, x1 = ax.get_xlim()
    assert np.isclose(x0, edges[0])
    assert np.isclose(x1, edges[-1])
    plt.close(fig)


def test_plot_simtel_peak_timing_returns_stats(monkeypatch):
    import sys
    from types import ModuleType

    scipy_mod = ModuleType("scipy")
    signal_mod = ModuleType("signal")

    def _find_peaks(trace, prominence=None):  # pylint:disable=unused-argument
        peak = int(np.argmax(trace))
        return np.array([peak]), {}

    def _find_peaks_cwt(trace, widths):  # pylint:disable=unused-argument
        return []

    signal_mod.find_peaks = _find_peaks
    signal_mod.find_peaks_cwt = _find_peaks_cwt
    scipy_mod.signal = signal_mod

    monkeypatch.setitem(sys.modules, "scipy", scipy_mod)
    monkeypatch.setitem(sys.modules, "scipy.signal", signal_mod)

    # Reload after scipy monkeypatch so the production import uses our stub
    importlib.reload(sep)

    n_pix, n_samp, peak_idx = 6, 20, 7
    w = np.zeros((n_pix, n_samp), dtype=float)
    for i in range(n_pix):
        w[i, peak_idx] = 10 + i
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig, stats = sep.plot_simtel_peak_timing(DUMMY_SIMTEL, return_stats=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(stats, dict)
    assert stats["considered"] == n_pix - 1
    assert stats["found"] == n_pix - 1
    assert np.isclose(stats["mean"], peak_idx)
    assert np.isclose(stats["std"], 0.0)
    plt.close(fig)


def test__detect_peaks_prefers_cwt():
    class _Sig:
        @staticmethod
        def find_peaks_cwt(trace, widths):  # pylint:disable=unused-argument
            return [3, 7]

        @staticmethod
        def find_peaks(trace, prominence=None):  # pylint:disable=unused-argument
            return np.array([5]), {}

    trace = np.zeros(10)
    trace[7] = 1.0
    peaks = sep._detect_peaks(trace, peak_width=4, signal_mod=_Sig)  # pylint:disable=protected-access
    np.testing.assert_array_equal(peaks, np.array([3, 7]))


def test__detect_peaks_fallback_to_find_peaks():
    class _Sig:
        @staticmethod
        def find_peaks_cwt(trace, widths):  # pylint:disable=unused-argument
            return []

        @staticmethod
        def find_peaks(trace, prominence=None):  # pylint:disable=unused-argument
            return np.array([2]), {}

    trace = np.array([0, 0.1, 2.0, 0.5, 0.0])
    peaks = sep._detect_peaks(trace, peak_width=3, signal_mod=_Sig)  # pylint:disable=protected-access
    np.testing.assert_array_equal(peaks, np.array([2]))


def test__detect_peaks_handles_errors():
    class _Sig:
        @staticmethod
        def find_peaks(trace, prominence=None):  # pylint:disable=unused-argument
            raise ValueError("bad fp")

    trace = np.ones(5)
    peaks = sep._detect_peaks(trace, peak_width=2, signal_mod=_Sig)  # pylint:disable=protected-access
    assert peaks.size == 0


def test__collect_peak_samples_basic():
    w = np.array(
        [
            [0, 1, 3, 2, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 2, 5, 1],
        ],
        dtype=float,
    )

    class _Sig:
        @staticmethod
        def find_peaks_cwt(trace, widths):  # pylint:disable=unused-argument
            return []

        @staticmethod
        def find_peaks(trace, prominence=None):  # pylint:disable=unused-argument
            return np.array([int(np.argmax(trace))]), {}

    peak_samples, pix_ids, found = sep._collect_peak_samples(  # pylint:disable=protected-access
        w, sum_threshold=5.0, peak_width=3, signal_mod=_Sig
    )
    np.testing.assert_array_equal(pix_ids, np.array([0, 2]))
    np.testing.assert_array_equal(peak_samples, np.array([2, 3]))
    assert found == 2


def test__collect_peak_samples_threshold_excludes_all():
    w = np.ones((2, 4), dtype=float)

    class _Sig:
        @staticmethod
        def find_peaks_cwt(trace, widths):  # pylint:disable=unused-argument
            return []

        @staticmethod
        def find_peaks(trace, prominence=None):  # pylint:disable=unused-argument
            return np.array([0]), {}

    peak_samples, pix_ids, found = sep._collect_peak_samples(  # pylint:disable=protected-access
        w, sum_threshold=10.0, peak_width=3, signal_mod=_Sig
    )
    assert peak_samples is None
    assert pix_ids is None
    assert found == 0


def test_plot_simtel_integrated_signal_image_returns_figure(monkeypatch):
    w = _make_waveforms(5, 16)
    w[0, 8] += 10
    w[1, 9] += 12
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_integrated_signal_image(DUMMY_SIMTEL, half_width=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_integrated_pedestal_image_returns_figure(monkeypatch):
    w = _make_waveforms(4, 20)
    w[2, 10] += 15
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_integrated_pedestal_image(DUMMY_SIMTEL, half_width=2, gap=5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test__time_axis_from_readout_valid_and_errors():  # pylint:disable=protected-access
    class R:
        def __init__(self, sr):
            self.sampling_rate = sr

    # Valid sampling rate: 1 GHz -> 1 ns steps
    t = sep._time_axis_from_readout(R(1 * u.GHz), 4)
    np.testing.assert_array_equal(t, np.array([0.0, 1.0, 2.0, 3.0]))

    # None sampling rate -> default dt=1.0
    t = sep._time_axis_from_readout(R(None), 3)
    np.testing.assert_array_equal(t, np.array([0.0, 1.0, 2.0]))

    # Zero division path -> default dt=1.0
    t = sep._time_axis_from_readout(R(0 * u.Hz), 2)
    np.testing.assert_array_equal(t, np.array([0.0, 1.0]))


def test__select_event_by_type_first_and_index_and_oob(caplog):  # pylint:disable=protected-access
    evs = ["e0", "e1", "e2"]
    selector = sep._select_event_by_type(evs)
    assert selector() == "e0"
    assert selector(event_index=1) == "e1"
    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):
        assert selector(event_index=99) is None
    assert any("out of range" in r.message for r in caplog.records)


def test_plot_simtel_waveform_matrix_no_r1(monkeypatch, caplog):
    # Event without R1 data
    ev, tel_id = _fake_event(dl1_image=np.array([1.0, 2.0, 3.0]), r1_waveforms=None)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):
        fig = sep.plot_simtel_waveform_matrix(DUMMY_SIMTEL)
    assert fig is None
    assert any("no R1 data for waveform plot" in r.message for r in caplog.records)


def test_plot_simtel_step_traces_no_r1(monkeypatch, caplog):
    ev, tel_id = _fake_event(dl1_image=np.array([0.0, 1.0, 2.0]), r1_waveforms=None)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):
        fig = sep.plot_simtel_step_traces(DUMMY_SIMTEL)
    assert fig is None
    assert any("no R1 data for traces plot" in r.message for r in caplog.records)


def test_plot_simtel_peak_timing_threshold_excludes_all(monkeypatch, caplog):
    w = np.ones((3, 5), dtype=float)  # sums are 5 each
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    caplog.clear()
    with caplog.at_level("WARNING", logger=sep._logger.name):
        fig = sep.plot_simtel_peak_timing(DUMMY_SIMTEL, sum_threshold=10.0)
    assert fig is None
    assert any("sum_threshold" in r.message for r in caplog.records)


def test_plot_simtel_time_traces_calibrator_error(monkeypatch):
    # Prepare event with waveforms but force calibrator to raise, so image=None path is used
    w = _make_waveforms(6, 12)
    ev, tel_id = _fake_event(dl1_image=np.arange(w.shape[0]), r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    class _CalibErr:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            raise ValueError("calib failed")

    # Monkeypatch the symbol used inside sep
    monkeypatch.setattr(sep, "CameraCalibrator", _CalibErr)

    fig = sep.plot_simtel_time_traces(DUMMY_SIMTEL, n_pixels=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_event_image_distance_float(monkeypatch):
    ev, tel_id = _fake_event(dl1_image=np.array([1.0, 2.0, 3.0]), r1_waveforms=_make_waveforms())
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_event_image(DUMMY_SIMTEL, distance=123.4)
    assert isinstance(fig, plt.Figure)
    assert any("distance: 123.4" in a.get_text() for a in fig.axes[0].texts)
    plt.close(fig)


def test_plot_simtel_waveform_matrix_defaults(monkeypatch):
    # Cover pixel_step=None branch
    w = _make_waveforms(5, 10)
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_waveform_matrix(DUMMY_SIMTEL, pixel_step=None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_simtel_step_traces_defaults(monkeypatch):
    # Cover max_pixels=None branch
    w = _make_waveforms(7, 9)
    ev, tel_id = _fake_event(r1_waveforms=w)
    src = _fake_source_with_event(ev, tel_id)

    _install_fake_ctapipe(monkeypatch, src)

    fig = sep.plot_simtel_step_traces(DUMMY_SIMTEL, pixel_step=3, max_pixels=None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test__histogram_edges_zero_bins():  # pylint:disable=protected-access
    edges = sep._histogram_edges(5, timing_bins=0)
    np.testing.assert_array_equal(edges, np.arange(-0.5, 5.5, 1.0))
