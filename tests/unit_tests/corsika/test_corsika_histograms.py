#!/usr/bin/python3

from pathlib import Path

import boost_histogram as bh
import numpy as np
import pytest
from astropy import units as u

from simtools.corsika.corsika_histograms import CorsikaHistograms


# Common test fixtures and helpers
@pytest.fixture
def photon_dtype():
    """Standard photon bunch data type."""
    return [
        ("x", "f8"),
        ("y", "f8"),
        ("cx", "f8"),
        ("cy", "f8"),
        ("time", "f8"),
        ("zem", "f8"),
        ("photons", "f8"),
        ("wavelength", "f8"),
    ]


@pytest.fixture
def dummy_photon(photon_dtype):
    """Standard dummy photon bunch."""
    return np.array([(10.0, 20.0, 0.1, 0.2, 5.0, 100.0, 1.0, 400.0)], dtype=photon_dtype)


@pytest.fixture
def telescope_positions():
    """Standard telescope positions array."""
    return np.array([(0.0, 0.0)], dtype=[("x", "f8"), ("y", "f8")])


@pytest.fixture
def event_dtype():
    """Standard event data type."""
    return [("azimuth_deg", "f8"), ("zenith_deg", "f8"), ("num_photons", "f8")]


def create_dummy_event(pid, energy, az, ze, photon_bunches=None):
    """Create a dummy event with given parameters."""

    class DummyEvent:
        def __init__(self):
            self.header = {
                "particle_id": pid,
                "total_energy": energy,
                "azimuth": az,
                "zenith": ze,
            }
            if photon_bunches is not None:
                self.photon_bunches = photon_bunches

    return DummyEvent()


def create_dummy_iact_file(events, telescope_positions_data=None):
    """Create a dummy IACT file class with given events and positions."""

    class DummyIACTFile:
        def __init__(self, *args, **kwargs):
            if telescope_positions_data is None:
                self.telescope_positions = [(0.0, 0.0)]
            else:
                self.telescope_positions = telescope_positions_data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Dummy context manager does not require cleanup,
            pass

        def __iter__(self):
            return iter(events)

    return DummyIACTFile


def test_update_distributions_runs(monkeypatch):
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.input_file = Path("dummy")
    ch.hist = ch._set_2d_distributions()
    ch.hist.update(ch._set_1d_distributions())
    monkeypatch.setattr(ch, "_populate_density_from_probes", lambda: None)
    monkeypatch.setattr(ch, "_populate_density_from_counts", lambda: None)
    monkeypatch.setattr(ch, "_filter_density_histograms", lambda: None)
    monkeypatch.setattr(
        ch,
        "get_hist_1d_projection",
        lambda key, value: (np.zeros((1, 10)), np.arange(11).reshape(1, -1), None),
    )
    monkeypatch.setattr(
        ch,
        "get_hist_2d_projection",
        lambda hist: (
            np.zeros((1, 10, 10)),
            np.arange(11).reshape(1, -1),
            np.arange(11).reshape(1, -1),
            None,
        ),
    )
    ch._update_distributions()
    for value in ch.hist.values():
        assert "input_file_name" in value
        if value["is_1d"]:
            assert "hist_values" in value
            assert "x_bin_edges" in value
            assert "uncertainties" in value
        else:
            assert "hist_values" in value
            assert "x_bin_edges" in value
            assert "y_bin_edges" in value
            assert "uncertainties" in value


def test_set_2d_distributions_basic():
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    hist_2d = ch._set_2d_distributions(xy_maximum=10 * u.m, xy_bin=5)
    assert isinstance(hist_2d, dict)
    expected_keys = {
        "counts_xy",
        "density_xy",
        "direction_xy",
        "time_altitude",
        "wavelength_altitude",
    }
    assert expected_keys.issubset(hist_2d.keys())
    for key in expected_keys:
        value = hist_2d[key]
        assert "histogram" in value
        assert hasattr(value["histogram"], "view")
        assert value["is_1d"] is False


def test_set_1d_distributions_returns_dict():
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    # Provide required 2D histograms so projections can resolve
    ch.hist = ch._set_2d_distributions(xy_maximum=1 * u.m, xy_bin=2)
    hist_1d = ch._set_1d_distributions()
    assert isinstance(hist_1d, dict)
    expected_keys = {
        "wavelength",
        "counts_r",
        "density_r",
        "density_x",
        "density_y",
        "time",
        "altitude",
        "direction_cosine_x",
        "direction_cosine_y",
        "num_photons",
    }
    assert expected_keys.issubset(hist_1d.keys())
    for key in expected_keys:
        assert "is_1d" in hist_1d[key]
        assert hist_1d[key]["is_1d"] is True


def test__get_hist_1d_from_numpy_linear(monkeypatch):
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.events = np.array([(1,), (2,), (3,), (4,), (5,)], dtype=[("dummy", "f8")])
    hist = {"x_bins": [5, 1, 6, "linear"]}
    values, edges, _uncertainties = ch._get_hist_1d_from_numpy("dummy", hist)
    assert values.shape == (1, 5)
    assert edges.shape == (1, 6)
    assert np.sum(values) == 5


def test__get_hist_1d_from_numpy_log(monkeypatch):
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.events = np.array([(1,), (10,), (100,), (1000,), (10000,)], dtype=[("dummy", "f8")])
    hist = {"x_bins": [5, 1, 10000, "log"]}
    values, edges, _uncertainties = ch._get_hist_1d_from_numpy("dummy", hist)
    assert values.shape == (1, 5)
    assert edges.shape == (1, 6)
    assert np.sum(values) == 5


def test__get_hist_1d_from_numpy_with_none(monkeypatch):
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.events = np.array([(2,), (4,), (6,), (8,)], dtype=[("dummy", "f8")])
    hist = {"x_bins": [4, None, None, "linear"]}
    values, edges, _uncertainties = ch._get_hist_1d_from_numpy("dummy", hist)
    assert values.shape == (1, 4)
    assert edges.shape == (1, 5)
    assert np.sum(values) == 4


def test_get_hist_2d_projection_returns_expected_shapes():
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    # Create a simple 2D histogram
    hist = bh.Histogram(bh.axis.Regular(3, -1, 2), bh.axis.Regular(4, 0, 4))
    # Fill with some values
    hist.fill(0, 1)
    hist.fill(1, 2)
    # Call the method
    values, x_edges, y_edges, _uncertainties = ch.get_hist_2d_projection(hist)
    assert values.shape == (1, 4, 3)  # Transposed: (1, y_bins, x_bins)
    assert x_edges.shape == (1, 4)  # First axis edges (3 bins -> 4 edges)
    assert y_edges.shape == (1, 5)  # Second axis edges (4 bins -> 5 edges)
    # Check that the returned arrays are numpy arrays
    assert isinstance(values, np.ndarray)
    assert isinstance(x_edges, np.ndarray)
    assert isinstance(y_edges, np.ndarray)


@pytest.mark.parametrize("rotate", [True, False])
def test__fill_histograms(monkeypatch, photon_dtype, rotate):
    """Test _fill_histograms with and without photon rotation."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch._density_samples = []
    event_dtype = [("azimuth_deg", "f8"), ("zenith_deg", "f8"), ("num_photons", "f8")]
    ch.events = np.zeros(1, dtype=event_dtype)
    ch.events["azimuth_deg"][0] = 0.0
    ch.events["zenith_deg"][0] = 0.0
    ch.hist = ch._set_2d_distributions(xy_maximum=1 * u.m, xy_bin=2)
    ch.hist.update(ch._set_1d_distributions(r_max=2 * u.m, bins=2))

    photons = [np.array([(10.0, 20.0, 0.1, 0.2, 5.0, 100.0, 1.0, 400.0)], dtype=photon_dtype)]
    telescope_positions = np.array(
        [(0.0, 0.0, 1000.0)], dtype=[("x", "f8"), ("y", "f8"), ("r", "f8")]
    )

    monkeypatch.setattr("simtools.corsika.corsika_histograms.rotate", lambda x, y, az, ze: (x, y))

    ch._fill_histograms(photons, 0, telescope_positions, rotate_photons=rotate)

    def assert_hist_filled(hist_key):
        view = ch.hist[hist_key]["histogram"].view()
        values = view["value"] if hasattr(view, "dtype") and view.dtype.names else view
        assert np.any(values > 0)

    for key in ["counts_xy", "counts_r"]:
        assert_hist_filled(key)

    if rotate:
        for key in [
            "direction_xy",
            "time_altitude",
            "wavelength_altitude",
        ]:
            assert_hist_filled(key)

    assert ch.events["num_photons"][0] > 0
    assert len(ch._density_samples) > 0


@pytest.mark.parametrize("scale", ["linear", "log", "with_units"])
def test_create_regular_axes_parametrized(scale):
    """Test axis creation with different scales (linear, log, and with units)."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    if scale == "linear":
        hist = {"x_bins": [5, 0, 10, "linear"], "y_bins": [4, -2, 2, "linear"]}
        axes = ch._create_regular_axes(hist, ["x_bins", "y_bins"])
        assert axes[0].size == 5
        assert axes[1].size == 4
        assert "transform" not in str(axes[0])
        assert "transform" not in str(axes[1])
    elif scale == "log":
        hist = {"x_bins": [3, 1, 100, "log"], "y_bins": [2, 10, 100, "log"]}
        axes = ch._create_regular_axes(hist, ["x_bins", "y_bins"])
        assert "transform=log" in str(axes[0])
        assert "transform=log" in str(axes[1])
    else:  # with units
        hist = {
            "x_bins": [2, 0 * u.m, 2 * u.m, "linear"],
            "y_bins": [2, -1 * u.s, 1 * u.s, "linear"],
        }
        axes = ch._create_regular_axes(hist, ["x_bins", "y_bins"])
        assert "transform" not in str(axes[0])
        assert "transform" not in str(axes[1])
        assert axes[0].edges[0] == 0
        assert axes[0].edges[-1] == 2
        assert axes[1].edges[0] == -1
        assert axes[1].edges[-1] == 1


def test_read_event_headers_creates_events(monkeypatch, tmp_path):
    """Test that _read_event_headers correctly parses event data from IACTFile."""
    events = [
        create_dummy_event(1, 10.0, np.deg2rad(30), np.deg2rad(45)),
        create_dummy_event(2, 20.0, np.deg2rad(60), np.deg2rad(80)),
    ]
    dummy_iact_file = create_dummy_iact_file(events)
    monkeypatch.setattr("simtools.corsika.corsika_histograms.IACTFile", dummy_iact_file)

    dummy_file = tmp_path / "dummy.iact"
    dummy_file.write_text("dummy")

    ch = CorsikaHistograms(dummy_file)
    ch._read_event_headers()

    assert ch.events.shape == (2,)
    assert ch.events["particle_id"][0] == 1
    assert ch.events["particle_id"][1] == 2
    assert np.isclose(ch.events["azimuth_deg"][0], 30)
    assert np.isclose(ch.events["zenith_deg"][1], 80)
    assert np.isclose(ch.events["num_photons"][0], 0.0)


def test_fill_runs_and_updates_hist(monkeypatch, tmp_path, photon_dtype):
    """Test that fill() properly fills histograms and updates distributions."""
    dummy_photon = np.array([(10.0, 20.0, 0.1, 0.2, 5.0, 100.0, 1.0, 400.0)], dtype=photon_dtype)
    event = create_dummy_event(
        1, 10.0, np.deg2rad(30), np.deg2rad(45), photon_bunches={0: dummy_photon}
    )
    telescope_pos = np.array([(0.0, 0.0, 1000.0)], dtype=[("x", "f8"), ("y", "f8"), ("r", "f8")])
    dummy_iact_file = create_dummy_iact_file([event], telescope_pos)

    monkeypatch.setattr("simtools.corsika.corsika_histograms.IACTFile", dummy_iact_file)
    monkeypatch.setattr("simtools.corsika.corsika_histograms.rotate", lambda x, y, az, ze: (x, y))

    dummy_file = tmp_path / "dummy.iact"
    dummy_file.write_text("dummy")

    ch = CorsikaHistograms(dummy_file)
    ch.fill()

    assert ch.events is not None
    # Check that histograms are filled
    for key in ["counts_xy", "density_xy", "counts_r"]:
        assert ch.hist[key]["hist_values"].shape[1:] == ch.hist[key]["histogram"].view().T.shape


def test_corsika_histograms_init_file_exists(tmp_path):
    dummy_file = tmp_path / "dummy.iact"
    dummy_file.write_text("test")
    ch = CorsikaHistograms(dummy_file)
    assert ch.input_file == dummy_file
    assert ch.events is None
    assert isinstance(ch.hist, dict)
    assert "counts_xy" in ch.hist
    assert "density_r" in ch.hist


def test_corsika_histograms_init_file_not_exists(tmp_path):
    non_existing_file = tmp_path / "notfound.iact"
    with pytest.raises(FileNotFoundError):
        CorsikaHistograms(non_existing_file)


def test_get_hist_1d_projection_numpy_hist(monkeypatch):
    """Test get_hist_1d_projection returns correct shape for numpy histogram."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.events = np.array([(1,), (2,), (3,), (4,), (5,)], dtype=[("dummy", "f8")])
    hist = {"x_bins": [5, 1, 5, "linear"]}
    result = ch.get_hist_1d_projection("dummy", hist)
    counts, edges, uncertainties = result
    assert counts.shape == (1, 5)
    assert edges.shape == (1, 6)
    assert uncertainties.shape == (1, 5)
    assert np.sum(counts) == 5


def test_get_hist_1d_projection_boost_hist():
    """Test get_hist_1d_projection returns correct shape for boost 1D histogram."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    hist = {
        "x_bins": [5, 0, 5, "linear"],
        "histogram": bh.Histogram(bh.axis.Regular(5, 0, 5)),
    }
    # Fill histogram with values
    hist["histogram"].fill([0, 1, 2, 3, 4])
    result = ch.get_hist_1d_projection("dummy", hist)
    counts, edges, uncertainties = result
    assert counts.shape == (1, 5)
    assert edges.shape == (1, 6)
    assert uncertainties is None
    assert np.sum(counts) == 5


def test_get_hist_1d_projection_boost_2d_projection():
    """Test get_hist_1d_projection returns correct shape for boost 2D histogram projection."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    hist_2d = bh.Histogram(bh.axis.Regular(3, 0, 3), bh.axis.Regular(2, 0, 2))
    hist_2d.fill(0, 0)
    hist_2d.fill(1, 1)
    ch.hist = {"test_2d": {"histogram": hist_2d}}
    hist = {"projection": ["test_2d", "x"]}
    result = ch.get_hist_1d_projection("dummy", hist)
    counts, edges, uncertainties = result
    assert counts.shape[1] == 3
    assert edges.shape[1] == 4
    assert uncertainties is None
    assert np.sum(counts) == 2


def test_get_hist_1d_projection_projection_none_no_events(monkeypatch):
    """Test get_hist_1d_projection when projection is None and events are missing."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    if hasattr(ch, "events"):
        delattr(ch, "events")
    hist = {"projection": None}
    # Should not raise, just skip the numpy histogram branch
    result = ch.get_hist_1d_projection("dummy", hist)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test__check_for_all_attributes_true():
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    class DummyView:
        dtype = type("dtype", (), {"names": ("value", "variance")})

    view = DummyView()
    assert ch._check_for_all_attributes(view) is True


def test__check_for_all_attributes_false():
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    class DummyView:
        dtype = type("dtype", (), {"names": ("value",)})

    view = DummyView()
    assert ch._check_for_all_attributes(view) is False


def test__check_for_all_attributes_no_dtype():
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    class DummyView:
        pass

    view = DummyView()
    assert ch._check_for_all_attributes(view) is False


@pytest.mark.parametrize("project_axis", ["x", "y"])
def test_fill_projected_density_values_numerical(project_axis):
    """
    Test that _fill_projected_density_values correctly computes 1D densities from 2D counts.

    This test validates the numerical computation by creating a simple 2D counts_xy histogram
    with known bin widths and counts, then verifying that the projected 1D density values
    match the expected counts-per-area calculation.
    """
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    x_bins, x_min, x_max = 3, -3.0, 3.0
    y_bins, y_min, y_max = 2, -2.0, 2.0

    h_2d = bh.Histogram(
        bh.axis.Regular(x_bins, x_min, x_max),
        bh.axis.Regular(y_bins, y_min, y_max),
        storage=bh.storage.Weight(),
    )

    x_bin_width = (x_max - x_min) / x_bins
    y_bin_width = (y_max - y_min) / y_bins

    view_2d = h_2d.view()
    view_2d["value"][0, 0] = 10.0
    view_2d["value"][1, 0] = 20.0
    view_2d["value"][2, 0] = 30.0
    view_2d["value"][0, 1] = 15.0
    view_2d["value"][1, 1] = 25.0
    view_2d["value"][2, 1] = 35.0
    view_2d["variance"][0, 0] = 10.0
    view_2d["variance"][1, 0] = 20.0
    view_2d["variance"][2, 0] = 30.0
    view_2d["variance"][0, 1] = 15.0
    view_2d["variance"][1, 1] = 25.0
    view_2d["variance"][2, 1] = 35.0

    ch.hist = {
        "counts_xy": {
            "histogram": h_2d,
            "is_1d": False,
        }
    }

    density_key = f"density_{project_axis}"
    ch.hist[density_key] = {
        "projection": ["counts_xy", project_axis],
        "is_1d": True,
    }

    ch._fill_projected_density_values(ch.hist[density_key])

    hist_values = ch.hist[density_key]["hist_values"]
    bin_edges = ch.hist[density_key]["x_bin_edges"]
    uncertainties = ch.hist[density_key]["uncertainties"]

    assert hist_values is not None
    assert bin_edges is not None
    assert uncertainties is not None

    if project_axis == "x":
        expected_counts = np.array([10.0 + 15.0, 20.0 + 25.0, 30.0 + 35.0])
        expected_variances = np.array([10.0 + 15.0, 20.0 + 25.0, 30.0 + 35.0])
        total_ortho_width = y_max - y_min
        bin_width = x_bin_width
        expected_edges = h_2d.axes[0].edges
        num_bins = x_bins
    else:
        expected_counts = np.array([10.0 + 20.0 + 30.0, 15.0 + 25.0 + 35.0])
        expected_variances = np.array([10.0 + 20.0 + 30.0, 15.0 + 25.0 + 35.0])
        total_ortho_width = x_max - x_min
        bin_width = y_bin_width
        expected_edges = h_2d.axes[1].edges
        num_bins = y_bins

    expected_areas = bin_width * total_ortho_width
    expected_density = expected_counts / expected_areas
    expected_uncertainty = np.sqrt(expected_variances) / expected_areas

    assert hist_values.shape == (1, num_bins)
    assert bin_edges.shape == (1, num_bins + 1)
    assert uncertainties.shape == (1, num_bins)

    np.testing.assert_allclose(hist_values[0], expected_density, rtol=1e-10)
    np.testing.assert_allclose(bin_edges[0], expected_edges, rtol=1e-10)
    np.testing.assert_allclose(uncertainties[0], expected_uncertainty, rtol=1e-10)


def test_fill_projected_density_values_without_weight_storage():
    """
    Test _fill_projected_density_values with non-Weight storage (fallback path).

    Validates that the fallback uncertainty calculation (sqrt(density)) works correctly
    when the 2D histogram does not use Weight storage.
    """
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    x_bins, x_min, x_max = 2, -2.0, 2.0
    y_bins, y_min, y_max = 2, -2.0, 2.0

    h_2d = bh.Histogram(
        bh.axis.Regular(x_bins, x_min, x_max),
        bh.axis.Regular(y_bins, y_min, y_max),
        storage=bh.storage.Double(),
    )

    x_bin_width = (x_max - x_min) / x_bins

    view_2d = h_2d.view()
    view_2d[0, 0] = 100.0
    view_2d[1, 0] = 200.0
    view_2d[0, 1] = 150.0
    view_2d[1, 1] = 250.0

    ch.hist = {
        "counts_xy": {
            "histogram": h_2d,
            "is_1d": False,
        }
    }

    ch.hist["density_x"] = {
        "projection": ["counts_xy", "x"],
        "is_1d": True,
    }

    ch._fill_projected_density_values(ch.hist["density_x"])

    hist_values = ch.hist["density_x"]["hist_values"]
    uncertainties = ch.hist["density_x"]["uncertainties"]

    expected_counts = np.array([100.0 + 150.0, 200.0 + 250.0])
    total_y_width = y_max - y_min
    expected_areas = x_bin_width * total_y_width
    expected_density = expected_counts / expected_areas
    expected_uncertainty = np.sqrt(expected_density)

    np.testing.assert_allclose(hist_values[0], expected_density, rtol=1e-10)
    np.testing.assert_allclose(uncertainties[0], expected_uncertainty, rtol=1e-10)


def test_filter_density_histograms_per_telescope():
    """Test _filter_density_histograms removes per-bin keys with per-telescope method."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.normalization_method = "per-telescope"
    ch.hist = {
        "density_xy": {"is_1d": False},
        "density_x": {"is_1d": True},
        "density_y": {"is_1d": True},
        "density_r": {"is_1d": True},
        "density_xy_from_counts": {"is_1d": False},
        "density_r_from_counts": {"is_1d": True},
        "counts_xy": {"is_1d": False},
    }
    ch._filter_density_histograms()
    assert "density_xy_from_counts" not in ch.hist
    assert "density_r_from_counts" not in ch.hist
    assert "density_xy" in ch.hist
    assert "density_x" in ch.hist
    assert "density_y" in ch.hist
    assert "density_r" in ch.hist
    assert "counts_xy" in ch.hist


def test_filter_density_histograms_per_bin():
    """Test _filter_density_histograms removes per-telescope keys with per-bin method."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.normalization_method = "per-bin"
    ch.hist = {
        "density_xy": {"is_1d": False},
        "density_x": {"is_1d": True},
        "density_y": {"is_1d": True},
        "density_r": {"is_1d": True},
        "density_xy_from_counts": {"is_1d": False},
        "density_r_from_counts": {"is_1d": True},
        "counts_xy": {"is_1d": False},
    }
    ch._filter_density_histograms()
    assert "density_xy" not in ch.hist
    assert "density_x" not in ch.hist
    assert "density_y" not in ch.hist
    assert "density_r" not in ch.hist
    assert "density_xy_from_counts" in ch.hist
    assert "density_r_from_counts" in ch.hist
    assert "counts_xy" in ch.hist


def test_filter_density_histograms_invalid_method():
    """Test _filter_density_histograms raises ValueError for invalid normalization method."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.normalization_method = "invalid-method"
    ch.hist = {"density_xy": {"is_1d": False}}
    with pytest.raises(ValueError, match="Unknown normalization_method"):
        ch._filter_density_histograms()


def test_filter_density_histograms_missing_keys():
    """Test _filter_density_histograms handles missing keys gracefully."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)
    ch.normalization_method = "per-telescope"
    ch.hist = {
        "counts_xy": {"is_1d": False},
        "density_xy": {"is_1d": False},
    }
    ch._filter_density_histograms()
    assert "density_xy" in ch.hist
    assert "counts_xy" in ch.hist


def test_density_and_unc_with_weight_storage():
    """Test _density_and_unc with Weight storage (has value and variance)."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    hist = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Weight())
    hist.fill([0.5, 1.5, 2.5], weight=[10, 20, 30])

    view = hist.view()
    areas = np.array([1.0, 1.0, 1.0])

    density, unc = ch._density_and_unc(view, areas)

    expected_density = np.array([10.0, 20.0, 30.0])
    expected_unc = np.sqrt(np.array([100.0, 400.0, 900.0])) / areas

    np.testing.assert_allclose(density, expected_density)
    np.testing.assert_allclose(unc, expected_unc)


def test_density_and_unc_with_double_storage():
    """Test _density_and_unc with Double storage (fallback path)."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    hist = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Double())
    hist.fill([0.5, 1.5, 2.5], weight=[5, 10, 15])

    view = hist.view()
    areas = np.array([2.0, 2.0, 2.0])

    density, unc = ch._density_and_unc(view, areas)

    expected_density = view / areas
    expected_unc = np.sqrt(view) / areas

    np.testing.assert_allclose(density, expected_density)
    np.testing.assert_allclose(unc, expected_unc)


def test_density_and_unc_with_varying_areas():
    """Test _density_and_unc with non-uniform bin areas."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    hist = bh.Histogram(bh.axis.Regular(4, 0, 10), storage=bh.storage.Weight())
    hist.fill([0.5, 2.5, 5.0, 7.5], weight=[100, 200, 300, 400])

    view = hist.view()
    areas = np.array([2.5, 2.5, 2.5, 2.5])

    density, unc = ch._density_and_unc(view, areas)

    expected_density = np.array([40.0, 80.0, 120.0, 160.0])
    expected_unc = np.sqrt(np.array([10000.0, 40000.0, 90000.0, 160000.0])) / areas

    np.testing.assert_allclose(density, expected_density)
    np.testing.assert_allclose(unc, expected_unc)


def test_density_and_unc_zero_areas():
    """Test _density_and_unc handles division by zero areas gracefully."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    hist = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.Weight())
    hist.fill([0.5, 1.5], weight=[10, 20])

    view = hist.view()
    areas = np.array([0.0, 1.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        density, unc = ch._density_and_unc(view, areas)

    assert np.isinf(density[0])
    assert np.isclose(density[1], 20.0)
    assert np.isinf(unc[0])
    assert np.isclose(unc[1], np.sqrt(400.0))


def test_density_and_unc_empty_histogram():
    """Test _density_and_unc with empty histogram."""
    ch = CorsikaHistograms.__new__(CorsikaHistograms)

    hist = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Weight())

    view = hist.view()
    areas = np.array([1.0, 1.0, 1.0])

    density, unc = ch._density_and_unc(view, areas)

    expected_density = np.array([0.0, 0.0, 0.0])
    expected_unc = np.array([0.0, 0.0, 0.0])

    np.testing.assert_allclose(density, expected_density)
    np.testing.assert_allclose(unc, expected_unc)
