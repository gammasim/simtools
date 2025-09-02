import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

import simtools.visualization.plot_incident_angles as pia


def _make_table(focal_vals, primary_vals=None, secondary_vals=None):
    t = QTable()
    t["angle_incidence_focal"] = np.array(focal_vals) * u.deg
    if primary_vals is not None:
        t["angle_incidence_primary"] = np.array(primary_vals) * u.deg
    if secondary_vals is not None:
        t["angle_incidence_secondary"] = np.array(secondary_vals) * u.deg
    return t


def test_plot_incident_angles_dual_mirror(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    results = {
        0.0: _make_table([0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [2.0, 2.2, 2.4]),
        1.0: _make_table([0.15, 0.25, 0.35], [1.5, 1.6, 1.7], [2.5, 2.6, 2.7]),
    }
    # Include telescope in label to mirror application behavior
    pia.plot_incident_angles(results, tmp_path, "unit_LSTN-01")
    out_dir = Path(tmp_path) / "plots"
    assert (out_dir / "incident_angles_multi_unit_LSTN-01.png").exists()
    assert (out_dir / "incident_angles_primary_multi_unit_LSTN-01.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_unit_LSTN-01.png").exists()


def test_plot_incident_angles_single_mirror(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    results = {
        0.0: _make_table([0.1, 0.2, 0.3], [1.0, 1.1, 1.2]),
        1.0: _make_table([0.15, 0.25, 0.35], [1.5, 1.6, 1.7]),
    }
    pia.plot_incident_angles(results, tmp_path, "single_SSTS-04")
    out_dir = Path(tmp_path) / "plots"
    assert (out_dir / "incident_angles_multi_single_SSTS-04.png").exists()


def test_warning_empty_results_for_offset(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    # Empty table for focal should trigger an "Empty results for off-axis" warning
    results = {0.0: QTable()}
    pia.plot_incident_angles(results, tmp_path, "empty")
    msgs = [r.message for r in caplog.records]
    assert any("Empty results for off-axis=0.0" in m for m in msgs)
    assert any("No non-empty results to plot" in m for m in msgs)
    # No plots should be created
    out_dir = Path(tmp_path) / "plots"
    assert not any(out_dir.glob("incident_angles_multi_*.png"))


def test_no_finite_focal_bins_none(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    # Focal column present but all NaNs => bins None and warning
    t = QTable()
    t["angle_incidence_focal"] = np.array([np.nan, np.nan]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(results, tmp_path, "nanfocal")
    msgs = [r.message for r in caplog.records]
    assert any("No finite focal-surface incidence angle values to plot" in m for m in msgs)
    out_dir = Path(tmp_path) / "plots"
    assert not (out_dir / "incident_angles_multi_nanfocal.png").exists()


def test_no_finite_nonfocal_bins_none(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    # Primary column present but all NaNs => component bins None and warning
    results = {
        0.0: _make_table(focal_vals=[0.1], primary_vals=[np.nan]),
    }
    # Run focal first to create main plot
    pia.plot_incident_angles(results, tmp_path, "nanprimary")
    # Now explicitly call component to isolate branch
    out_dir = Path(tmp_path) / "plots"
    pia._plot_component(
        results_by_offset=results,
        column="angle_incidence_primary",
        title_suffix="on primary mirror (w.r.t. normal)",
        out_path=out_dir / "should_not_exist.png",
        bin_width_deg=0.1,
        log=logging.getLogger(__name__),
    )
    # Expect warning and no file
    assert not (out_dir / "should_not_exist.png").exists()


def test_invalid_bin_edges_warning_with_monkeypatch(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.WARNING)
    # Force np.floor to return NaN to trigger invalid edges path
    monkeypatch.setattr(pia.np, "floor", lambda x: np.nan)
    results = {0.0: _make_table(focal_vals=[0.1, 0.2], primary_vals=[1.0, 1.1])}
    # Call component to avoid affecting focal plot path
    out_dir = Path(tmp_path) / "plots"
    pia._plot_component(
        results_by_offset=results,
        column="angle_incidence_primary",
        title_suffix="on primary mirror (w.r.t. normal)",
        out_path=out_dir / "invalid_bins.png",
        bin_width_deg=0.1,
        log=logging.getLogger(__name__),
    )
    msgs = [r.message for r in caplog.records]
    assert any(
        "Invalid bin edges computed for angle_incidence_primary histogram" in m for m in msgs
    )
    assert not (out_dir / "invalid_bins.png").exists()


def test_bins_adjust_when_vmax_le_vmin(tmp_path):
    # Directly exercise _compute_bins when min == max
    logger = logging.getLogger(__name__)
    arr = np.array([1.234, 1.234])
    bins = pia._compute_bins(arr, bin_width_deg=0.1, log=logger, context="angle_incidence_primary")
    assert bins is not None
    # Expect at least two edges
    assert len(bins) >= 2


def test_overlay_skips_missing_and_empty_columns(tmp_path, monkeypatch):
    # Prepare valid and invalid entries
    valid = _make_table([0.1, 0.2], [1.0, 1.1])
    missing = QTable()  # no columns
    nan_primary = _make_table([0.3, 0.4], [np.nan, np.nan])
    results = {0.0: valid, 1.0: missing, 2.0: nan_primary}
    # Build bins from valid primary values
    arrays = [valid["angle_incidence_primary"].to(u.deg).value]
    bins = pia._compute_bins(
        np.concatenate(arrays), 0.1, logging.getLogger(__name__), "angle_incidence_primary"
    )
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    orig_hist = ax.hist
    calls: list[tuple] = []

    def _wrapped_hist(*args, **kwargs):
        calls.append((args, kwargs))
        return orig_hist(*args, **kwargs)

    monkeypatch.setattr(ax, "hist", _wrapped_hist)
    pia._plot_overlay(results, "angle_incidence_primary", bins, ax, use_zorder=False)
    # Expect exactly three hist calls for the single valid off-axis (step, filled, step)
    assert len(calls) == 3
    plt.close(fig)


def test_top_level_no_results_and_no_arrays(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    # No results_by_offset
    pia.plot_incident_angles({}, tmp_path, "none")
    msgs = [r.message for r in caplog.records]
    assert any("No results provided for multi-offset plot" in m for m in msgs)
    # Only empty tables -> no arrays
    caplog.clear()
    pia.plot_incident_angles({0.0: QTable()}, tmp_path, "empty2")
    msgs = [r.message for r in caplog.records]
    assert any("No non-empty results to plot" in m for m in msgs)
