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


def test_plot_incident_angles_dual_mirror(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    results = {
        0.0: _make_table([0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [2.0, 2.2, 2.4]),
        1.0: _make_table([0.15, 0.25, 0.35], [1.5, 1.6, 1.7], [2.5, 2.6, 2.7]),
    }
    pia.plot_incident_angles(results, tmp_test_directory, "unit_LSTN-01")
    out_dir = Path(tmp_test_directory) / "plots"
    assert (out_dir / "incident_angles_multi_unit_LSTN-01.png").exists()
    assert (out_dir / "incident_angles_primary_multi_unit_LSTN-01.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_unit_LSTN-01.png").exists()


def test_plot_incident_angles_single_mirror(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    results = {
        0.0: _make_table([0.1, 0.2, 0.3], [1.0, 1.1, 1.2]),
        1.0: _make_table([0.15, 0.25, 0.35], [1.5, 1.6, 1.7]),
    }
    pia.plot_incident_angles(results, tmp_test_directory, "single_SSTS-04")
    out_dir = Path(tmp_test_directory) / "plots"
    assert (out_dir / "incident_angles_multi_single_SSTS-04.png").exists()


def test_warning_empty_results_for_offset(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    results = {0.0: QTable()}
    pia.plot_incident_angles(results, tmp_test_directory, "empty")
    msgs = [r.message for r in caplog.records]
    assert any("Empty results for off-axis=0.0" in m for m in msgs)
    out_dir = Path(tmp_test_directory) / "plots"
    assert not any(out_dir.glob("incident_angles_multi_*.png"))


def test_no_finite_focal_bins_none(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    t = QTable()
    t["angle_incidence_focal"] = np.array([np.nan, np.nan]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(results, tmp_test_directory, "nanfocal")
    msgs = [r.message for r in caplog.records]
    assert any("No focal-surface incidence angle values to plot" in m for m in msgs)
    out_dir = Path(tmp_test_directory) / "plots"
    assert not (out_dir / "incident_angles_multi_nanfocal.png").exists()


def test_no_finite_nonfocal_bins_none(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    results = {0.0: _make_table(focal_vals=[0.1], primary_vals=[np.nan])}
    out_dir = Path(tmp_test_directory) / "plots"
    pia._plot_component_angles(
        results_by_offset=results,
        column="angle_incidence_primary",
        title_suffix="on primary mirror (w.r.t. normal)",
        out_path=out_dir / "should_not_exist.png",
        bin_width_deg=0.1,
        log=logging.getLogger(__name__),
    )
    msgs = [r.message for r in caplog.records]
    assert any("No angle_incidence_primary values to plot" in m for m in msgs)
    assert not (out_dir / "should_not_exist.png").exists()


def test_invalid_bin_edges_warning_with_monkeypatch(tmp_test_directory, caplog, monkeypatch):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(pia.np, "floor", lambda x: np.nan)
    results = {0.0: _make_table(focal_vals=[0.1, 0.2], primary_vals=[1.0, 1.1])}
    out_dir = Path(tmp_test_directory) / "plots"
    pia._plot_component_angles(
        results_by_offset=results,
        column="angle_incidence_primary",
        title_suffix="on primary mirror (w.r.t. normal)",
        out_path=out_dir / "invalid_bins.png",
        bin_width_deg=0.1,
        log=logging.getLogger(__name__),
    )
    msgs = [r.message for r in caplog.records]
    assert any("Invalid bin edges for angle_incidence_primary" in m for m in msgs)
    assert not (out_dir / "invalid_bins.png").exists()


def test_bins_adjust_when_vmax_le_vmin():
    logger = logging.getLogger(__name__)
    arr = np.array([1.234, 1.234])
    bins = pia._compute_bins(arr, 0.1, logger, "angle_incidence_primary")
    assert bins is not None
    assert len(bins) >= 2


def test_overlay_skips_missing_and_empty_columns(monkeypatch):
    valid = _make_table([0.1, 0.2], [1.0, 1.1])
    missing = QTable()
    nan_primary = _make_table([0.3, 0.4], [np.nan, np.nan])
    results = {0.0: valid, 1.0: missing, 2.0: nan_primary}
    arrays = [valid["angle_incidence_primary"].to(u.deg).value]
    bins = pia._compute_bins(
        np.concatenate(arrays), 0.1, logging.getLogger(__name__), "angle_incidence_primary"
    )
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    orig_hist = ax.hist
    calls = []

    def _wrapped_hist(*args, **kwargs):
        calls.append((args, kwargs))
        return orig_hist(*args, **kwargs)

    monkeypatch.setattr(ax, "hist", _wrapped_hist)
    pia._plot_overlay_angles(results, "angle_incidence_primary", bins, ax, use_zorder=False)
    assert len(calls) == 3
    plt.close(fig)


def test_top_level_no_results_and_no_arrays(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    pia.plot_incident_angles({}, tmp_test_directory, "none")
    msgs = [r.message for r in caplog.records]
    assert any("No results provided for multi-offset plot" in m for m in msgs)
    caplog.clear()
    pia.plot_incident_angles({0.0: QTable()}, tmp_test_directory, "empty2")
    msgs = [r.message for r in caplog.records]
    assert any("Empty results for off-axis=0.0" in m for m in msgs)


def test_primary_component_empty_does_not_emit_focal_empty_warning(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    results = {0.0: QTable()}
    out_dir = Path(tmp_test_directory) / "plots"
    pia._plot_component_angles(
        results_by_offset=results,
        column="angle_incidence_primary",
        title_suffix="on primary mirror (w.r.t. normal)",
        out_path=out_dir / "should_not_exist.png",
        bin_width_deg=0.1,
        log=logging.getLogger(__name__),
    )
    msgs = [r.message for r in caplog.records]
    assert not any("Empty results for off-axis=" in m for m in msgs)
    assert not (out_dir / "should_not_exist.png").exists()


def test_plot_filters_nonfinite_values_and_succeeds(tmp_test_directory):
    t = QTable()
    t["angle_incidence_focal"] = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(results, tmp_test_directory, "finite_filter")
    out = Path(tmp_test_directory) / "plots" / "incident_angles_multi_finite_filter.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_compute_bins_edges_follow_floor_ceil():
    arr = np.array([0.05, 0.24])
    bins = pia._compute_bins(arr, 0.1, logging.getLogger(__name__), "angle_incidence_primary")
    assert bins is not None
    assert np.isclose(bins[0], 0.0)
    assert np.isclose(bins[-1], 0.3)
    assert len(bins) == 4


def test_overlay_plots_offsets_in_sorted_order(tmp_test_directory, monkeypatch):
    results = {
        1.0: _make_table([0.1, 0.2], [1.0, 1.1]),
        0.0: _make_table([0.2, 0.3], [1.2, 1.3]),
        2.0: _make_table([0.3, 0.4], [1.4, 1.5]),
    }
    arrays = [
        results[0.0]["angle_incidence_primary"].to(u.deg).value,
        results[1.0]["angle_incidence_primary"].to(u.deg).value,
        results[2.0]["angle_incidence_primary"].to(u.deg).value,
    ]
    bins = pia._compute_bins(
        np.concatenate(arrays), 0.1, logging.getLogger(__name__), "angle_incidence_primary"
    )
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    first_labels = []
    orig_hist = ax.hist

    def _wrapped_hist(*args, **kwargs):
        lab = kwargs.get("label")
        if lab and lab != "_nolegend_":
            first_labels.append(lab)
        return orig_hist(*args, **kwargs)

    monkeypatch.setattr(ax, "hist", _wrapped_hist)
    pia._plot_overlay_angles(results, "angle_incidence_primary", bins, ax, use_zorder=False)
    plt.close(fig)
    assert first_labels == ["off-axis 0 deg", "off-axis 1 deg", "off-axis 2 deg"]


def test_logger_injection_used_for_warnings(tmp_test_directory, caplog):
    custom_logger = logging.getLogger("simtools.test.custom_logger")
    caplog.set_level(logging.WARNING, logger=custom_logger.name)
    pia.plot_incident_angles({}, tmp_test_directory, "nores", logger=custom_logger)
    assert any(
        r.name == custom_logger.name and "No results provided" in r.message for r in caplog.records
    )
    assert not (Path(tmp_test_directory) / "plots").exists()


def test_invalid_edges_warning_for_focal_monkeypatch(tmp_test_directory, caplog, monkeypatch):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(pia.np, "floor", lambda x: np.nan)
    t = _make_table([0.1, 0.2])
    pia.plot_incident_angles({0.0: t}, tmp_test_directory, "invfocal")
    msgs = [r.message for r in caplog.records]
    assert any("Invalid bin edges for focal" in m for m in msgs)
    assert not (Path(tmp_test_directory) / "plots" / "incident_angles_multi_invfocal.png").exists()


def test_compute_bins_adjusts_when_vmax_equals_vmin():
    # Values exactly on a bin edge produce vmin==vmax; branch should adjust vmax
    logger = logging.getLogger(__name__)
    arr = np.array([0.1, 0.1])
    bins = pia._compute_bins(arr, 0.1, logger, "angle_incidence_primary")
    assert bins is not None
    # Should cover the vmax <= vmin branch
    assert np.isclose(bins[0], 0.1)
    assert np.isclose(bins[1], 0.2)


def test_plot_xy_heatmap_missing_columns_continue_and_warns(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "xy_missing_cols.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = QTable()
    t["primary_hit_x"] = np.array([0.0]) * u.m  # y column missing, len>0
    pia._plot_xy_heatmap(
        {0.0: t},
        "primary_hit_x",
        "primary_hit_y",
        "XY Missing Cols",
        out_path,
        logging.getLogger(__name__),
        bins=8,
    )
    assert any("No valid data to plot for XY Missing Cols" in r.message for r in caplog.records)
    assert not out_path.exists()


def test_plot_xy_heatmap_empty_after_mask_continue(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "xy_empty_after_mask.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = QTable()
    t["primary_hit_x"] = np.array([np.nan]) * u.m
    t["primary_hit_y"] = np.array([np.nan]) * u.m
    pia._plot_xy_heatmap(
        {0.0: t},
        "primary_hit_x",
        "primary_hit_y",
        "XY Empty After Mask",
        out_path,
        logging.getLogger(__name__),
        bins=8,
    )
    assert any("No valid data to plot for XY Empty After Mask" in r.message for r in caplog.records)
    assert not out_path.exists()


def test_plot_xy_heatmaps_per_offset_continue_paths(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_missing = QTable()  # triggers len==0 continue
    t_partial = QTable()
    t_partial["primary_hit_x"] = np.array([0.0]) * u.m  # y missing
    # Also include a table with nan-only to hit x.size==0 continue
    t_nan = QTable()
    t_nan["primary_hit_x"] = np.array([np.nan]) * u.m
    t_nan["primary_hit_y"] = np.array([np.nan]) * u.m
    pia._plot_xy_heatmaps_per_offset(
        {0.0: t_missing, 1.0: t_partial, 2.0: t_nan},
        x_col="primary_hit_x",
        y_col="primary_hit_y",
        title_prefix="Primary mirror: X-Y hit distribution",
        file_stem="incident_primary_xy_heatmap_off",
        out_dir=out_dir,
        label="ut",
        bins=16,
    )
    # No files expected as all entries continue
    assert not any(out_dir.glob("incident_primary_xy_heatmap_off*_ut.png"))


def test_plot_radius_histograms_early_returns_and_continues(
    tmp_test_directory, caplog, monkeypatch
):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "radius_none.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__)
    # 1) arrays empty -> early return
    pia._plot_radius_histograms(
        {0.0: QTable()},
        radius_col="primary_hit_radius",
        title="T",
        xlabel="X",
        out_path=out_path,
        bin_width_m=0.1,
        log=log,
    )
    assert not out_path.exists()

    # 2) bins_m None -> early return (monkeypatch np.floor to force NaN)
    t = QTable()
    t["primary_hit_radius"] = np.array([0.1, 0.2]) * u.m
    monkeypatch.setattr(pia.np, "floor", lambda x: np.nan)
    pia._plot_radius_histograms(
        {0.0: t},
        radius_col="primary_hit_radius",
        title="T2",
        xlabel="X",
        out_path=out_path,
        bin_width_m=0.1,
        log=log,
    )
    assert not out_path.exists()

    # 3) loop continues for missing col and for data filtered to size 0
    t_missing = QTable()  # missing column continue
    t_zero = QTable()
    t_zero["primary_hit_radius"] = np.array([np.nan, np.inf]) * u.m  # size 0 after mask
    pia._plot_radius_histograms(
        {0.0: t_missing, 1.0: t_zero},
        radius_col="primary_hit_radius",
        title="T3",
        xlabel="X",
        out_path=out_path,
        bin_width_m=0.1,
        log=log,
    )
    # Still no file produced
    assert not out_path.exists()


def test_plot_radius_vs_angle_missing_columns_continue(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "radius_vs_angle_missing_cols.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = QTable()
    # Include only radius column with one finite row, angle column missing
    t["primary_hit_radius"] = np.array([0.1]) * u.m
    pia._plot_radius_vs_angle(
        {0.0: t},
        radius_col="primary_hit_radius",
        angle_col="angle_incidence_primary",
        title="R vs A Missing",
        out_path=out_path,
        log=logging.getLogger(__name__),
    )
    assert any("No valid data to plot for R vs A Missing" in r.message for r in caplog.records)
    assert not out_path.exists()


def test_plot_radius_histograms_primary(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = QTable()
    t1 = QTable()
    t0["primary_hit_radius"] = np.array([0.1, 0.2, np.nan, 0.3]) * u.m
    t1["primary_hit_radius"] = np.array([0.05, 0.15]) * u.m
    pia._plot_radius_histograms(
        {0.0: t0, 1.0: t1},
        radius_col="primary_hit_radius",
        title="Primary mirror hit radius vs off-axis angle",
        xlabel="Primary-hit radius on M1 (m)",
        out_path=out_dir / "incident_radius_primary_multi_ut.png",
        bin_width_m=0.05,
        log=logging.getLogger(__name__),
    )
    assert (out_dir / "incident_radius_primary_multi_ut.png").exists()


def test_plot_radius_vs_angle_primary(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = QTable()
    t0["primary_hit_radius"] = np.array([0.1, 0.2, 0.3]) * u.m
    t0["angle_incidence_primary"] = np.array([1.0, 1.2, 1.4]) * u.deg
    pia._plot_radius_vs_angle(
        {0.0: t0},
        radius_col="primary_hit_radius",
        angle_col="angle_incidence_primary",
        title="Primary mirror: hit radius vs incidence angle",
        out_path=out_dir / "incident_primary_radius_vs_angle_multi_ut.png",
        log=logging.getLogger(__name__),
    )
    assert (out_dir / "incident_primary_radius_vs_angle_multi_ut.png").exists()


def test_xy_heatmaps_per_offset_primary(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = QTable()
    t1 = QTable()
    t0["primary_hit_x"] = np.array([0.0, 0.1, 0.2]) * u.m
    t0["primary_hit_y"] = np.array([0.0, -0.1, 0.05]) * u.m
    t1["primary_hit_x"] = np.array([0.05, -0.1]) * u.m
    t1["primary_hit_y"] = np.array([0.05, 0.1]) * u.m
    pia._plot_xy_heatmaps_per_offset(
        {0.0: t0, 1.0: t1},
        x_col="primary_hit_x",
        y_col="primary_hit_y",
        title_prefix="Primary mirror: X-Y hit distribution",
        file_stem="incident_primary_xy_heatmap_off",
        out_dir=out_dir,
        label="ut",
        bins=50,
    )
    assert (out_dir / "incident_primary_xy_heatmap_off0_ut.png").exists()
    assert (out_dir / "incident_primary_xy_heatmap_off1_ut.png").exists()


def test_iter_xy_valid_points_sorted_and_filtered():
    t0 = QTable()
    t1 = QTable()
    t2 = QTable()
    # valid at 1.0
    t1["primary_hit_x"] = np.array([0.0, 0.1, np.nan]) * u.m
    t1["primary_hit_y"] = np.array([0.0, -0.1, 0.2]) * u.m
    # missing y at 0.0 -> skipped
    t0["primary_hit_x"] = np.array([0.5]) * u.m
    # nan-only at 2.0 -> skipped
    t2["primary_hit_x"] = np.array([np.nan]) * u.m
    t2["primary_hit_y"] = np.array([np.nan]) * u.m
    res = {1.0: t1, 0.0: t0, 2.0: t2}
    out = list(pia._iter_xy_valid_points(res, "primary_hit_x", "primary_hit_y"))
    # Only the 1.0 entry should survive filtering
    assert len(out) == 1
    off, x, y = out[0]
    assert np.isclose(off, 1.0)
    # Last nan should be filtered out
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_debug_plots_generate_expected_files(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t0 = QTable()
    t0["angle_incidence_focal"] = np.array([0.1, 0.2, 0.3]) * u.deg
    t0["angle_incidence_primary"] = np.array([1.0, 1.1, 1.2]) * u.deg
    t0["angle_incidence_secondary"] = np.array([2.0, 2.1, 2.2]) * u.deg
    t0["primary_hit_radius"] = np.array([0.1, 0.12, 0.09]) * u.m
    t0["secondary_hit_radius"] = np.array([0.05, 0.07, 0.06]) * u.m
    t0["primary_hit_x"] = np.array([0.0, 0.1, 0.2]) * u.m
    t0["primary_hit_y"] = np.array([0.0, -0.1, 0.1]) * u.m
    t0["secondary_hit_x"] = np.array([0.02, -0.02, 0.01]) * u.m
    t0["secondary_hit_y"] = np.array([0.03, 0.01, -0.02]) * u.m
    results = {0.0: t0}
    pia.plot_incident_angles(results, tmp_test_directory, "dbg", debug_plots=True)
    assert (out_dir / "incident_radius_primary_multi_dbg.png").exists()
    assert (out_dir / "incident_radius_secondary_multi_dbg.png").exists()
    assert (out_dir / "incident_primary_radius_vs_angle_multi_dbg.png").exists()
    assert (out_dir / "incident_secondary_radius_vs_angle_multi_dbg.png").exists()
    assert (out_dir / "incident_primary_xy_heatmap_off0_dbg.png").exists()
    assert (out_dir / "incident_secondary_xy_heatmap_off0_dbg.png").exists()


def test_gather_radius_arrays_handles_units_and_errors(caplog):
    caplog.set_level(logging.WARNING)
    t_ok = QTable()
    t_ok["primary_hit_radius"] = np.array([0.1, 0.2]) * u.m
    t_missing = QTable()  # missing column
    t_empty = QTable([[]])  # empty table rows
    t_bad = QTable()
    # Put plain floats without units to trigger AttributeError on .to(u.m)
    t_bad["primary_hit_radius"] = np.array([1.0, 2.0])
    arrays = pia._gather_radius_arrays(
        {0.0: t_ok, 1.0: t_missing, 2.0: t_empty, 3.0: t_bad},
        "primary_hit_radius",
        logging.getLogger(__name__),
    )
    # Only the good one should be gathered
    assert len(arrays) == 1
    assert np.allclose(arrays[0], np.array([0.1, 0.2]))
    # Warning for the bad one
    assert any("Skipping radius values for off-axis=3.0" in r.message for r in caplog.records)


def test_plot_radius_vs_angle_warns_when_no_points(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "no_points.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Table with finite but filtered out due to no matching columns
    t = QTable()
    t["primary_hit_radius"] = np.array([np.nan, np.inf]) * u.m
    t["angle_incidence_primary"] = np.array([np.nan, np.inf]) * u.deg
    pia._plot_radius_vs_angle(
        {0.0: t},
        radius_col="primary_hit_radius",
        angle_col="angle_incidence_primary",
        title="No Points Title",
        out_path=out_path,
        log=logging.getLogger(__name__),
    )
    assert any("No valid data to plot for No Points Title" in r.message for r in caplog.records)
    assert not out_path.exists()


def test_plot_xy_heatmap_success_and_empty(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_ok = Path(tmp_test_directory) / "plots" / "xy_ok.png"
    out_empty = Path(tmp_test_directory) / "plots" / "xy_empty.png"
    out_ok.parent.mkdir(parents=True, exist_ok=True)
    t_ok = QTable()
    t_ok["primary_hit_x"] = np.array([0.0, 0.1, 0.2, np.nan]) * u.m
    t_ok["primary_hit_y"] = np.array([0.0, -0.1, 0.05, 0.2]) * u.m
    pia._plot_xy_heatmap(
        {0.0: t_ok},
        "primary_hit_x",
        "primary_hit_y",
        "XY OK",
        out_ok,
        logging.getLogger(__name__),
        bins=32,
    )
    assert out_ok.exists()
    assert out_ok.stat().st_size > 0
    # Now empty case: wrong columns and empty table
    t_empty = QTable()
    pia._plot_xy_heatmap(
        {0.0: t_empty},
        "primary_hit_x",
        "primary_hit_y",
        "XY EMPTY",
        out_empty,
        logging.getLogger(__name__),
        bins=32,
    )
    assert any("No valid data to plot for XY EMPTY" in r.message for r in caplog.records)
    assert not out_empty.exists()


def test_gather_angle_arrays_ignores_none_and_logs_for_empty(caplog):
    caplog.set_level(logging.WARNING)
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.1, 0.2]) * u.deg
    arrays = pia._gather_angle_arrays(
        {0.0: None, 1.0: QTable(), 2.0: t}, "angle_incidence_focal", logging.getLogger(__name__)
    )
    assert len(arrays) == 1
    assert np.allclose(arrays[0], np.array([0.1, 0.2]))
    assert any("Empty results for off-axis=1.0" in r.message for r in caplog.records)


def test_plot_radius_vs_angle_with_missing_columns_warns(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "no_cols.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pia._plot_radius_vs_angle(
        {0.0: QTable()},
        radius_col="primary_hit_radius",
        angle_col="angle_incidence_primary",
        title="No Cols",
        out_path=out_path,
        log=logging.getLogger(__name__),
    )
    assert any("No valid data to plot for No Cols" in r.message for r in caplog.records)
    assert not out_path.exists()


def test_plot_xy_heatmap_with_none_and_missing_warns(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "xy_none.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pia._plot_xy_heatmap(
        {0.0: None, 1.0: QTable()},
        "primary_hit_x",
        "primary_hit_y",
        "XY None/Missing",
        out_path,
        logging.getLogger(__name__),
        bins=16,
    )
    assert any("No valid data to plot for XY None/Missing" in r.message for r in caplog.records)
    assert not out_path.exists()


def test_plot_radius_histograms_covers_continue_lines(tmp_test_directory):
    out_path = Path(tmp_test_directory) / "plots" / "radius_continue.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__)
    # One valid table to ensure bins are computed and plotting proceeds
    t_ok = QTable()
    t_ok["primary_hit_radius"] = np.array([0.1, 0.2, 0.15]) * u.m
    # One table missing the radius column -> triggers first continue branch
    t_missing = QTable()
    # One table with non-finite only -> triggers second continue branch after mask
    t_zero = QTable()
    t_zero["primary_hit_radius"] = np.array([np.nan, np.inf]) * u.m
    pia._plot_radius_histograms(
        {0.0: t_ok, 1.0: t_missing, 2.0: t_zero},
        radius_col="primary_hit_radius",
        title="Radius Hist Continue Coverage",
        xlabel="Primary-hit radius on M1 (m)",
        out_path=out_path,
        bin_width_m=0.05,
        log=log,
    )
    # Plot should still be produced thanks to the valid table
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_incident_angles_with_model_version(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.1, 0.2, 0.3]) * u.deg
    t["angle_incidence_primary"] = np.array([1.0, 1.1, 1.2]) * u.deg
    t["angle_incidence_secondary"] = np.array([2.0, 2.1, 2.2]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(
        results,
        tmp_test_directory,
        "version_test",
        model_version="1.0.0",
    )
    assert (out_dir / "incident_angles_multi_version_test.png").exists()
    assert (out_dir / "incident_angles_primary_multi_version_test.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_version_test.png").exists()


def test_plot_incident_angles_none_logger_uses_module_logger(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    results = {}
    pia.plot_incident_angles(results, tmp_test_directory, "no_logger", logger=None)
    msgs = [r.message for r in caplog.records]
    assert any("No results provided for multi-offset plot" in m for m in msgs)


def test_plot_incident_angles_custom_bin_widths(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.05, 0.15, 0.25, 0.35]) * u.deg
    t["angle_incidence_primary"] = np.array([0.5, 1.5, 2.5, 3.5]) * u.deg
    t["angle_incidence_secondary"] = np.array([1.0, 2.0, 3.0, 4.0]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(
        results,
        tmp_test_directory,
        "custom_bins",
        bin_width_deg=0.2,
        radius_bin_width_m=0.02,
    )
    assert (out_dir / "incident_angles_multi_custom_bins.png").exists()
    assert (out_dir / "incident_angles_primary_multi_custom_bins.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_custom_bins.png").exists()


def test_plot_incident_angles_debug_plots_true(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t0 = QTable()
    t0["angle_incidence_focal"] = np.array([0.1, 0.2]) * u.deg
    t0["angle_incidence_primary"] = np.array([1.0, 1.1]) * u.deg
    t0["angle_incidence_secondary"] = np.array([2.0, 2.1]) * u.deg
    t0["primary_hit_radius"] = np.array([0.1, 0.12]) * u.m
    t0["secondary_hit_radius"] = np.array([0.05, 0.07]) * u.m
    t0["primary_hit_x"] = np.array([0.0, 0.1]) * u.m
    t0["primary_hit_y"] = np.array([0.0, -0.1]) * u.m
    t0["secondary_hit_x"] = np.array([0.02, -0.02]) * u.m
    t0["secondary_hit_y"] = np.array([0.03, 0.01]) * u.m
    results = {0.0: t0}
    pia.plot_incident_angles(
        results,
        tmp_test_directory,
        "dbg_enabled",
        debug_plots=True,
    )
    assert (out_dir / "incident_angles_multi_dbg_enabled.png").exists()
    assert (out_dir / "incident_angles_primary_multi_dbg_enabled.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_dbg_enabled.png").exists()
    assert (out_dir / "incident_radius_primary_multi_dbg_enabled.png").exists()
    assert (out_dir / "incident_radius_secondary_multi_dbg_enabled.png").exists()
    assert (out_dir / "incident_primary_radius_vs_angle_multi_dbg_enabled.png").exists()
    assert (out_dir / "incident_secondary_radius_vs_angle_multi_dbg_enabled.png").exists()
    assert (out_dir / "incident_primary_xy_heatmap_off0_dbg_enabled.png").exists()
    assert (out_dir / "incident_secondary_xy_heatmap_off0_dbg_enabled.png").exists()


def test_plot_incident_angles_debug_plots_false(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.1, 0.2]) * u.deg
    t["angle_incidence_primary"] = np.array([1.0, 1.1]) * u.deg
    t["angle_incidence_secondary"] = np.array([2.0, 2.1]) * u.deg
    t["primary_hit_radius"] = np.array([0.1, 0.12]) * u.m
    results = {0.0: t}
    pia.plot_incident_angles(
        results,
        tmp_test_directory,
        "dbg_disabled",
        debug_plots=False,
    )
    assert (out_dir / "incident_angles_multi_dbg_disabled.png").exists()
    assert not any(out_dir.glob("incident_radius_primary_multi_*.png"))


def test_plot_incident_angles_multiple_offsets(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t0 = QTable()
    t0["angle_incidence_focal"] = np.array([0.1, 0.2]) * u.deg
    t0["angle_incidence_primary"] = np.array([1.0, 1.1]) * u.deg
    t0["angle_incidence_secondary"] = np.array([2.0, 2.1]) * u.deg
    t1 = QTable()
    t1["angle_incidence_focal"] = np.array([0.3, 0.4]) * u.deg
    t1["angle_incidence_primary"] = np.array([1.5, 1.6]) * u.deg
    t1["angle_incidence_secondary"] = np.array([2.5, 2.6]) * u.deg
    results = {0.0: t0, 1.0: t1}
    pia.plot_incident_angles(results, tmp_test_directory, "multi_offset")
    assert (out_dir / "incident_angles_multi_multi_offset.png").exists()


def test_plot_incident_angles_no_focal_angles(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    out_dir = Path(tmp_test_directory) / "plots"
    t = QTable()
    t["angle_incidence_primary"] = np.array([1.0, 1.1]) * u.deg
    t["angle_incidence_secondary"] = np.array([2.0, 2.1]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(results, tmp_test_directory, "no_focal")
    assert not (out_dir / "incident_angles_multi_no_focal.png").exists()
    assert (out_dir / "incident_angles_primary_multi_no_focal.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_no_focal.png").exists()


def test_plot_incident_angles_only_focal_angles(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.1, 0.2, 0.3]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(results, tmp_test_directory, "focal_only")
    assert (out_dir / "incident_angles_multi_focal_only.png").exists()
    assert not (out_dir / "incident_angles_primary_multi_focal_only.png").exists()
    assert not (out_dir / "incident_angles_secondary_multi_focal_only.png").exists()


def test_plot_incident_angles_creates_output_directory(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    assert not out_dir.exists()
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.1]) * u.deg
    results = {0.0: t}
    pia.plot_incident_angles(results, tmp_test_directory, "creates_dir")
    assert out_dir.exists()
    assert out_dir.is_dir()
