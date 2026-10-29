import logging
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

import simtools.visualization.plot_incident_angles as pia

LOG = logging.getLogger(__name__)


def _make_table(focal_vals, primary_vals=None, secondary_vals=None):
    t = QTable()
    t["angle_incidence_focal"] = np.array(focal_vals) * u.deg
    if primary_vals is not None:
        t["angle_incidence_primary"] = np.array(primary_vals) * u.deg
    if secondary_vals is not None:
        t["angle_incidence_secondary"] = np.array(secondary_vals) * u.deg
    return t


def _messages(caplog):
    return [r.message for r in caplog.records]


def test_plot_incident_angles_dual_mirror(tmp_test_directory):
    results = {
        0.0: _make_table([0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [2.0, 2.2, 2.4]),
        1.0: _make_table([0.15, 0.25, 0.35], [1.5, 1.6, 1.7], [2.5, 2.6, 2.7]),
    }
    pia.plot_incident_angles(results, tmp_test_directory, "unit_LSTN-01")
    out_dir = Path(tmp_test_directory) / "plots"
    assert (out_dir / "incident_angles_multi_unit_LSTN-01.png").exists()
    assert (out_dir / "incident_angles_primary_multi_unit_LSTN-01.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_unit_LSTN-01.png").exists()


def test_plot_incident_angles_single_mirror(tmp_test_directory):
    results = {
        0.0: _make_table([0.1, 0.2, 0.3], [1.0, 1.1, 1.2]),
        1.0: _make_table([0.15, 0.25, 0.35], [1.5, 1.6, 1.7]),
    }
    pia.plot_incident_angles(results, tmp_test_directory, "single_SSTS-04")
    out_dir = Path(tmp_test_directory) / "plots"
    assert (out_dir / "incident_angles_multi_single_SSTS-04.png").exists()


def test_warning_empty_results_for_offset(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    pia.plot_incident_angles({0.0: QTable()}, tmp_test_directory, "empty")
    assert any("Empty results for off-axis=0.0" in m for m in _messages(caplog))
    out_dir = Path(tmp_test_directory) / "plots"
    assert not any(out_dir.glob("incident_angles_multi_*.png"))


def test_no_finite_focal_bins_none(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    t = QTable()
    t["angle_incidence_focal"] = np.array([np.nan, np.nan]) * u.deg
    pia.plot_incident_angles({0.0: t}, tmp_test_directory, "nanfocal")
    assert any("No focal-surface incidence angle values to plot" in m for m in _messages(caplog))
    out_dir = Path(tmp_test_directory) / "plots"
    assert not (out_dir / "incident_angles_multi_nanfocal.png").exists()


def test_no_finite_nonfocal_bins_none(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    results = {0.0: _make_table(focal_vals=[0.1], primary_vals=[np.nan])}
    out_path = Path(tmp_test_directory) / "plots" / "should_not_exist.png"
    pia._plot_angles(results, "angle_incidence_primary", "T", "X", out_path, 0.1, LOG)
    assert any("No angle_incidence_primary values to plot" in m for m in _messages(caplog))
    assert not out_path.exists()


def test_invalid_bin_edges_warning_with_monkeypatch(tmp_test_directory, caplog, monkeypatch):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(pia.np, "floor", lambda x: np.nan)
    results = {0.0: _make_table(focal_vals=[0.1, 0.2], primary_vals=[1.0, 1.1])}
    out_path = Path(tmp_test_directory) / "plots" / "invalid_bins.png"
    pia._plot_angles(results, "angle_incidence_primary", "T", "X", out_path, 0.1, LOG)
    assert any("Invalid bin edges for angle_incidence_primary" in m for m in _messages(caplog))
    assert not out_path.exists()


def test_top_level_no_results_and_no_arrays(tmp_test_directory, caplog):
    caplog.set_level(logging.WARNING)
    pia.plot_incident_angles({}, tmp_test_directory, "none")
    assert any("No results provided for multi-offset plot" in m for m in _messages(caplog))
    caplog.clear()
    pia.plot_incident_angles({0.0: QTable()}, tmp_test_directory, "empty2")
    assert any("Empty results for off-axis=0.0" in m for m in _messages(caplog))


def test_logger_injection_used_for_warnings(tmp_test_directory, caplog):
    custom_logger = logging.getLogger("simtools.test.custom_logger")
    caplog.set_level(logging.WARNING, logger=custom_logger.name)
    pia.plot_incident_angles({}, tmp_test_directory, "nores", logger=custom_logger)
    assert any(
        r.name == custom_logger.name and "No results provided" in r.message for r in caplog.records
    )
    assert not (Path(tmp_test_directory) / "plots").exists()


def test_compute_bins_adjusts_when_vmax_equals_vmin():
    # Values exactly on a bin edge produce vmin==vmax; branch should adjust vmax
    bins = pia._compute_bins([np.array([0.1, 0.1])], 0.1, LOG, "angle_incidence_primary")
    assert bins is not None
    assert np.isclose(bins[0], 0.1)
    assert np.isclose(bins[1], 0.2)


def test_plot_radius_histograms_bins_none_and_empty_paths(tmp_test_directory, caplog, monkeypatch):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "radius_none.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "radius_col": "primary_hit_radius",
        "title": "T",
        "xlabel": "X",
        "out_path": out_path,
        "bin_width_m": 0.1,
        "log": LOG,
    }
    # No table has the column -> no arrays -> early return
    pia._plot_radius_histograms({0.0: QTable()}, **kwargs)
    assert not out_path.exists()

    # Bin edges invalid -> early return
    t = QTable()
    t["primary_hit_radius"] = np.array([0.1, 0.2]) * u.m
    monkeypatch.setattr(pia.np, "floor", lambda x: np.nan)
    pia._plot_radius_histograms({0.0: t}, **kwargs)
    assert not out_path.exists()


def test_plot_radius_histograms_skips_non_finite_tables(tmp_test_directory):
    out_path = Path(tmp_test_directory) / "plots" / "radius_continue.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_ok = QTable()
    t_ok["primary_hit_radius"] = np.array([0.1, 0.2, 0.15]) * u.m
    t_missing = QTable()
    t_zero = QTable()
    t_zero["primary_hit_radius"] = np.array([np.nan, np.inf]) * u.m
    pia._plot_radius_histograms(
        {0.0: t_ok, 1.0: t_missing, 2.0: t_zero},
        radius_col="primary_hit_radius",
        title="Radius Hist Continue Coverage",
        xlabel="Primary-hit radius on M1 (m)",
        out_path=out_path,
        bin_width_m=0.05,
        log=LOG,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


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
        log=LOG,
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
        log=LOG,
    )
    assert (out_dir / "incident_primary_radius_vs_angle_multi_ut.png").exists()


@pytest.mark.parametrize(
    ("table", "title", "missing_msg"),
    [
        (QTable(), "No Cols", "No Cols"),
        (
            # Radius present but angle missing
            QTable({"primary_hit_radius": np.array([0.1]) * u.m}),
            "R vs A Missing",
            "R vs A Missing",
        ),
        (
            # Columns present but no finite rows
            QTable(
                {
                    "primary_hit_radius": np.array([np.nan, np.inf]) * u.m,
                    "angle_incidence_primary": np.array([np.nan, np.inf]) * u.deg,
                }
            ),
            "No Points Title",
            "No Points Title",
        ),
    ],
)
def test_plot_radius_vs_angle_warns_when_no_valid_points(
    tmp_test_directory, caplog, table, title, missing_msg
):
    caplog.set_level(logging.WARNING)
    out_path = Path(tmp_test_directory) / "plots" / "no_points.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pia._plot_radius_vs_angle(
        {0.0: table},
        radius_col="primary_hit_radius",
        angle_col="angle_incidence_primary",
        title=title,
        out_path=out_path,
        log=LOG,
    )
    assert any(f"No valid data to plot for {missing_msg}" in m for m in _messages(caplog))
    assert not out_path.exists()


def test_xy_heatmaps_per_offset(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = QTable()
    t1 = QTable()
    t0["primary_hit_x"] = np.array([0.0, 0.1, 0.2]) * u.m
    t0["primary_hit_y"] = np.array([0.0, -0.1, 0.05]) * u.m
    t1["primary_hit_x"] = np.array([0.05, -0.1]) * u.m
    t1["primary_hit_y"] = np.array([0.05, 0.1]) * u.m
    # Tables that are empty, partial, or non-finite produce no output
    t_partial = QTable({"primary_hit_x": np.array([0.0]) * u.m})
    t_nan = QTable(
        {"primary_hit_x": np.array([np.nan]) * u.m, "primary_hit_y": np.array([np.nan]) * u.m}
    )
    pia._plot_xy_heatmaps_per_offset(
        {0.0: t0, 1.0: t1, 2.0: QTable(), 3.0: t_partial, 4.0: t_nan},
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
    assert not (out_dir / "incident_primary_xy_heatmap_off2_ut.png").exists()
    assert not (out_dir / "incident_primary_xy_heatmap_off3_ut.png").exists()
    assert not (out_dir / "incident_primary_xy_heatmap_off4_ut.png").exists()


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
    pia.plot_incident_angles({0.0: t0}, tmp_test_directory, "dbg", debug_plots=True)
    assert (out_dir / "incident_radius_primary_multi_dbg.png").exists()
    assert (out_dir / "incident_radius_secondary_multi_dbg.png").exists()
    assert (out_dir / "incident_primary_radius_vs_angle_multi_dbg.png").exists()
    assert (out_dir / "incident_secondary_radius_vs_angle_multi_dbg.png").exists()
    assert (out_dir / "incident_primary_xy_heatmap_off0_dbg.png").exists()
    assert (out_dir / "incident_secondary_xy_heatmap_off0_dbg.png").exists()


def test_gather_angle_arrays_logs_for_empty_focal(caplog):
    caplog.set_level(logging.WARNING)
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.1, 0.2]) * u.deg
    arrays = pia._gather_arrays(
        {0.0: None, 1.0: QTable(), 2.0: t}, "angle_incidence_focal", u.deg, LOG
    )
    assert len(arrays) == 1
    assert np.allclose(arrays[0], np.array([0.1, 0.2]))
    assert any("Empty results for off-axis=1.0" in m for m in _messages(caplog))


def test_plot_incident_angles_with_model_version(tmp_test_directory):
    out_dir = Path(tmp_test_directory) / "plots"
    t = QTable()
    t["angle_incidence_focal"] = np.array([0.1, 0.2, 0.3]) * u.deg
    t["angle_incidence_primary"] = np.array([1.0, 1.1, 1.2]) * u.deg
    t["angle_incidence_secondary"] = np.array([2.0, 2.1, 2.2]) * u.deg
    pia.plot_incident_angles({0.0: t}, tmp_test_directory, "version_test", model_version="1.0.0")
    assert (out_dir / "incident_angles_multi_version_test.png").exists()
    assert (out_dir / "incident_angles_primary_multi_version_test.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_version_test.png").exists()
