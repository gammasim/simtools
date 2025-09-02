import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable

from simtools.visualization.plot_incident_angles import plot_incident_angles


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
    plot_incident_angles(results, tmp_path, "unit")
    out_dir = Path(tmp_path) / "plots"
    assert (out_dir / "incident_angles_multi_unit.png").exists()
    assert (out_dir / "incident_angles_primary_multi_unit.png").exists()
    assert (out_dir / "incident_angles_secondary_multi_unit.png").exists()


def test_plot_incident_angles_single_mirror(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    results = {
        0.0: _make_table([0.1, 0.2, 0.3], [1.0, 1.1, 1.2]),
        1.0: _make_table([0.15, 0.25, 0.35], [1.5, 1.6, 1.7]),
    }
    plot_incident_angles(results, tmp_path, "single")
    out_dir = Path(tmp_path) / "plots"
    assert (out_dir / "incident_angles_multi_single.png").exists()
    assert (out_dir / "incident_angles_primary_multi_single.png").exists()
    # Secondary may be skipped; ensure it either exists or a warning was recorded
    sec_path = out_dir / "incident_angles_secondary_multi_single.png"
    if not sec_path.exists():
        assert any(
            "No finite angle_incidence_secondary values to plot" in r.message
            for r in caplog.records
        )
