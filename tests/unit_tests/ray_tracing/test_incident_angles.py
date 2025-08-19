#!/usr/bin/python3
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.ray_tracing import incident_angles as ia
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator


@pytest.fixture
def config_data():
    return {"telescope": "LST-1", "site": "North", "model_version": "prod6"}


@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path


@pytest.fixture
def mock_models(monkeypatch):
    tel = MagicMock()
    tel.name = "LST-1"
    tel.config_file_path = Path("cfg.cfg")
    tel.config_file_directory = Path()
    tel.write_sim_telarray_config_file = MagicMock()
    tel.get_parameter_value.side_effect = lambda key: 280.0 if key == "focal_length" else 0.0

    site = MagicMock()
    site.site = "North"
    site.get_parameter_value.side_effect = (
        lambda key: 2150.0 if key == "corsika_observation_level" else 0.0
    )

    monkeypatch.setattr(ia, "initialize_simulation_models", lambda *a, **k: (tel, site))
    return SimpleNamespace(tel=tel, site=site)


@pytest.fixture
def calculator(mock_models, config_data, tmp_output_dir):
    return IncidentAnglesCalculator(
        simtel_path=Path("/simtel"),
        db_config={"db": "config"},
        config_data=config_data,
        output_dir=tmp_output_dir,
        label="test-label",
    )


def test_initialization(calculator, config_data):
    assert calculator._simtel_path == Path("/simtel")
    assert calculator.config_data == config_data
    assert calculator.output_dir.is_dir()
    assert calculator.results is None
    # rt_params should carry units
    assert calculator.rt_params["zenith_angle"].unit == u.deg
    assert calculator.rt_params["off_axis_angle"].unit == u.deg
    assert calculator.rt_params["source_distance"].unit == u.km


def test_run_produces_results(monkeypatch, calculator):
    class _FakeImage:
        def __init__(self, *a, **k):
            # Intentionally left empty: no initialization needed for this test
            pass

        def read_photon_list_from_simtel_file(self, _fname):
            self.photon_pos_x = [0.0, 1.0, 2.0]
            self.photon_pos_y = [0.0, 1.0, 2.0]

    monkeypatch.setattr(ia, "PSFImage", _FakeImage)
    monkeypatch.setattr(ia.subprocess, "check_call", lambda *a, **k: 0)
    saved = {}
    monkeypatch.setattr(ia.plt, "savefig", lambda path, **k: saved.setdefault("png", path))

    res = calculator.run()

    assert isinstance(res, QTable)
    assert len(res) == 3
    assert {"x_pix", "y_pix", "incident_angle"}.issubset(res.colnames)
    assert res["incident_angle"].unit == u.deg
    assert (calculator.output_dir / f"incident_angles_{calculator.label}.png").exists() or saved


def test_plot_incident_angles_saves_png(monkeypatch, calculator):
    calculator.results = QTable()
    calculator.results["x_pix"] = [0.0, 1.0, 2.0]
    calculator.results["y_pix"] = [0.0, 1.0, 2.0]
    calculator.results["incident_angle"] = [10, 20, 30] * u.deg

    called = {}
    monkeypatch.setattr(ia.plt, "savefig", lambda *a, **k: called.setdefault("ok", True))

    calculator.plot_incident_angles()
    assert called.get("ok", False)


def test_plot_no_results_logs_warning(caplog, calculator):
    calculator.results = QTable()  # empty
    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator.plot_incident_angles()
    assert any("No results to plot" in rec.message for rec in caplog.records)


def test_repr_contains_label_and_models(calculator):
    s = repr(calculator)
    assert "IncidentAnglesCalculator(" in s
    assert f"label={calculator.label}" in s
    assert "telescope=" in s
    assert "site=" in s


def test_write_run_script_includes_use_real_camera_flag(calculator):
    # Prepare IO files and write script with real camera option
    calculator.use_real_camera = True
    photons, stars, log_file = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-C USE_REAL_CAMERA=1" in txt


def test_run_script_raises_runtime_error_on_failure(monkeypatch, calculator, tmp_output_dir):
    script = tmp_output_dir / "fail.sh"
    script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    log_file = tmp_output_dir / "run.log"

    def _raise(*a, **k):
        raise subprocess.CalledProcessError(1, "cmd")

    monkeypatch.setattr(ia.subprocess, "check_call", lambda *a, **k: _raise())

    with pytest.raises(RuntimeError, match="Incident angles run failed, see log"):
        calculator._run_script(script, log_file)


def test_save_results_no_data_logs_warning(caplog, calculator, tmp_output_dir):
    calculator.results = QTable()  # empty
    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator._save_results()
    assert any("No results to save" in rec.message for rec in caplog.records)
    # No file should be created
    out = list(calculator.output_dir.glob("incident_angles_*.ecsv"))
    assert not out


def test_export_results_success_and_no_results(caplog, calculator):
    # No results path
    calculator.results = QTable()
    caplog.clear()
    caplog.set_level(logging.ERROR, logger=ia.__name__)
    calculator.export_results()
    assert any("Cannot export results" in rec.message for rec in caplog.records)

    # Valid results path
    calculator.results = QTable()
    calculator.results["x_pix"] = [0.0, 1.0]
    calculator.results["y_pix"] = [0.0, 1.0]
    calculator.results["incident_angle"] = [1.0, 2.0] * u.deg

    calculator.export_results()
    table_file = calculator.output_dir / f"incident_angles_{calculator.label}.ecsv"
    summary_file = calculator.output_dir / f"incident_angles_summary_{calculator.label}.txt"
    assert table_file.exists()
    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "Incident angle results for" in content
    assert "Site:" in content
    assert "Number of data points:" in content
