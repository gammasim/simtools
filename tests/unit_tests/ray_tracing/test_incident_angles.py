#!/usr/bin/python3
import logging
import math
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
    # subdirectories are created
    assert calculator.logs_dir.is_dir()
    assert calculator.scripts_dir.is_dir()
    assert calculator.photons_dir.is_dir()
    assert calculator.results_dir.is_dir()


def test_run_produces_results(monkeypatch, calculator, tmp_output_dir):
    # Avoid external run
    monkeypatch.setattr(ia.IncidentAnglesCalculator, "_run_script", lambda *a, **k: None)

    # Prepare custom file paths returned by _prepare_psf_io_files
    photons_file = calculator.photons_dir / f"incident_angles_photons_{calculator.label}.lis"
    stars_file = calculator.photons_dir / f"incident_angles_stars_{calculator.label}.lis"
    log_file = calculator.logs_dir / f"incident_angles_{calculator.label}.log"

    def _prep_files(self):
        photons_file.parent.mkdir(parents=True, exist_ok=True)
        # Make imaging list with:
        # - focal angle in column 26 (index 25)
        # - primary in column 32 (index 31)
        # - secondary in column 36 (index 35)
        rows = ["# header\n"]
        triplets = [(10.0, 1.0, 2.0), (20.0, 3.0, 4.0), (30.0, 5.0, 6.0)]
        for foc, pri, sec in triplets:
            parts = ["0"] * 25 + [str(foc)]
            # pad up to 31 then primary, pad to 35 then secondary
            parts += ["0"] * 5 + [str(pri)] + ["0"] * 4 + [str(sec)]
            rows.append(" ".join(parts) + "\n")
        photons_file.write_text("".join(rows), encoding="utf-8")
        stars_file.write_text("0 0 1 10\n", encoding="utf-8")
        return photons_file, stars_file, log_file

    monkeypatch.setattr(ia.IncidentAnglesCalculator, "_prepare_psf_io_files", _prep_files)

    res = calculator.run()

    assert isinstance(res, QTable)
    assert len(res) == 3
    assert {"angle_incidence_focal"}.issubset(res.colnames)
    assert res["angle_incidence_focal"].unit == u.deg
    # new columns present and have units
    assert "angle_incidence_primary" in res.colnames
    assert "angle_incidence_secondary" in res.colnames
    assert res["angle_incidence_primary"].unit == u.deg
    assert res["angle_incidence_secondary"].unit == u.deg
    # Results saved under results_dir
    out = calculator.results_dir / f"incident_angles_{calculator.label}.ecsv"
    assert out.exists()


def test_run_for_offsets_restores_label_and_collects(monkeypatch, calculator):
    base_label = calculator.label

    def _fake_run(self):
        t = QTable()
        t["angle_incidence_focal"] = [1.0, 2.0] * u.deg
        self.results = t
        return t

    monkeypatch.setattr(ia.IncidentAnglesCalculator, "run", _fake_run)

    offsets = [0.0, 1.5, 3.0]
    res = calculator.run_for_offsets(offsets)
    assert set(res.keys()) == {0.0, 1.5, 3.0}
    assert all(isinstance(v, QTable) for v in res.values())
    assert calculator.label == base_label  # restored


def test_repr_contains_label(calculator):
    s = repr(calculator)
    assert "IncidentAnglesCalculator(" in s
    assert f"label={calculator.label}" in s


def test_write_run_script_perfect_mirror_flags(calculator):
    calculator.perfect_mirror = True
    photons, stars, log_file = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-DPERFECT_DISH=1" in txt
    assert "-C telescope_random_angle=0" in txt
    assert "-C mirror_reflection_random_angle=0" in txt


def test_write_run_script_reflection_angle_and_overwrite(calculator):
    # explicit value wins
    calculator.mirror_reflection_random_angle = 0.123
    photons, stars, log_file = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-C mirror_reflection_random_angle=0.123" in txt

    # overwrite flag sets to zero if no explicit value
    calculator.mirror_reflection_random_angle = None
    calculator.overwrite_rdna = True
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-C mirror_reflection_random_angle=0" in txt


def test_write_run_script_alignment_algn(calculator):
    calculator.algn = 0.05
    photons, stars, log_file = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-C mirror_align_random_horizontal=0.05,28.,0.0,0.0" in txt
    assert "-C mirror_align_random_vertical=0.05,28.,0.0,0.0" in txt


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
    out = list(calculator.results_dir.glob("incident_angles_*.ecsv"))
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
    calculator.results["angle_incidence_focal"] = [1.0, 2.0] * u.deg

    calculator.export_results()
    table_file = calculator.results_dir / f"incident_angles_{calculator.label}.ecsv"
    summary_file = calculator.results_dir / f"incident_angles_summary_{calculator.label}.txt"
    assert table_file.exists()
    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "Incident angle results for" in content
    assert "Site:" in content
    assert "Number of data points:" in content


def test_compute_incidence_angles_parsing(calculator, tmp_path):
    # Create a photons file with mixed content
    pfile = tmp_path / "mixed.lis"
    lines = [
        "# comment line\n",
        "\n",
        "1 2 3\n",  # too few columns
        " ".join(["0"] * 25 + ["not_a_number"]) + "\n",  # bad value
        " ".join(["0"] * 25 + ["42.5"]) + "\n",  # valid
        " ".join(["0"] * 25 + ["99"] + ["0"] * 5) + "\n",
    ]
    pfile.write_text("".join(lines), encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert "angle_incidence_focal_deg" in out
    assert out["angle_incidence_focal_deg"] == [42.5, 99.0]


def test_prepare_psf_io_files_creates_in_subdirs(calculator):
    photons, stars, log = calculator._prepare_psf_io_files()
    assert photons.parent == calculator.photons_dir
    assert stars.parent == calculator.photons_dir
    assert log.parent == calculator.logs_dir


def test_write_run_script_path_and_content(calculator):
    photons, stars, log = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log)
    assert script.parent == calculator.scripts_dir
    txt = script.read_text(encoding="utf-8")
    assert str(photons) in txt
    assert str(stars) in txt


def test_prepare_psf_io_files_unlink_warning(monkeypatch, caplog, calculator):
    # Create an existing photons file to trigger the unlink branch
    photons_path = calculator.photons_dir / f"incident_angles_photons_{calculator.label}.lis"
    photons_path.parent.mkdir(parents=True, exist_ok=True)
    photons_path.write_text("dummy", encoding="utf-8")

    # Force unlink to raise OSError to exercise the warning path
    def _raise_unlink(self):  # self is a Path
        raise OSError("simulated unlink failure")

    monkeypatch.setattr(ia.Path, "unlink", _raise_unlink)
    caplog.set_level(logging.WARNING, logger=ia.__name__)

    photons, stars, log = calculator._prepare_psf_io_files()

    assert photons == photons_path
    # Warning was logged
    assert any("Failed to remove existing photons file" in rec.message for rec in caplog.records)
    # File should exist and be (re)written despite unlink failure
    assert photons.exists()


def test_primary_valueerror_results_in_nan(calculator, tmp_path):
    # Build one valid line with focal=1.0, primary=bad (ValueError), secondary=2.0
    # Ensure we also include X,Y on primary (cols 29,30) to avoid radius parsing errors
    parts = ["0"] * 25 + ["1.0"]  # focal at col 26
    parts += ["0", "0", "0", "0", "0"]  # pad cols 27-31
    parts[28] = "0.0"  # x cm (col 29)
    parts[29] = "0.0"  # y cm (col 30)
    parts.append("bad")  # primary at col 32 -> ValueError
    parts += ["0", "0", "0", "2.0"]  # pad cols 33-35, secondary at col 36
    pfile = tmp_path / "one.lis"
    pfile.write_text(" ".join(parts) + "\n", encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert math.isnan(out["angle_incidence_primary_deg"][0])
    assert out["angle_incidence_secondary_deg"][0] == 2.0


def test_secondary_valueerror_results_in_nan(calculator, tmp_path):
    # Build one valid line with focal=1.0, primary=3.0, secondary=bad (ValueError)
    parts = ["0"] * 25 + ["1.0"]  # focal at col 26
    parts += ["0", "0", "0", "0", "0"]  # pad cols 27-31
    parts[28] = "0.0"  # x cm (col 29)
    parts[29] = "0.0"  # y cm (col 30)
    parts.append("3.0")  # primary at col 32
    parts += ["0", "0", "0", "bad"]  # pad cols 33-35 and secondary=bad at col 36
    pfile = tmp_path / "one2.lis"
    pfile.write_text(" ".join(parts) + "\n", encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert out["angle_incidence_primary_deg"][0] == 3.0
    assert math.isnan(out["angle_incidence_secondary_deg"][0])
