#!/usr/bin/python3
# pylint: disable=protected-access

import logging
import math
import re
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
    return {
        "telescope": "LST-1",
        "site": "North",
        "model_version": "prod6",
        "off_axis_angle": 0.0 * u.deg,
        "source_distance": 10.0,  # km
        "number_of_photons": 1000,
    }


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
    # subdirectories are created
    assert calculator.logs_dir.is_dir()
    assert calculator.scripts_dir.is_dir()
    assert calculator.photons_dir.is_dir()
    assert calculator.results_dir.is_dir()


def test_run_produces_results(monkeypatch, calculator, tmp_output_dir):
    # Avoid external run
    monkeypatch.setattr(ia.IncidentAnglesCalculator, "_run_script", lambda *a, **k: None)

    # Prepare custom file paths returned by _prepare_psf_io_files
    # Suffix now includes telescope and off-axis angle
    suffix = f"{calculator.label}_{calculator.config_data['telescope']}_off0"
    photons_file = calculator.photons_dir / f"incident_angles_photons_{suffix}.lis"
    stars_file = calculator.photons_dir / f"incident_angles_stars_{suffix}.lis"
    log_file = calculator.logs_dir / f"incident_angles_{suffix}.log"

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
    out = (
        calculator.results_dir
        / f"incident_angles_{calculator.label}_{calculator.config_data['telescope']}_off0.ecsv"
    )
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


def test_label_suffix_includes_noninteger_off_axis(monkeypatch, calculator, tmp_output_dir):
    # Avoid running external tool
    monkeypatch.setattr(ia.IncidentAnglesCalculator, "_run_script", lambda *a, **k: None)

    # Set a non-integer off-axis and ensure file names include off1.5 (no trailing zeros)
    calculator.config_data["off_axis_angle"] = 1.5 * u.deg
    photons_file, stars_file, log_file = calculator._prepare_psf_io_files()

    assert "_off1.5.lis" in photons_file.name
    assert "_off1.5.lis" in stars_file.name
    assert "_off1.5.log" in log_file.name


def test_repr_contains_label(calculator):
    s = repr(calculator)
    assert "IncidentAnglesCalculator" in s


def test_write_run_script_perfect_mirror_flags(calculator):
    calculator.perfect_mirror = True
    photons, stars, log_file = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-DPERFECT_DISH=1" in txt
    assert "-C telescope_random_angle=0" in txt
    assert "-C mirror_reflection_random_angle=0" in txt


def test_write_run_script_reflection_angle(calculator):
    # explicit value wins
    calculator.mirror_reflection_random_angle = 0.123
    photons, stars, log_file = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-C mirror_reflection_random_angle=0.123" in txt


def test_write_run_script_alignment_mirror_alignment_random(calculator):
    calculator.mirror_alignment_random = 0.05
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
    table_file = calculator.results_dir / (
        f"incident_angles_{calculator.label}_{calculator.config_data['telescope']}_off0.ecsv"
    )
    assert table_file.exists()
    # Read table and check metadata
    tbl = QTable.read(table_file, format="ascii.ecsv")
    meta = tbl.meta
    assert meta.get("zenith_angle_deg") == calculator.ZENITH_ANGLE_DEG
    assert meta.get("off_axis_angle_deg") == pytest.approx(0.0)
    assert meta.get("telescope_name") == "LST-1"
    assert meta.get("site") == "North"
    assert meta.get("data_points") == len(tbl)


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
    assert out["angle_incidence_focal_deg"] == pytest.approx([42.5, 99.0])


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
    photons_path = (
        calculator.photons_dir
        / f"incident_angles_photons_{calculator.label}_{calculator.config_data['telescope']}_off0.lis"
    )
    photons_path.parent.mkdir(parents=True, exist_ok=True)
    photons_path.write_text("dummy", encoding="utf-8")

    # Force unlink to raise OSError to exercise the warning path
    def _raise_unlink(self):  # self is a Path
        raise OSError("simulated unlink failure")

    monkeypatch.setattr(ia.Path, "unlink", _raise_unlink)
    caplog.set_level(logging.WARNING, logger=ia.__name__)

    photons, _, _ = calculator._prepare_psf_io_files()

    assert photons == photons_path
    # Warning was logged
    assert any("Failed to remove existing photons file" in rec.message for rec in caplog.records)
    # File should exist and be (re)written despite unlink failure
    assert photons.exists()


def test_attach_results_metadata_sets_expected_fields(calculator):
    # Prepare minimal results table
    calculator.results = QTable()
    calculator.results["angle_incidence_focal"] = [1.0, 2.0] * u.deg

    # Sanity: config contains expected values from fixtures
    assert calculator.config_data["site"] == "North"
    assert calculator.config_data["telescope"] == "LST-1"

    # Invoke metadata attachment
    calculator._attach_results_metadata()

    meta = calculator.results.meta
    # Core fields
    assert meta.get("label") == calculator.label
    assert meta.get("telescope_name") == "LST-1"
    assert meta.get("site") == "North"
    assert meta.get("zenith_angle_deg") == calculator.ZENITH_ANGLE_DEG
    # Off-axis from fixture is 0 deg
    assert meta.get("off_axis_angle_deg") == pytest.approx(0.0)
    # Source distance from fixture
    assert meta.get("source_distance_km") == pytest.approx(10.0)
    # Config file path recorded as string
    assert isinstance(meta.get("config_file"), str)
    # Data points equals table length
    assert meta.get("data_points") == len(calculator.results)


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
    assert math.isclose(out["angle_incidence_secondary_deg"][0], 2.0, rel_tol=0.0, abs_tol=1e-12)


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
    assert math.isclose(out["angle_incidence_primary_deg"][0], 3.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isnan(out["angle_incidence_secondary_deg"][0])


def test_header_driven_column_detection(calculator, tmp_path):
    # Provide header lines that define custom column positions (1-based):
    # focal=30, primary=34, secondary=38
    pfile = tmp_path / "header.lis"
    header_lines = [
        "# Column 30: Angle of incidence at focal surface, with respect to the optical axis [deg]\n",
        "# Column 34: Angle of incidence onto primary mirror [deg]\n",
        "# Column 38: Angle of incidence onto secondary mirror [deg]\n",
    ]
    data = ["0"] * 40
    data[29] = "11.1"  # focal at col 30
    data[33] = "22.2"  # primary at col 34
    data[37] = "33.3"  # secondary at col 38
    pfile.write_text("".join(header_lines) + " ".join(data) + "\n", encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert out["angle_incidence_focal_deg"] == pytest.approx([11.1])
    assert out["angle_incidence_primary_deg"] == pytest.approx([22.2])
    assert out["angle_incidence_secondary_deg"] == pytest.approx([33.3])


def test_match_header_column_variants():
    col_pat = re.compile(r"^\s*#\s*Column\s+(\d{1,4})\s*$", re.IGNORECASE)
    # Focal surface with optical axis mention
    raw = "# Column 30: Angle of incidence at focal surface, with respect to the optical axis [deg]"
    assert ia.IncidentAnglesCalculator._match_header_column(col_pat, raw) == ("focal", 30)

    # Primary mirror with onto
    raw = "# Column 34: Angle of incidence onto primary mirror [deg]"
    assert ia.IncidentAnglesCalculator._match_header_column(col_pat, raw) == ("primary", 34)

    # Primary mirror with on
    raw = "# Column 35: Angle of incidence on primary mirror [deg]"
    assert ia.IncidentAnglesCalculator._match_header_column(col_pat, raw) == ("primary", 35)

    # Secondary mirror
    raw = "# Column 40: Angle of incidence onto secondary mirror [deg]"
    assert ia.IncidentAnglesCalculator._match_header_column(col_pat, raw) == ("secondary", 40)

    # No match
    raw = "# Column 99: Some other description"
    assert ia.IncidentAnglesCalculator._match_header_column(col_pat, raw) is None


def test_find_column_indices_reflection_headers(calculator, tmp_path):
    # Build a file that declares reflection point columns for primary and secondary
    pfile = tmp_path / "refl_headers.lis"
    header_lines = [
        "# Column 10: X reflection point on primary mirror [cm]\n",
        "# Column 11: Y reflection point on primary mirror [cm]\n",
        "# Column 12: X reflection point on secondary mirror [cm]\n",
        "# Column 13: Y reflection point on secondary mirror [cm]\n",
    ]
    data = ["0"] * 26
    pfile.write_text("".join(header_lines) + " ".join(data) + "\n", encoding="utf-8")

    idx = calculator._find_column_indices(pfile)
    # Focal remains default (25), check XY indices converted to 0-based
    assert idx["prim_x"] == 9
    assert idx["prim_y"] == 10
    assert idx["sec_x"] == 11
    assert idx["sec_y"] == 12


def test_compute_with_header_driven_reflection_points_values(calculator, tmp_path):
    # Provide both angle and reflection point headers and a single data line
    pfile = tmp_path / "full_headers_values.lis"
    header_lines = [
        "# Column 30: Angle of incidence at focal surface, with respect to the optical axis [deg]\n",
        "# Column 34: Angle of incidence onto primary mirror [deg]\n",
        "# Column 38: Angle of incidence onto secondary mirror [deg]\n",
        "# Column 10: X reflection point on primary mirror [cm]\n",
        "# Column 11: Y reflection point on primary mirror [cm]\n",
        "# Column 12: X reflection point on secondary mirror [cm]\n",
        "# Column 13: Y reflection point on secondary mirror [cm]\n",
    ]
    data = ["0"] * 40
    data[29] = "11.0"  # focal at 30
    data[33] = "21.0"  # primary angle at 34
    data[37] = "31.0"  # secondary angle at 38
    data[9] = "10.0"  # prim x cm at 10
    data[10] = "-20.0"  # prim y cm at 11
    data[11] = "3.0"  # sec x cm at 12
    data[12] = "4.0"  # sec y cm at 13
    pfile.write_text("".join(header_lines) + " ".join(data) + "\n", encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    # Angles picked from header-driven indices
    assert out["angle_incidence_focal_deg"] == pytest.approx([11.0])
    assert out["angle_incidence_primary_deg"] == pytest.approx([21.0])
    assert out["angle_incidence_secondary_deg"] == pytest.approx([31.0])
    # Hit coordinates converted from cm to meters
    assert out["primary_hit_x_m"] == pytest.approx([0.10])
    assert out["primary_hit_y_m"] == pytest.approx([-0.20])
    assert out["secondary_hit_x_m"] == pytest.approx([0.03])
    assert out["secondary_hit_y_m"] == pytest.approx([0.04])
    # Radii computed correctly in meters
    assert out["primary_hit_radius_m"] == pytest.approx([((10.0**2 + 20.0**2) ** 0.5) / 100.0])
    assert out["secondary_hit_radius_m"] == pytest.approx([((3.0**2 + 4.0**2) ** 0.5) / 100.0])


def test_compute_angles_skips_primary_secondary_when_disabled(tmp_path):
    # Minimal targeted test for the early return when primary/secondary are disabled
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = False

    pfile = tmp_path / "angles.lis"
    parts = ["0"] * 25 + ["42.5"]  # focal at col 26
    pfile.write_text(" ".join(parts) + "\n", encoding="utf-8")

    out = calc._compute_incidence_angles_from_imaging_list(pfile)
    assert "angle_incidence_focal_deg" in out
    assert out["angle_incidence_focal_deg"] == [42.5]
    assert "angle_incidence_primary_deg" not in out
    assert "angle_incidence_secondary_deg" not in out


def test_primary_hit_geometry_nan_when_missing_coords(calculator, tmp_path):
    # focal valid; missing/invalid primary x/y should yield NaNs for radius and coords
    parts = ["0"] * 25 + ["42.0"]  # focal at col 26
    # pad cols 27-31, then invalidate prim_x (col 29, idx 28) and prim_y (col 30, idx 29)
    parts += ["0", "0", "0", "0", "0"]
    parts[28] = "badx"  # prim_x invalid
    parts[29] = "bady"  # prim_y invalid
    # primary and secondary angles present to avoid other issues
    parts.append("1.0")  # primary at 32
    parts += ["0", "0", "0", "2.0"]  # secondary at 36
    pfile = tmp_path / "prim_nan.lis"
    pfile.write_text(" ".join(parts) + "\n", encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert len(out["angle_incidence_focal_deg"]) == 1
    # Primary radius and coords should be NaN
    assert math.isnan(out["primary_hit_radius_m"][0])
    assert math.isnan(out["primary_hit_x_m"][0])
    assert math.isnan(out["primary_hit_y_m"][0])


def test_secondary_hit_geometry_nan_when_missing_coords(calculator, tmp_path):
    # focal valid; missing/invalid secondary x/y should yield NaNs for secondary radius/coords
    parts = ["0"] * 25 + ["42.0"]  # focal at col 26
    parts += ["0", "0", "0", "0", "0"]  # pad 27-31
    parts[28] = "0.0"  # valid primary x cm (col 29)
    parts[29] = "0.0"  # valid primary y cm (col 30)
    parts.append("1.0")  # primary at 32
    # secondary x/y (cols 33/34) will be invalid strings
    parts += ["badx", "bady", "0", "2.0"]  # pad cols and secondary=2.0 at col 36
    pfile = tmp_path / "sec_nan.lis"
    pfile.write_text(" ".join(parts) + "\n", encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert len(out["angle_incidence_focal_deg"]) == 1
    # Secondary radius and coords should be NaN
    assert math.isnan(out["secondary_hit_radius_m"][0])
    assert math.isnan(out["secondary_hit_x_m"][0])
    assert math.isnan(out["secondary_hit_y_m"][0])


def test_iter_data_rows_skips_comments_and_blank(tmp_path):
    p = tmp_path / "rows.lis"
    p.write_text("""# header line\n\n1 2 3\n  # another comment\n4 5 6\n""", encoding="utf-8")
    rows = list(IncidentAnglesCalculator._iter_data_rows(p))
    assert rows == [
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]


def test_parse_float_with_nan_out_of_range():
    parts = ["1.0", "2.0"]
    # Index out of range returns NaN
    assert math.isnan(IncidentAnglesCalculator._parse_float_with_nan(parts, 5))
    # Negative index returns NaN
    assert math.isnan(IncidentAnglesCalculator._parse_float_with_nan(parts, -1))


def test_set_reflection_index_if_match_ignores_when_not_primary_or_secondary():
    # Desc that mentions an axis but not primary/secondary mirror should not modify indices
    calc = object.__new__(IncidentAnglesCalculator)
    indices = {"prim_x": 28, "prim_y": 29, "sec_x": 32, "sec_y": 33}
    desc = "x reflection point on mirror center [cm]"  # no 'primary mirror' or 'secondary mirror'
    calc._set_reflection_index_if_match(desc, 15, indices)
    assert indices == {"prim_x": 28, "prim_y": 29, "sec_x": 32, "sec_y": 33}


def test_set_reflection_index_if_match_requires_axis_token():
    # Desc that mentions primary mirror but no standalone x/y should not modify indices
    calc = object.__new__(IncidentAnglesCalculator)
    indices = {}
    desc = "reflection point on primary mirror [cm]"  # no 'x' or 'y' token
    calc._set_reflection_index_if_match(desc, 12, indices)
    assert indices == {}


def test_update_indices_reflection_skips_when_flag_disabled():
    # If calculate_primary_secondary_angles is False, reflection point handling should return early
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = False
    indices = {"prim_x": 28, "prim_y": 29}
    desc = "X reflection point on primary mirror [cm]"
    calc._update_indices_from_header_desc(desc.lower(), 15, indices)
    # unchanged
    assert indices == {"prim_x": 28, "prim_y": 29}


def test_update_indices_reflection_skips_when_missing_keyword():
    # With flag True but description not containing 'reflection point', nothing should change
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = True
    indices = {"sec_x": 32, "sec_y": 33}
    desc = "X point on secondary mirror [cm]"  # missing 'reflection point'
    calc._update_indices_from_header_desc(desc.lower(), 20, indices)
    assert indices == {"sec_x": 32, "sec_y": 33}


def test_default_column_indices_flag_behavior():
    # With flag True
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = True
    idx = calc._default_column_indices()
    assert idx["focal"] == 25
    for k in ("primary", "secondary", "prim_x", "prim_y", "sec_x", "sec_y"):
        assert k in idx

    # With flag False
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = False
    idx2 = calc._default_column_indices()
    assert idx2 == {"focal": 25}


def test_update_indices_from_header_desc_angles():
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = True
    indices = {"focal": 25}
    # Focal surface
    calc._update_indices_from_header_desc(
        "angle of incidence at focal surface with respect to the optical axis", 30, indices
    )
    assert indices["focal"] == 29
    # Primary mirror
    calc._update_indices_from_header_desc(
        "angle of incidence onto primary mirror [deg]", 34, indices
    )
    assert indices["primary"] == 33
    # Secondary mirror
    calc._update_indices_from_header_desc(
        "angle of incidence on secondary mirror [deg]", 38, indices
    )
    assert indices["secondary"] == 37


def test_append_values_minimal_success():
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = True
    parts = ["42.0", "1.0", "2.0", "10.0", "20.0", "3.0", "4.0"]
    col_idx = {
        "focal": 0,
        "primary": 1,
        "secondary": 2,
        "prim_x": 3,
        "prim_y": 4,
        "sec_x": 5,
        "sec_y": 6,
    }
    focal, primary, secondary = [], [], []
    r1, r2 = [], []
    px, py, sx, sy = [], [], [], []
    calc._append_values(parts, col_idx, focal, primary, secondary, r1, r2, px, py, sx, sy)
    assert focal == [42.0]
    assert primary == [1.0]
    assert secondary == [2.0]
    # radii in meters
    assert r1 == [((10.0**2 + 20.0**2) ** 0.5) / 100.0]
    assert r2 == [((3.0**2 + 4.0**2) ** 0.5) / 100.0]
    # hits in meters
    assert px == [0.10]
    assert py == [0.20]
    assert sx == [0.03]
    assert sy == [0.04]


def test_append_primary_secondary_angles_direct():
    parts = ["0", "3.0", "bad"]
    col_idx = {"primary": 1, "secondary": 2}
    primary, secondary = [], []
    IncidentAnglesCalculator._append_primary_secondary_angles(
        object.__new__(IncidentAnglesCalculator), parts, col_idx, primary, secondary
    )
    assert primary == [3.0]
    assert len(secondary) == 1

    assert math.isnan(secondary[0])


def test_find_column_indices_defaults(calculator, tmp_path):
    # No header lines -> defaults (0-based): focal=25; others present only when flag True
    pfile = tmp_path / "no_headers.lis"
    pfile.write_text("0 1 2 3\n", encoding="utf-8")

    idx = calculator._find_column_indices(pfile)
    assert idx["focal"] == 25
    # When flag True, default mirror-related indices should exist
    assert idx["primary"] == 31
    assert idx["secondary"] == 35
    assert idx["prim_x"] == 28
    assert idx["prim_y"] == 29
    assert idx["sec_x"] == 32
    assert idx["sec_y"] == 33


def test_find_column_indices_header_overrides(calculator, tmp_path):
    # Header provides custom 1-based columns for angles and reflection points
    pfile = tmp_path / "headers_all.lis"
    header = "\n".join(
        [
            "# Column 30: Angle of incidence at focal surface, with respect to the optical axis [deg]",
            "# Column 34: Angle of incidence onto primary mirror [deg]",
            "# Column 38: Angle of incidence onto secondary mirror [deg]",
            "# Column 10: X reflection point on primary mirror [cm]",
            "# Column 11: Y reflection point on primary mirror [cm]",
            "# Column 12: X reflection point on secondary mirror [cm]",
            "# Column 13: Y reflection point on secondary mirror [cm]",
        ]
    )
    pfile.write_text(header + "\n0\n", encoding="utf-8")

    idx = calculator._find_column_indices(pfile)
    # Converted to 0-based
    assert idx["focal"] == 29
    assert idx["primary"] == 33
    assert idx["secondary"] == 37
    assert idx["prim_x"] == 9
    assert idx["prim_y"] == 10
    assert idx["sec_x"] == 11
    assert idx["sec_y"] == 12


def test_find_column_indices_ignores_mirror_when_disabled(tmp_path):
    # Even with headers present, when calculate_primary_secondary_angles is False,
    # only 'focal' should be returned/overridden.
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = False

    pfile = tmp_path / "headers_disabled.lis"
    header = "\n".join(
        [
            "# Column 26: Angle of incidence at focal surface, with respect to the optical axis [deg]",
            "# Column 32: Angle of incidence onto primary mirror [deg]",
            "# Column 36: Angle of incidence onto secondary mirror [deg]",
            "# Column 29: X reflection point on primary mirror [cm]",
            "# Column 30: Y reflection point on primary mirror [cm]",
        ]
    )
    pfile.write_text(header + "\n0\n", encoding="utf-8")

    idx = calc._find_column_indices(pfile)
    assert set(idx.keys()) == {"focal"}
    assert idx["focal"] == 25  # 1-based 26 -> 0-based 25


def test_parse_float_success_and_failure():
    parts = ["1.5", "bad"]
    ok, val = IncidentAnglesCalculator._parse_float(parts, 0)
    assert ok
    assert math.isclose(val, 1.5, rel_tol=0.0, abs_tol=1e-12)
    ok, val = IncidentAnglesCalculator._parse_float(parts, 1)
    assert not ok
    assert math.isclose(val, 0.0, rel_tol=0.0, abs_tol=1e-12)
    ok, val = IncidentAnglesCalculator._parse_float(parts, 5)
    assert not ok
    assert math.isclose(val, 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_contains_axis_cases():
    assert IncidentAnglesCalculator._contains_axis("x reflection point on primary", "x")
    assert IncidentAnglesCalculator._contains_axis("y reflection point", "y")
    # Should not match axis embedded in a word
    assert not IncidentAnglesCalculator._contains_axis("xcoord value", "x")
    # Case-insensitive check
    assert IncidentAnglesCalculator._contains_axis("Axis: X", "x")


def test_update_indices_angle_headers():
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = True
    indices = {"focal": 25, "primary": 31, "secondary": 35}
    # focal
    calc._update_indices_from_header_desc(
        "angle of incidence at focal surface with respect to the optical axis", 30, indices
    )
    assert indices["focal"] == 29
    # primary
    calc._update_indices_from_header_desc("angle of incidence onto primary mirror", 40, indices)
    assert indices["primary"] == 39
    # secondary
    calc._update_indices_from_header_desc("angle of incidence on secondary mirror", 44, indices)
    assert indices["secondary"] == 43


def test_label_suffix_returns_expected_format(calculator):
    # Default off-axis is 0 deg
    s = calculator._label_suffix()
    assert calculator.config_data["telescope"] in s
    assert f"{calculator.label}_" in s
    assert s.endswith("off0")


def test_append_values_only_focal_when_disabled(tmp_path):
    # Minimal test to check early return after focal append
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = False
    parts = ["0"] * 25 + ["42.0"]
    col_idx = {"focal": 25}
    focal = []
    calc._append_values(
        parts,
        col_idx,
        focal,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert len(focal) == 1
    assert math.isclose(focal[0], 42.0, rel_tol=0.0, abs_tol=1e-12)


def test_append_primary_secondary_angles_values():
    calc = object.__new__(IncidentAnglesCalculator)
    parts = ["0"] * 40
    parts[31] = "3.0"  # primary at idx 31
    parts[35] = "bad"  # secondary invalid -> NaN
    col_idx = {"primary": 31, "secondary": 35}
    primary, secondary = [], []
    calc._append_primary_secondary_angles(parts, col_idx, primary, secondary)
    assert math.isclose(primary[0], 3.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isnan(secondary[0])


def test_append_primary_hit_geometry_valid_and_invalid():
    calc = object.__new__(IncidentAnglesCalculator)
    # Valid coordinates
    parts = ["0"] * 40
    parts[28] = "10.0"  # x cm
    parts[29] = "-20.0"  # y cm
    col_idx = {"prim_x": 28, "prim_y": 29}
    radius, x_m, y_m = [], [], []
    calc._append_primary_hit_geometry(parts, col_idx, radius, x_m, y_m)
    assert math.isclose(radius[0], ((10.0**2 + 20.0**2) ** 0.5) / 100.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(x_m[0], 0.10, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(y_m[0], -0.20, rel_tol=0.0, abs_tol=1e-12)
    # Invalid coordinates -> NaNs
    parts[28] = "bad"
    parts[29] = "-20.0"
    calc._append_primary_hit_geometry(parts, col_idx, radius, x_m, y_m)
    assert math.isnan(radius[1])
    assert math.isnan(x_m[1])
    assert math.isnan(y_m[1])


def test_append_secondary_hit_geometry_valid_and_invalid():
    calc = object.__new__(IncidentAnglesCalculator)
    # Valid coordinates
    parts = ["0"] * 40
    parts[32] = "3.0"  # x cm
    parts[33] = "4.0"  # y cm
    col_idx = {"sec_x": 32, "sec_y": 33}
    radius2, x2_m, y2_m = [], [], []
    calc._append_secondary_hit_geometry(parts, col_idx, radius2, x2_m, y2_m)
    assert math.isclose(radius2[0], 0.05, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(x2_m[0], 0.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(y2_m[0], 0.04, rel_tol=0.0, abs_tol=1e-12)
    # Invalid -> NaNs
    parts[32] = "bad"
    parts[33] = "4.0"
    calc._append_secondary_hit_geometry(parts, col_idx, radius2, x2_m, y2_m)
    assert math.isnan(radius2[1])
    assert math.isnan(x2_m[1])
    assert math.isnan(y2_m[1])


def test_update_indices_reflection_positive():
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = True
    indices = {}
    desc = "X reflection point on primary mirror [cm]".lower()
    calc._update_indices_from_header_desc(desc, 15, indices)
    assert indices["prim_x"] == 14
