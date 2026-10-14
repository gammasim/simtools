#!/usr/bin/python3
# pylint: disable=protected-access

import logging
import math
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.ray_tracing import incident_angles as ia
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator
from simtools.utils.general import cleanup_intermediate_files

_IMAGING_HEADER = """# Column 2: Pixel number hit
# Column 22: Angle to pixel normal (if pixel hit) or focal surface normal [deg.]
# Column 26: Angle of incidence at focal surface, w.r.t. optical axis [deg.]
# Column 32: Angle of incidence on primary mirror, w.r.t. normal [deg.]
# Column 36: Angle of incidence on secondary mirror, w.r.t. normal [deg.]
"""


@pytest.fixture
def config_data():
    return {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "prod6",
        "off_axis_angle": 0.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "number_of_photons": 1000,
        "debug_plots": True,
    }


@pytest.fixture
def mock_models(monkeypatch):
    tel = MagicMock()
    tel.name = "LST-1"
    tel.config_file_path = Path("cfg.cfg")
    tel.config_file_directory = Path()
    tel.write_sim_telarray_config_file = MagicMock()
    tel.get_parameter_value.side_effect = lambda key: {
        "focal_length": 280.0,
        "mirror_class": 2,
    }.get(key, 0.0)

    site = MagicMock()
    site.site = "North"
    site.get_parameter_value.side_effect = lambda key: (
        2150.0 if key == "corsika_observation_level" else 0.0
    )

    dummy = object()
    monkeypatch.setattr(ia, "initialize_simulation_models", lambda *a, **k: (tel, site, dummy))
    return SimpleNamespace(tel=tel, site=site)


@pytest.fixture
def calculator(mock_models, config_data, tmp_test_directory):
    return IncidentAnglesCalculator(
        config_data=config_data,
        output_dir=tmp_test_directory,
        label="test-label",
    )


def test_initialization(calculator, config_data):
    assert calculator.zenith_angle_deg == pytest.approx(0)
    assert calculator.config_data == config_data
    assert calculator.output_dir.is_dir()
    assert calculator.results is None
    # subdirectories are created
    assert calculator.logs_dir.is_dir()
    assert calculator.scripts_dir.is_dir()
    assert calculator.photons_dir.is_dir()
    assert calculator.results_dir.is_dir()


@pytest.mark.parametrize("keep_photons", [False, True])
def test_cleanup_intermediate_files(mock_models, config_data, tmp_test_directory, keep_photons):
    root = Path(tmp_test_directory)
    config_data.update(debug_plots=False, keep_photon_files=keep_photons)
    calc = IncidentAnglesCalculator(config_data, root / "output")
    photons, stars, _ = calc._prepare_psf_io_files()
    calc._intermediate_files.add(stars)
    if not keep_photons:
        calc._intermediate_files.add(photons)
    calc.model_dir.mkdir(parents=True)
    model_file = calc.model_dir / "model.cfg"
    model_file.write_text("model", encoding="utf-8")
    unrelated = calc.scripts_dir / "unrelated.sh"
    unrelated.write_text("unrelated", encoding="utf-8")
    cleanup_intermediate_files(calc.output_dir, **calc.cleanup_options)
    assert photons.exists() is keep_photons
    assert not stars.exists()
    assert not model_file.exists()
    assert unrelated.exists()
    assert calc.results_dir.is_dir()


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_application_cleans_working_directory_and_controls_plots(
    monkeypatch, tmp_test_directory, debug, fails
):
    from simtools.applications import derive_incident_angle as app

    context = MagicMock()
    context.args = {
        "application_label": "test",
        "telescope": "SSTS-01",
        "debug_plots": debug,
        "model_version": "7.0.0",
    }
    context.io_handler.get_output_directory.return_value = Path(tmp_test_directory)
    monkeypatch.setattr(app, "APPLICATION", MagicMock(start=MagicMock(return_value=context)))
    factory = MagicMock()
    factory.return_value.run_for_offsets.return_value = {0: [1]}
    monkeypatch.setattr(app, "IncidentAnglesCalculator", factory)
    plot = MagicMock()
    monkeypatch.setattr(app, "plot_incident_angles", plot)
    cleanup = MagicMock()
    monkeypatch.setattr(app, "cleanup_intermediate_files", cleanup)
    factory.return_value.cleanup_options = {"files": []}
    if fails:
        factory.return_value.run_for_offsets.side_effect = RuntimeError("Simulation failed")
        with pytest.raises(RuntimeError, match="Simulation failed"):
            app.main()
        cleanup.assert_not_called()
        return
    app.main()
    assert plot.call_count == int(debug)
    cleanup.assert_called_once_with(Path(tmp_test_directory), files=[])
    factory.return_value.save_model_parameters.assert_called_once()


@pytest.mark.parametrize("mirror_class", [0, 1, 2])
def test_mirror_angles_follow_model(mock_models, config_data, tmp_test_directory, mirror_class):
    mock_models.tel.get_parameter_value.side_effect = None
    mock_models.tel.get_parameter_value.return_value = mirror_class
    calculator = IncidentAnglesCalculator(config_data, tmp_test_directory)
    assert calculator.calculate_primary_secondary_angles is (mirror_class == 2)


@pytest.mark.parametrize(("arguments", "expected"), [([], 0), (["--zenith_angle", "40"], 40)])
def test_zenith_angle_cli_default_and_override(arguments, expected):
    from simtools.applications.derive_incident_angle import APPLICATION

    args = APPLICATION.build_parser().parse_args(arguments)
    assert args.zenith_angle.to_value(u.deg) == pytest.approx(expected)


def test_mirror_angle_cli_option_removed():
    from simtools.applications.derive_incident_angle import APPLICATION

    parser = APPLICATION.build_parser()
    assert not hasattr(parser.parse_args([]), "calculate_primary_secondary_angles")
    with pytest.raises(SystemExit):
        parser.parse_args(["--calculate_primary_secondary_angles"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-calculate_primary_secondary_angles"])


@pytest.mark.parametrize("zenith", [0 * u.deg, 40 * u.deg, (math.pi / 6) * u.rad])
@pytest.mark.parametrize("offset", [-2, 0, 2])
def test_source_and_telescope_pointing(
    mock_models, config_data, tmp_test_directory, zenith, offset
):
    config_data.update(zenith_angle=zenith, off_axis_angle=offset * u.deg)
    calculator = IncidentAnglesCalculator(config_data, tmp_test_directory)
    photons, stars, log_file = calculator._prepare_psf_io_files()
    source = stars.read_text().splitlines()[-1].split()
    assert float(source[1]) == pytest.approx(90 - zenith.to_value(u.deg))
    script = calculator._write_run_script(photons, stars, log_file).read_text()
    options = {
        key: value
        for token in shlex.split(script)
        if "=" in token
        for key, value in [token.split("=", 1)]
    }
    assert float(options["telescope_theta"]) == pytest.approx(zenith.to_value(u.deg) - offset)
    assert float(options["telescope_phi"]) == pytest.approx(0)


def test_run_produces_results(monkeypatch, calculator, tmp_test_directory):
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
        rows = [_IMAGING_HEADER]
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


def test_label_suffix_includes_noninteger_off_axis(monkeypatch, calculator, tmp_test_directory):
    # Avoid running external tool
    monkeypatch.setattr(ia.IncidentAnglesCalculator, "_run_script", lambda *a, **k: None)

    # Set a non-integer off-axis and ensure file names include off1.5 (no trailing zeros)
    calculator.config_data["off_axis_angle"] = 1.5 * u.deg
    photons_file, stars_file, log_file = calculator._prepare_psf_io_files()

    assert "_off1.5.lis" in photons_file.name
    assert "_off1.5.lis" in stars_file.name
    assert "_off1.5.log" in log_file.name


def test_write_run_script_perfect_mirror_flags(calculator):
    calculator.perfect_mirror = True
    photons, stars, log_file = calculator._prepare_psf_io_files()
    script = calculator._write_run_script(photons, stars, log_file)
    txt = script.read_text(encoding="utf-8")
    assert "-DPERFECT_DISH=1" in txt
    assert "-C telescope_random_angle=0" in txt


def test_run_script_raises_runtime_error_on_failure(monkeypatch, calculator, tmp_test_directory):
    script = tmp_test_directory / "fail.sh"
    script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    log_file = tmp_test_directory / "run.log"

    def _raise(*a, **k):
        raise ia.job_manager.JobExecutionError("Mock job execution failed")

    monkeypatch.setattr(ia.job_manager, "submit", lambda *a, **k: _raise())

    with pytest.raises(RuntimeError, match="Incident angles run failed, see log"):
        calculator._run_script(script, log_file)


def test_save_results_no_data_logs_warning(caplog, calculator, tmp_test_directory):
    calculator.results = QTable()  # empty
    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator._save_results()
    assert any("No results to save" in rec.message for rec in caplog.records)
    # No file should be created
    out = list(calculator.results_dir.glob("incident_angles_*.ecsv"))
    assert not out


def test_compute_incidence_angles_parsing(calculator, tmp_test_directory):
    # Create a photons file with mixed content
    pfile = tmp_test_directory / "mixed.lis"
    lines = [
        "# comment line\n",
        "\n",
        "1 2 3\n",  # too few columns
        " ".join(["0"] * 25 + ["not_a_number"]) + "\n",  # bad value
        " ".join(["0"] * 25 + ["42.5"]) + "\n",  # valid
        " ".join(["0"] * 25 + ["99"] + ["0"] * 5) + "\n",
    ]
    pfile.write_text(_IMAGING_HEADER + "".join(lines), encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert "angle_incidence_focal_deg" in out
    assert out["angle_incidence_focal_deg"] == pytest.approx([42.5, 99.0])


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


## Removed _attach_results_metadata test (method removed)


def test_primary_valueerror_results_in_nan(calculator, tmp_test_directory):
    # Build one valid line with focal=1.0, primary=bad (ValueError), secondary=2.0
    # Ensure we also include X,Y on primary (cols 29,30) to avoid radius parsing errors
    parts = ["0"] * 25 + ["1.0"]  # focal at col 26
    parts += ["0", "0", "0", "0", "0"]  # pad cols 27-31
    parts[28] = "0.0"  # x cm (col 29)
    parts[29] = "0.0"  # y cm (col 30)
    parts.append("bad")  # primary at col 32 -> ValueError
    parts += ["0", "0", "0", "2.0"]  # pad cols 33-35, secondary at col 36
    pfile = tmp_test_directory / "one.lis"
    pfile.write_text(_IMAGING_HEADER + " ".join(parts) + "\n", encoding="utf-8")

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert math.isnan(out["angle_incidence_primary_deg"][0])
    assert math.isclose(out["angle_incidence_secondary_deg"][0], 2.0, rel_tol=0.0, abs_tol=1e-12)


def test_header_driven_column_detection(calculator, tmp_test_directory):
    # Provide header lines that define custom column positions (1-based):
    # focal=30, primary=34, secondary=38
    pfile = tmp_test_directory / "header.lis"
    header_lines = [
        "# Column 30: Angle of incidence at focal surface, with respect to the optical axis [deg]\n",
        "# Column 34: Angle of incidence onto primary mirror [deg]\n",
        "# Column 38: Angle of incidence onto secondary mirror [deg]\n",
    ]
    data = ["0"] * 40
    data[29] = "11.1"  # focal at col 30
    data[33] = "22.2"  # primary at col 34
    data[37] = "33.3"  # secondary at col 38
    pfile.write_text(
        _IMAGING_HEADER + "".join(header_lines) + " ".join(data) + "\n", encoding="utf-8"
    )

    out = calculator._compute_incidence_angles_from_imaging_list(pfile)
    assert out["angle_incidence_focal_deg"] == pytest.approx([11.1])
    assert out["angle_incidence_primary_deg"] == pytest.approx([22.2])
    assert out["angle_incidence_secondary_deg"] == pytest.approx([33.3])


def test_find_column_indices_reflection_headers(calculator, tmp_test_directory):
    # Build a file that declares reflection point columns for primary and secondary
    pfile = tmp_test_directory / "refl_headers.lis"
    header_lines = [
        "# Column 10: X reflection point on primary mirror [cm]\n",
        "# Column 11: Y reflection point on primary mirror [cm]\n",
        "# Column 12: X reflection point on secondary mirror [cm]\n",
        "# Column 13: Y reflection point on secondary mirror [cm]\n",
    ]
    data = ["0"] * 26
    pfile.write_text(
        _IMAGING_HEADER + "".join(header_lines) + " ".join(data) + "\n", encoding="utf-8"
    )

    idx = calculator._find_column_indices(pfile)
    # Focal remains default (25), check XY indices converted to 0-based
    assert idx["prim_x"] == 9
    assert idx["prim_y"] == 10
    assert idx["sec_x"] == 11
    assert idx["sec_y"] == 12


def test_compute_angles_skips_primary_secondary_when_disabled(tmp_test_directory):
    # Minimal targeted test for the early return when primary/secondary are disabled
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = False

    pfile = tmp_test_directory / "angles.lis"
    parts = ["0"] * 25 + ["42.5"]  # focal at col 26
    pfile.write_text(_IMAGING_HEADER + " ".join(parts) + "\n", encoding="utf-8")

    out = calc._compute_incidence_angles_from_imaging_list(pfile)
    assert "angle_incidence_focal_deg" in out
    assert out["angle_incidence_focal_deg"] == [42.5]
    assert "angle_incidence_primary_deg" not in out
    assert "angle_incidence_secondary_deg" not in out


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


def test_find_column_indices_ignores_mirror_when_disabled(tmp_test_directory):
    # Even with headers present, when calculate_primary_secondary_angles is False,
    # only 'focal' should be returned/overridden.
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = False

    pfile = tmp_test_directory / "headers_disabled.lis"
    header = "\n".join(
        [
            "# Column 26: Angle of incidence at focal surface, with respect to the optical axis [deg]",
            "# Column 32: Angle of incidence onto primary mirror [deg]",
            "# Column 36: Angle of incidence onto secondary mirror [deg]",
            "# Column 29: X reflection point on primary mirror [cm]",
            "# Column 30: Y reflection point on primary mirror [cm]",
        ]
    )
    pfile.write_text(_IMAGING_HEADER + header + "\n0\n", encoding="utf-8")

    idx = calc._find_column_indices(pfile)
    assert set(idx.keys()) == {"focal"}
    assert idx["focal"] == 25  # 1-based 26 -> 0-based 25


def test_save_model_parameters(calculator, tmp_test_directory, monkeypatch):
    # Mock ModelDataWriter and MetadataCollector
    mock_writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", mock_writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())

    # Create dummy results
    t1 = QTable()
    t1["angle_incidence_focal"] = [1.0, 2.0] * u.deg
    t1["angle_incidence_filter"] = [1.0, 2.0] * u.deg
    t1["angle_incidence_primary"] = [10.0, 20.0] * u.deg
    t1["angle_incidence_secondary"] = [25.0, 35.0] * u.deg

    results_by_offset = {0.0: t1}

    calculator.save_model_parameters(results_by_offset)

    # write_product_data and write_model_parameter are static methods called directly
    assert mock_writer.write_product_data.call_count > 0
    assert mock_writer.write_model_parameter.call_count > 0


def test_save_model_parameters_no_results_logs_warning(
    caplog, calculator, tmp_test_directory, monkeypatch
):
    monkeypatch.setattr(
        ia,
        "ModelDataWriter",
        MagicMock(side_effect=AssertionError("ModelDataWriter should not be called")),
    )

    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator.save_model_parameters({0.0: QTable()})
    assert any("No results to write model parameters." in rec.message for rec in caplog.records)

    assert not list(Path(tmp_test_directory).glob("*.ecsv"))
    assert not list(Path(tmp_test_directory).glob("*.json"))


# --- Tests for 100% coverage ---


def test_source_distance_km_with_plain_float(calculator):
    """Cover _source_distance_km when source_distance is a plain float."""
    calculator.config_data["source_distance"] = 15.0
    assert calculator._source_distance_km() == pytest.approx(15.0)


def test_append_primary_secondary_angles_with_none_arrays():
    """Cover _append_primary_secondary_angles when arrays are None."""
    calc = object.__new__(IncidentAnglesCalculator)
    parts = ["0"] * 40
    col_idx = {"primary": 31, "secondary": 35}
    calc._append_primary_secondary_angles(parts, col_idx, None, None)


def test_append_primary_hit_geometry_with_none_arrays():
    """Cover _append_primary_hit_geometry when arrays are None."""
    calc = object.__new__(IncidentAnglesCalculator)
    parts = ["0"] * 40
    col_idx = {"prim_x": 28, "prim_y": 29}
    calc._append_primary_hit_geometry(parts, col_idx, None, None, None)


def test_append_secondary_hit_geometry_with_none_arrays():
    """Cover _append_secondary_hit_geometry when arrays are None."""
    calc = object.__new__(IncidentAnglesCalculator)
    parts = ["0"] * 40
    col_idx = {"sec_x": 32, "sec_y": 33}
    calc._append_secondary_hit_geometry(parts, col_idx, None, None, None)


def test_update_indices_from_header_desc_primary_secondary():
    """Cover _update_indices_from_header_desc with primary/secondary mirror headers."""
    calc = object.__new__(IncidentAnglesCalculator)
    calc.calculate_primary_secondary_angles = True
    indices = {"focal": 25}

    calc._update_indices_from_header_desc(
        "angle of incidence onto primary mirror [deg]", 32, indices
    )
    assert indices["primary"] == 31

    calc._update_indices_from_header_desc(
        "angle of incidence on secondary mirror [deg]", 36, indices
    )
    assert indices["secondary"] == 35


def test_calculate_histogram_empty_after_filtering():
    """Cover _calculate_histogram when all data is filtered out as non-finite."""
    bin_centers, hist = IncidentAnglesCalculator._calculate_histogram(
        [float("nan"), float("inf"), float("-inf")], bins=10
    )
    assert len(bin_centers) == 0
    assert len(hist) == 0


@pytest.mark.parametrize("bins", [100, 1000])
def test_calculate_histogram_probability_density(bins):
    """Normalize the integral, including endpoints and ignoring non-finite rays."""
    centers, density = IncidentAnglesCalculator._calculate_histogram(
        [0.0, 0.0, 90.0, float("nan"), float("inf")], bins=bins
    )
    width = 90.0 / bins
    assert len(centers) == bins
    assert centers[[0, -1]] == pytest.approx([width / 2, 90 - width / 2])
    assert sum(density) * width == pytest.approx(1)
    assert density[0] == pytest.approx(2 / (3 * width))
    assert density[-1] == pytest.approx(1 / (3 * width))


def test_save_model_parameters_mirror_class_2(tmp_test_directory, monkeypatch, mock_models):
    """Cover save_model_parameters with mirror_class == 2 (dual-mirror telescope)."""
    mock_writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", mock_writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())

    config_data = {
        "telescope": "SSTS-01",
        "site": "South",
        "model_version": "7.0.0",
        "parameter_version": "1.0.0",
    }

    calculator = IncidentAnglesCalculator(
        config_data=config_data,
        output_dir=tmp_test_directory,
        label="test",
    )

    calculator.telescope_model = MagicMock()
    calculator.telescope_model.get_parameter_value.return_value = 2

    t1 = QTable()
    t1["angle_incidence_focal"] = [1.0, 2.0, 3.0] * u.deg
    t1["angle_incidence_filter"] = [1.0, 2.0, 3.0] * u.deg
    t1["angle_incidence_primary"] = [10.0, 20.0, 30.0] * u.deg
    t1["angle_incidence_secondary"] = [5.0, 10.0, 15.0] * u.deg

    results_by_offset = {0.0: t1}
    calculator.save_model_parameters(results_by_offset)

    assert mock_writer.write_product_data.call_count >= 3
    assert mock_writer.write_model_parameter.call_count >= 3


def test_save_model_parameters_no_results(tmp_test_directory, monkeypatch, caplog, mock_models):
    """Cover early return in save_model_parameters when no results."""
    mock_writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", mock_writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())

    config_data = {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "7.0.0",
    }

    calculator = IncidentAnglesCalculator(
        config_data=config_data,
        output_dir=tmp_test_directory,
        label="test",
    )
    calculator.telescope_model = MagicMock()
    calculator.telescope_model.get_parameter_value.return_value = 1

    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator.save_model_parameters({})

    assert any("No results to write model parameters." in rec.message for rec in caplog.records)
    assert mock_writer.write_product_data.call_count == 0


def test_run_with_all_columns(monkeypatch, calculator, tmp_test_directory):
    """Cover run() with all columns including primary/secondary angles and geometry."""
    monkeypatch.setattr(ia.IncidentAnglesCalculator, "_run_script", lambda *a, **k: None)

    suffix = f"{calculator.label}_{calculator.config_data['telescope']}_off0"
    photons_file = calculator.photons_dir / f"incident_angles_photons_{suffix}.lis"
    stars_file = calculator.photons_dir / f"incident_angles_stars_{suffix}.lis"
    log_file = calculator.logs_dir / f"incident_angles_{suffix}.log"

    def _prep_files(self):
        photons_file.parent.mkdir(parents=True, exist_ok=True)
        rows = [_IMAGING_HEADER]
        triplets = [(10.0, 1.0, 2.0, 100.0, 200.0, 300.0, 400.0)]
        for foc, pri, sec, px, py, sx, sy in triplets:
            parts = ["0"] * 25 + [str(foc)]
            parts += ["0", "0", str(px), str(py), "0"]
            parts += [str(pri)]
            parts += [str(sx), str(sy), "0", "0"]
            parts += [str(sec)]
            rows.append(" ".join(parts) + "\n")
        photons_file.write_text("".join(rows), encoding="utf-8")
        stars_file.write_text("0 0 1 10\n", encoding="utf-8")
        return photons_file, stars_file, log_file

    monkeypatch.setattr(ia.IncidentAnglesCalculator, "_prepare_psf_io_files", _prep_files)

    res = calculator.run()

    assert isinstance(res, QTable)
    assert len(res) == 1
    assert "angle_incidence_focal" in res.colnames
    assert "angle_incidence_primary" in res.colnames
    assert "angle_incidence_secondary" in res.colnames
    assert "primary_hit_radius" in res.colnames
    assert "secondary_hit_radius" in res.colnames


def test_build_incidence_distribution_table():
    """Cover _build_incidence_distribution_table static method."""
    calc = object.__new__(IncidentAnglesCalculator)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    table = calc._build_incidence_distribution_table(data)

    assert isinstance(table, QTable)
    assert "incidence_angle" in table.colnames
    assert "fraction" in table.colnames
    assert table["incidence_angle"].unit == u.deg
    assert len(table) == 100


@pytest.mark.parametrize("mirror_class", [0, 1, 2])
def test_exports_only_incidence_parameters(calculator, monkeypatch, mirror_class):
    """Keep the laboratory lightguide file intact while exporting ray-traced angles."""
    calculator.telescope_model.get_parameter_value.side_effect = None
    calculator.telescope_model.get_parameter_value.return_value = mirror_class
    writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())
    parameter_dir = calculator.output_dir / calculator.config_data["telescope"]
    parameter_dir.mkdir()
    lab_file = parameter_dir / "lightguide_efficiency_vs_incidence_angle.ecsv"
    original_data = "# laboratory measurement\n0 0.85\n30 0.72\n"
    lab_file.write_text(original_data, encoding="utf-8")
    table = QTable()
    for surface in ("focal", "primary", "secondary"):
        table[f"angle_incidence_{surface}"] = [10, 20, 30] * u.deg

    calculator.save_model_parameters({0.0: table})

    expected = ["camera_filter_photon_incident_angle"]
    if mirror_class == 2:
        expected.extend(["primary_mirror_incidence_angle", "secondary_mirror_incidence_angle"])
    assert [
        call.kwargs["parameter_name"] for call in writer.write_model_parameter.call_args_list
    ] == expected
    assert [
        call.kwargs["output_file"].name for call in writer.write_product_data.call_args_list
    ] == [f"{name}.ecsv" for name in expected]
    for call in writer.write_product_data.call_args_list:
        table = call.kwargs["product_data"]
        assert table.colnames == ["incidence_angle", "fraction"]
        assert table["incidence_angle"].unit == u.deg
        assert table.meta["normalization"] == "probability density per degree"
        assert sum(table["fraction"]) * table.meta["bin_width_deg"] == pytest.approx(1)
    assert lab_file.read_text(encoding="utf-8") == original_data


def test_filter_distribution_includes_rays_without_pixel_hits(calculator, monkeypatch):
    calculator.calculate_primary_secondary_angles = False
    photons = calculator.photons_dir / "filter.lis"
    photons.write_text(
        "# Column 1: Pixel number hit\n"
        "# Column 2: Angle to pixel normal [deg.]\n"
        "# Column 3: Angle of incidence at focal surface, w.r.t. optical axis [deg.]\n"
        "-1 5 30\n0 10 60\n",
        encoding="utf-8",
    )
    data = calculator._compute_incidence_angles_from_imaging_list(photons)
    assert data["angle_incidence_focal_deg"] == [30, 60]
    calculator.telescope_model.get_parameter_value.side_effect = None
    calculator.telescope_model.get_parameter_value.return_value = 1
    writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())
    table = QTable({"angle_incidence_focal": data["angle_incidence_focal_deg"] * u.deg})
    calculator.save_model_parameters({0: table})
    exported = writer.write_product_data.call_args.kwargs["product_data"]
    nonzero = exported[exported["fraction"] > 0]
    assert list(nonzero["fraction"]) == pytest.approx([0.5 / 0.9, 0.5 / 0.9])
    assert nonzero["incidence_angle"].to_value(u.deg) == pytest.approx([30.15, 59.85])
    assert exported.meta["angle_reference"] == "optical axis at focal surface"
