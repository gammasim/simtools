#!/usr/bin/python3
# pylint: disable=protected-access

import math
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from simtools.ray_tracing import parameter_changes
from simtools.ray_tracing.backends import sim_telarray as backend_module
from simtools.ray_tracing.backends.interface import StarImageRequest
from simtools.ray_tracing.backends.sim_telarray import SimtelBackend, _ImagingList

_IMAGING_HEADER = """# Column 2: Pixel number hit
# Column 22: Angle to pixel normal (if pixel hit) or focal surface normal [deg.]
# Column 26: Angle of incidence at focal surface, w.r.t. optical axis [deg.]
# Column 32: Angle of incidence on primary mirror, w.r.t. normal [deg.]
# Column 36: Angle of incidence on secondary mirror, w.r.t. normal [deg.]
"""


@pytest.fixture
def models():
    tel = MagicMock()
    tel.name = "LSTN-01"
    tel.site = "North"
    tel.config_file_path = Path("cfg.cfg")
    tel.config_file_directory = Path()
    site = MagicMock()
    site.get_parameter_value.return_value = 2150.0
    return SimpleNamespace(tel=tel, site=site)


@pytest.fixture
def backend(models):
    return SimtelBackend(models.tel, models.site, "test", 20.0, False)


def _request(off_axis_x=0.0, changes=None, **kwargs):
    return StarImageRequest(
        off_axis_x=off_axis_x,
        off_axis_y=0.0,
        source_distance_km=10.0,
        seeds=((11, 22),),
        photons_per_instance=1000,
        parameter_changes=changes or {},
        **kwargs,
    )


def _imaging_rows(rows):
    """Imaging list rows: focal angle (col 26), primary hit x/y (29, 30), primary angle (32),
    secondary hit x/y (33, 34) and secondary angle (36)."""
    lines = [
        _IMAGING_HEADER,
        "# Column 29: X reflection point on primary mirror [cm]\n",
        "# Column 30: Y reflection point on primary mirror [cm]\n",
        "# Column 33: X reflection point on secondary mirror [cm]\n",
        "# Column 34: Y reflection point on secondary mirror [cm]\n",
    ]
    for focal, primary, secondary, px, py, sx, sy in rows:
        parts = ["0"] * 25 + [str(focal), "0", "0", str(px), str(py), "0", str(primary)]
        parts += [str(sx), str(sy), "0", str(secondary)]
        lines.append(" ".join(parts) + "\n")
    return "".join(lines)


@pytest.mark.parametrize("offset", [-2.0, 0.0, 2.0])
def test_incidence_script_points_the_telescope(backend, tmp_test_directory, offset):
    script = Path(tmp_test_directory) / "run.sh"
    photons, stars = Path(tmp_test_directory) / "p.lis", Path(tmp_test_directory) / "s.lis"
    request = _request(offset, parameter_changes.PERFECT_MIRROR)
    backend._write_incidence_inputs(request, photons, stars)
    backend._write_incidence_script(request, script, photons, stars, Path("run.log"))
    assert float(stars.read_text().split()[1]) == pytest.approx(70.0)
    text = script.read_text()
    assert "exec > 'run.log' 2> 'run.err'" in text
    options = dict(token.split("=", 1) for token in shlex.split(text) if "=" in token)
    assert float(options["telescope_theta"]) == pytest.approx(20.0 - offset)
    assert float(options["telescope_phi"]) == pytest.approx(0.0)
    assert options["random_seed"] == "11,22"
    assert options["mirror_reflection_random_angle"] == "0.0,0.0,0.0"


@pytest.mark.parametrize("keep_files", [False, True])
def test_incidence_angles_run(backend, tmp_test_directory, monkeypatch, keep_files):
    rows = [(10.0, 1.0, 2.0, 300.0, 400.0, 30.0, -40.0), (20.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0)]

    def run(script):
        text = Path(script).read_text()
        photons = Path(re.search(r"IMAGING_LIST=(\S+)", text).group(1))
        photons.write_text(photons.read_text() + _imaging_rows(rows), encoding="utf-8")

    monkeypatch.setattr(backend_module.job_manager, "submit", run)
    data, files = backend.incidence_angles(
        _request(), Path(tmp_test_directory), "name", True, keep_files=keep_files
    )
    assert data["angle_incidence_focal"] == pytest.approx([10.0, 20.0])
    assert data["angle_incidence_secondary"] == pytest.approx([2.0, 4.0])
    assert data["primary_hit_x"][0] == pytest.approx(3.0)
    assert data["secondary_hit_y"][0] == pytest.approx(-0.4)
    names = {file.name for file in files}
    assert names >= {"run_incident_angles_name.sh", "incident_angles_stars_name.lis"}
    assert ("incident_angles_photons_name.lis" in names) is not keep_files


def test_incidence_angles_run_failure(backend, tmp_test_directory, monkeypatch):
    def fail(*_):
        raise backend_module.job_manager.JobExecutionError("failed")

    monkeypatch.setattr(backend_module.job_manager, "submit", fail)
    with pytest.raises(RuntimeError, match="Incident angles run failed, see log"):
        backend.incidence_angles(_request(), Path(tmp_test_directory), "name", False)


def test_imaging_list_rays_without_pixel_hits(tmp_test_directory):
    photons = Path(tmp_test_directory) / "filter.lis"
    photons.write_text(
        "# Column 1: Pixel number hit\n"
        "# Column 2: Angle to pixel normal [deg.]\n"
        "# Column 3: Angle of incidence at focal surface, w.r.t. optical axis [deg.]\n"
        "-1 5 30\n0 10 60\n",
        encoding="utf-8",
    )
    assert _ImagingList(False).read(photons) == {"angle_incidence_focal": [30, 60]}


def test_compute_incidence_angles_parsing(tmp_test_directory):
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

    out = _ImagingList(True)._compute_incidence_angles_from_imaging_list(pfile)
    assert "angle_incidence_focal_deg" in out
    assert out["angle_incidence_focal_deg"] == pytest.approx([42.5, 99.0])


def test_primary_valueerror_results_in_nan(tmp_test_directory):
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

    out = _ImagingList(True)._compute_incidence_angles_from_imaging_list(pfile)
    assert math.isnan(out["angle_incidence_primary_deg"][0])
    assert math.isclose(out["angle_incidence_secondary_deg"][0], 2.0, rel_tol=0.0, abs_tol=1e-12)


def test_header_driven_column_detection(tmp_test_directory):
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

    out = _ImagingList(True)._compute_incidence_angles_from_imaging_list(pfile)
    assert out["angle_incidence_focal_deg"] == pytest.approx([11.1])
    assert out["angle_incidence_primary_deg"] == pytest.approx([22.2])
    assert out["angle_incidence_secondary_deg"] == pytest.approx([33.3])


def test_find_column_indices_reflection_headers(tmp_test_directory):
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

    idx = _ImagingList(True)._find_column_indices(pfile)
    # Focal remains default (25), check XY indices converted to 0-based
    assert idx["prim_x"] == 9
    assert idx["prim_y"] == 10
    assert idx["sec_x"] == 11
    assert idx["sec_y"] == 12


def test_compute_angles_skips_primary_secondary_when_disabled(tmp_test_directory):
    # Minimal targeted test for the early return when primary/secondary are disabled
    calc = _ImagingList(False)

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
    assert math.isnan(_ImagingList._parse_float_with_nan(parts, 5))
    # Negative index returns NaN
    assert math.isnan(_ImagingList._parse_float_with_nan(parts, -1))


def test_set_reflection_index_if_match_ignores_when_not_primary_or_secondary():
    # Desc that mentions an axis but not primary/secondary mirror should not modify indices
    calc = _ImagingList(True)
    indices = {"prim_x": 28, "prim_y": 29, "sec_x": 32, "sec_y": 33}
    desc = "x reflection point on mirror center [cm]"  # no 'primary mirror' or 'secondary mirror'
    calc._set_reflection_index_if_match(desc, 15, indices)
    assert indices == {"prim_x": 28, "prim_y": 29, "sec_x": 32, "sec_y": 33}


def test_set_reflection_index_if_match_requires_axis_token():
    # Desc that mentions primary mirror but no standalone x/y should not modify indices
    calc = _ImagingList(True)
    indices = {}
    desc = "reflection point on primary mirror [cm]"  # no 'x' or 'y' token
    calc._set_reflection_index_if_match(desc, 12, indices)
    assert indices == {}


def test_update_indices_reflection_skips_when_flag_disabled():
    # If calculate_primary_secondary_angles is False, reflection point handling should return early
    calc = _ImagingList(False)
    indices = {"prim_x": 28, "prim_y": 29}
    desc = "X reflection point on primary mirror [cm]"
    calc._update_indices_from_header_desc(desc.lower(), 15, indices)
    # unchanged
    assert indices == {"prim_x": 28, "prim_y": 29}


def test_update_indices_reflection_skips_when_missing_keyword():
    # With flag True but description not containing 'reflection point', nothing should change
    calc = _ImagingList(True)
    indices = {"sec_x": 32, "sec_y": 33}
    desc = "X point on secondary mirror [cm]"  # missing 'reflection point'
    calc._update_indices_from_header_desc(desc.lower(), 20, indices)
    assert indices == {"sec_x": 32, "sec_y": 33}


def test_find_column_indices_ignores_mirror_when_disabled(tmp_test_directory):
    # Even with headers present, when calculate_primary_secondary_angles is False,
    # only 'focal' should be returned/overridden.
    calc = _ImagingList(False)

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


def test_append_primary_secondary_angles_with_none_arrays():
    """Cover _append_primary_secondary_angles when arrays are None."""
    calc = _ImagingList(True)
    parts = ["0"] * 40
    col_idx = {"primary": 31, "secondary": 35}
    calc._append_primary_secondary_angles(parts, col_idx, None, None)


def test_append_primary_hit_geometry_with_none_arrays():
    """Cover _append_primary_hit_geometry when arrays are None."""
    calc = _ImagingList(True)
    parts = ["0"] * 40
    col_idx = {"prim_x": 28, "prim_y": 29}
    calc._append_primary_hit_geometry(parts, col_idx, None, None, None)


def test_append_secondary_hit_geometry_with_none_arrays():
    """Cover _append_secondary_hit_geometry when arrays are None."""
    calc = _ImagingList(True)
    parts = ["0"] * 40
    col_idx = {"sec_x": 32, "sec_y": 33}
    calc._append_secondary_hit_geometry(parts, col_idx, None, None, None)


def test_update_indices_from_header_desc_primary_secondary():
    """Cover _update_indices_from_header_desc with primary/secondary mirror headers."""
    calc = _ImagingList(True)
    indices = {"focal": 25}

    calc._update_indices_from_header_desc(
        "angle of incidence onto primary mirror [deg]", 32, indices
    )
    assert indices["primary"] == 31

    calc._update_indices_from_header_desc(
        "angle of incidence on secondary mirror [deg]", 36, indices
    )
    assert indices["secondary"] == 35


def test_simulate_creates_one_simulator_per_image(backend, models, mocker):
    simulator = mocker.patch.object(backend_module, "SimulatorRayTracing")
    requests = [_request(0.0), _request(1.0)]

    backend.simulate(requests, test=True, force=True, compress_photons=False)

    models.tel.write_sim_telarray_config_file.assert_called_once()
    assert simulator.call_count == 2
    config = simulator.call_args.kwargs["config_data"]
    assert config["off_axis_x"] == pytest.approx(1.0)
    assert config["seeds"] == ((11, 22),)
    assert config["photons_per_run"] == 1000
    assert config["parameter_changes"] == {}
    assert simulator.return_value._write_config is False
    simulator.return_value.run.assert_called_with(test=True)


@pytest.mark.parametrize(("max_workers", "expected_workers"), [(8, 8), (3, 3)])
def test_simulate_limits_workers(backend, mocker, max_workers, expected_workers):
    mocker.patch.object(backend_module, "SimulatorRayTracing")
    executor = mocker.patch.object(backend_module, "ThreadPoolExecutor")
    executor.return_value.__enter__.return_value.map.return_value = []

    backend.simulate([_request(float(i)) for i in range(10)], max_workers=max_workers)

    executor.assert_called_once_with(max_workers=expected_workers)


def test_simulate_propagates_failure(backend, mocker):
    simulator = mocker.patch.object(backend_module, "SimulatorRayTracing")
    simulator.return_value.run.side_effect = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        backend.simulate([_request(0.0), _request(1.0)], compress_photons=False)


def test_simulate_compresses_photon_lists(backend, mocker):
    mocker.patch.object(backend_module, "SimulatorRayTracing")
    request = _request()
    photons_file = backend._files(request)[0]
    photons_file.write_text("photons", encoding="utf-8")

    backend.simulate([request], compress_photons=True)

    assert not photons_file.exists()
    assert photons_file.with_suffix(".lis.gz").exists()


@pytest.mark.parametrize("save_photons", [False, True])
def test_psf_image_reads_and_removes_photon_lists(backend, mocker, save_photons):
    psf_image = mocker.patch.object(backend_module, "PSFImage")
    request = _request()
    files = backend._files(request)
    for file in files:
        file.write_text("data", encoding="utf-8")

    image, traced_structure = backend.psf_image(request, 2800.0, 0.8, save_photons=save_photons)

    psf_image.assert_called_once_with(focal_length=2800.0, containment_fraction=0.8)
    psf_image.return_value.process_photon_list.assert_called_once_with(files[0], False)
    assert image is psf_image.return_value
    assert traced_structure is False
    assert all(file.exists() is save_photons for file in files)


def test_psf_image_with_rx_prefers_the_compressed_list(backend, mocker):
    psf_image = mocker.patch.object(backend_module, "PSFImage")
    request = _request()
    files = backend._files(request)
    for file in files[:2]:
        file.write_text("data", encoding="utf-8")

    backend.psf_image(request, 2800.0, 0.8, use_rx=True)

    psf_image.return_value.process_photon_list.assert_called_once_with(files[1], True)


def test_file_names_of_single_mirrors(models):
    backend = SimtelBackend(models.tel, models.site, "test", 20.0, True)
    photons, compressed, stars = backend._files(_request(mirror_number=3))
    assert "mirror3" in photons.name
    assert "d10" not in photons.name
    assert compressed.name == photons.name + ".gz"
    assert stars.name.startswith("ray_tracing_stars")
