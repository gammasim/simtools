"""Tests for sim_telarray trigger-patch mapping."""

from pathlib import Path
from unittest.mock import Mock

import astropy.units as u
import pytest

import simtools.simtel.trigger_patch_mapping as mapping


def _models(tmp_test_directory):
    """Return models whose exported configuration defines a camera file."""
    tmp_test_directory = Path(tmp_test_directory)
    config_file = tmp_test_directory / "telescope.cfg"
    camera_file = tmp_test_directory / "camera-test.dat"
    camera_file.write_text("Pixel 0\nMajorityTrigger 1 of 0\n", encoding="utf-8")
    telescope = Mock(name="LSTN-01")
    telescope.name = "LSTN-01"
    telescope.config_file_path = config_file
    telescope.config_file_directory = tmp_test_directory

    def write_config(additional_models):
        assert additional_models is site
        config_file.write_text("camera_config_file = camera-test.dat\n", encoding="utf-8")

    telescope.write_sim_telarray_config_file.side_effect = write_config
    site = Mock()
    site.parameters = {"atmospheric_transmission": {"value": "atmosphere.ecsv"}}
    site.get_parameter_value.side_effect = lambda name: {
        "atmospheric_transmission": "atmosphere.ecsv"
    }[name]
    site.get_parameter_value_with_unit.return_value = 2150 * u.m
    return telescope, site


def _simtel_installation(tmp_test_directory, monkeypatch):
    """Configure fake sim_telarray executables."""
    tmp_test_directory = Path(tmp_test_directory)
    simtel_path = tmp_test_directory / "sim_telarray"
    pixled = simtel_path / "LightEmission" / "pixled"
    simtel = simtel_path / "bin" / "sim_telarray"
    read_cta = simtel_path / "bin" / "read_cta"
    for executable in (pixled, simtel, read_cta):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
        executable.chmod(0o755)
    monkeypatch.setattr(mapping.settings.config, "_sim_telarray_path", str(simtel_path))
    monkeypatch.setattr(mapping.settings.config, "_sim_telarray_exe", "sim_telarray")
    return pixled, simtel, read_cta


def test_run_trigger_patch_mapping_runs_simtel_tools(tmp_test_directory, monkeypatch, mocker):
    telescope, site = _models(tmp_test_directory)
    pixled, simtel, read_cta = _simtel_installation(tmp_test_directory, monkeypatch)
    submit = mocker.patch.object(mapping.job_manager, "submit")

    files = mapping.run_trigger_patch_mapping(
        telescope,
        site,
        tmp_test_directory / "output",
        include_presum_slaves=True,
        photons_per_pixel=700,
        events_per_combination=2,
        run_number=3,
    )

    assert files.camera_file == tmp_test_directory / "camera-test.dat"
    assert files.postscript_file.name == "trigger-patches-LSTN-01-run3.ps"
    assert len(files.log_files) == 6
    assert submit.call_count == 3
    pixled_command = submit.call_args_list[0].args[0]
    assert pixled_command == [
        str(pixled),
        "--camera-file",
        str(files.camera_file),
        "--use-trg-all",
        "--require",
        "2",
        "--photons",
        "700",
        "--events",
        "2",
        "--run",
        "3",
        "-o",
        str(files.iact_file),
        "--slaves",
    ]
    simtel_command = submit.call_args_list[1].args[0]
    assert simtel_command[:6] == [
        str(simtel),
        "-c",
        str(telescope.config_file_path),
        f"-I{telescope.config_file_directory}",
        "-DNUM_TELESCOPES=1",
        "-C",
    ]
    assert "Altitude=2150.0" in simtel_command
    assert "atmospheric_transmission=atmospheric_transmission-telescope.dat" in simtel_command
    assert submit.call_args_list[2].args[0] == [
        str(read_cta),
        "-p",
        str(files.postscript_file),
        str(files.simtel_file),
    ]


def test_run_trigger_patch_mapping_rejects_invalid_options(tmp_test_directory):
    telescope, site = _models(tmp_test_directory)

    with pytest.raises(ValueError, match="positive"):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory, required_pixels=0)
    with pytest.raises(ValueError, match="PostScript filename"):
        mapping.run_trigger_patch_mapping(
            telescope, site, tmp_test_directory, output_name="result.pdf"
        )


def test_run_trigger_patch_mapping_requires_executables(tmp_test_directory, monkeypatch):
    telescope, site = _models(tmp_test_directory)
    monkeypatch.setattr(
        mapping.settings.config, "_sim_telarray_path", str(tmp_test_directory / "missing")
    )

    with pytest.raises(FileNotFoundError, match="pixled"):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)


def test_export_camera_file_requires_generated_camera(tmp_test_directory):
    telescope, site = _models(tmp_test_directory)
    (Path(tmp_test_directory) / "camera-test.dat").unlink()

    with pytest.raises(FileNotFoundError, match="Generated camera"):
        mapping._export_camera_file(telescope, site)
