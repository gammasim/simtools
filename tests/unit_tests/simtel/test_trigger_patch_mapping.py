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
        config_file.write_text(
            'camera_config_file = "camera-test.dat" # camera\n'
            'atmospheric_transmission = "atmosphere-North.dat"\n',
            encoding="utf-8",
        )

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


def _produce_output(command, **_kwargs):
    """Stand in for each program by writing its requested output."""
    option = {"sim_telarray": "-r", "read_cta": "-p", "pixled": "-o"}[Path(command[0]).name]
    Path(command[command.index(option) + 1]).write_bytes(b"generated output\n")


def test_run_trigger_patch_mapping_runs_simtel_tools(tmp_test_directory, monkeypatch, mocker):
    telescope, site = _models(tmp_test_directory)
    pixled, simtel, read_cta = _simtel_installation(tmp_test_directory, monkeypatch)
    submit = mocker.patch.object(mapping.job_manager, "submit", side_effect=_produce_output)

    files = mapping.run_trigger_patch_mapping(
        telescope,
        site,
        tmp_test_directory / "output",
        include_presum_slaves=True,
        photons_per_pixel=700,
        events_per_combination=2,
        run_number=3,
    )

    assert (
        files.camera_file == tmp_test_directory / "output/trigger-patches-LSTN-01-run3.camera.dat"
    )
    assert (
        files.camera_file.read_bytes()
        == (Path(tmp_test_directory) / "camera-test.dat").read_bytes()
    )
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
        "--altitude",
        "2150.0",
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
    assert "atmospheric_transmission=atmosphere-North.dat" in simtel_command
    assert simtel_command[-3:] == ["-r", str(files.simtel_file), str(files.iact_file)]
    assert not any(token.startswith("input_file=") for token in simtel_command)
    assert submit.call_args_list[2].args[0] == [
        str(read_cta),
        "-p",
        str(files.postscript_file),
        str(files.simtel_file),
    ]
    for index, call in enumerate(submit.call_args_list):
        assert call.kwargs == {
            "out_file": files.log_files[index * 2],
            "err_file": files.log_files[index * 2 + 1],
            "env": mapping.SIM_TELARRAY_ENV,
        }


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


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True, None, 2147483648])
def test_numeric_options_require_positive_integers(value):
    with pytest.raises(ValueError, match="positive 32-bit integers"):
        mapping._validate_options(value)


@pytest.mark.parametrize("name", ["../result.ps", "nested/result.ps", "/result.ps", "result.pdf"])
def test_output_name_rejects_directories_and_wrong_suffix(tmp_test_directory, name):
    with pytest.raises(ValueError, match="PostScript filename"):
        mapping._output_file(tmp_test_directory, name)


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_missing_executable_fails_before_model_export(
    tmp_test_directory, monkeypatch, mocker, missing_index
):
    telescope, site = _models(tmp_test_directory)
    executables = _simtel_installation(tmp_test_directory, monkeypatch)
    executables[missing_index].unlink()
    submit = mocker.patch.object(mapping.job_manager, "submit")
    with pytest.raises(FileNotFoundError):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    telescope.write_sim_telarray_config_file.assert_not_called()
    submit.assert_not_called()


def test_nonexecutable_program_fails_early(tmp_test_directory, monkeypatch):
    telescope, site = _models(tmp_test_directory)
    pixled, _, _ = _simtel_installation(tmp_test_directory, monkeypatch)
    pixled.chmod(0o644)
    with pytest.raises(PermissionError, match=str(pixled)):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)


@pytest.mark.parametrize("failed_step", [0, 1, 2])
def test_failed_step_stops_pipeline_and_retains_context(
    tmp_test_directory, monkeypatch, mocker, failed_step
):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)
    failure = mapping.job_manager.JobExecutionError("external failure")
    outputs = [_produce_output] * failed_step + [failure]

    def submit_step(command, **kwargs):
        step = outputs.pop(0)
        if isinstance(step, Exception):
            raise step
        step(command, **kwargs)

    submit = mocker.patch.object(mapping.job_manager, "submit", side_effect=submit_step)
    with pytest.raises(mapping.job_manager.JobExecutionError, match="external failure") as error:
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    assert error.value.__cause__ is failure
    assert submit.call_count == failed_step + 1
    for suffix in (
        "camera-test.dat",
        ".iact.gz",
        ".simtel.gz",
        ".ps",
        ".stdout.log",
        ".stderr.log",
    ):
        assert suffix in str(error.value)


@pytest.mark.parametrize("empty", [False, True])
def test_successful_exit_without_output_is_an_error(tmp_test_directory, monkeypatch, mocker, empty):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)

    def submit_step(command, **_kwargs):
        if empty:
            Path(command[command.index("-o") + 1]).touch()

    submit = mocker.patch.object(mapping.job_manager, "submit", side_effect=submit_step)
    with pytest.raises(mapping.job_manager.JobExecutionError, match="Missing or empty output"):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    submit.assert_called_once()


def test_defaults_and_site_altitude(tmp_test_directory, monkeypatch, mocker):
    telescope, site = _models(tmp_test_directory)
    site.get_parameter_value_with_unit.return_value = 1.8 * u.km
    _simtel_installation(tmp_test_directory, monkeypatch)
    submit = mocker.patch.object(mapping.job_manager, "submit", side_effect=_produce_output)
    files = mapping.run_trigger_patch_mapping(
        telescope, site, tmp_test_directory, output_name="custom.ps"
    )
    assert files.camera_file == tmp_test_directory / "camera-test.dat"
    assert files.postscript_file == tmp_test_directory / "custom.ps"
    command = submit.call_args_list[0].args[0]
    assert "--slaves" not in command
    assert command[command.index("--altitude") + 1] == "1800.0"
    assert "Altitude=1800.0" in submit.call_args_list[1].args[0]


@pytest.mark.parametrize("missing_step", [0, 1, 2])
def test_repeat_run_cannot_accept_previous_outputs(
    tmp_test_directory, monkeypatch, mocker, missing_step
):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)
    submit = mocker.patch.object(mapping.job_manager, "submit", side_effect=_produce_output)
    files = mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    outputs = (files.iact_file, files.simtel_file, files.postscript_file)
    for output in outputs:
        output.write_bytes(b"previous run\n")
    submit.reset_mock()

    def submit_step(command, **kwargs):
        index = submit.call_count - 1
        assert not outputs[index].exists()
        if index != missing_step:
            _produce_output(command, **kwargs)

    submit.side_effect = submit_step
    with pytest.raises(mapping.job_manager.JobExecutionError, match="Missing or empty output"):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    assert submit.call_count == missing_step + 1
    assert not outputs[missing_step].exists()
    for output in outputs[:missing_step]:
        assert output.read_bytes() == b"generated output\n"
    for output in outputs[missing_step + 1 :]:
        assert output.read_bytes() == b"previous run\n"


@pytest.mark.parametrize("text", ["# no camera\n", "camera_config_file =\n"])
def test_exported_config_requires_camera_setting(tmp_test_directory, text):
    telescope, site = _models(tmp_test_directory)
    telescope.write_sim_telarray_config_file.side_effect = None
    telescope.config_file_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="camera_config_file"):
        mapping._export_camera_file(telescope, site)
