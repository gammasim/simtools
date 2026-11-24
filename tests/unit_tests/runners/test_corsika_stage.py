"""Tests for the CORSIKA simulation stage."""

import pathlib
import shlex

import astropy.units as u
import pytest

from simtools.runners.corsika_stage import CorsikaStage
from simtools.runners.stages import StageInputs, StreamTo


def test_corsika_stage_directories(corsika_stage_mock_array_model):
    stage = corsika_stage_mock_array_model

    assert "corsika_sim_telarray" not in str(stage.runner_service.directory)
    assert "corsika" in str(stage.runner_service.directory)
    assert isinstance(stage.runner_service.directory, pathlib.Path)


def test_instances_use_first_shower_configuration(corsika_config_mock_array_model):
    inputs = StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(corsika_config_mock_array_model, None),
        args={"sequential": True, "save_corsika_output": True},
    )

    (stage,) = CorsikaStage.instances(inputs, label="test")

    assert stage.corsika_config is corsika_config_mock_array_model
    assert stage.sequential
    assert stage.save_output


def test_prepare_writes_corsika_lines(corsika_stage_mock_array_model):
    lines = corsika_stage_mock_array_model.prepare(run_number=1)

    assert lines[0].startswith("export CORSIKA_DATA=")
    assert lines[-2] == "# Running corsika"
    assert "corsika" in lines[-1]
    assert "corsika_output" in corsika_stage_mock_array_model.file_list


def test_prepare_rejects_input_and_invalid_runs(corsika_stage_mock_array_model):
    with pytest.raises(ValueError, match="CORSIKA does not read an input product"):
        corsika_stage_mock_array_model.prepare(run_number=1, input_file="input")
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        corsika_stage_mock_array_model.prepare(run_number=-2)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        corsika_stage_mock_array_model.prepare(run_number="test")


def test_prepare_streams_to_downstream_commands(corsika_stage_mock_array_model):
    stage = CorsikaStage(
        corsika_stage_mock_array_model.corsika_config,
        label="test-stream",
        sequential=True,
        context=corsika_stage_mock_array_model.context,
    )

    stage.prepare(run_number=1, output=StreamTo(("first command", "second command")))

    config = stage.stream_service.get_file_name("multi_pipe_config", run_number=1)
    assert config.read_text(encoding="utf-8") == "first command\n\nsecond command\n\n"
    script = stage.stream_service.get_file_name("multi_pipe_script", run_number=1)
    script_text = script.read_text(encoding="utf-8")
    assert "multipipe_corsika" in script_text
    assert "--sequential" in script_text
    assert "multi_pipe_config" in stage.file_list


def test_prepare_stream_saves_corsika_output(corsika_stage_mock_array_model):
    stage = CorsikaStage(
        corsika_stage_mock_array_model.corsika_config,
        label="test-save",
        save_output=True,
        context=corsika_stage_mock_array_model.context,
    )

    stage.prepare(run_number=1, output=StreamTo(("command",)))

    config_text = stage.stream_service.get_file_name("multi_pipe_config", run_number=1).read_text(
        encoding="utf-8"
    )
    assert config_text.count("zstd >") == 1
    assert f"zstd > {stage.output_file(1)}" in config_text


def test_corsika_executable(corsika_stage_mock_array_model):
    stage = corsika_stage_mock_array_model
    stage.corsika_config.use_curved_atmosphere = False
    assert stage._corsika_executable() == stage.context.software.corsika_path / "corsika"
    stage.corsika_config.use_curved_atmosphere = True
    assert stage._corsika_executable() == stage.context.software.corsika_path / "corsika_curved"


@pytest.mark.parametrize("transition_energy", [None, 120 * u.GeV])
def test_environment(corsika_stage_mock_array_model, transition_energy):
    stage = corsika_stage_mock_array_model
    stage.context.software.he_interaction = "qgs3"
    stage.context.software.le_interaction = "urqmd"
    stage.context = stage.context.replace(
        args={"corsika_hadronic_transition_energy": transition_energy},
    )

    expected = {
        "SIMTOOLS_CORSIKA_HE_INTERACTION": "qgs3",
        "SIMTOOLS_CORSIKA_LE_INTERACTION": "urqmd",
    }
    if transition_energy is not None:
        expected["SIMTOOLS_CORSIKA_HADRONIC_TRANSITION_ENERGY_GEV"] = "120.0"
    assert stage.environment() == expected


def test_resources(corsika_stage_mock_array_model):
    assert corsika_stage_mock_array_model.get_resources(runtime=2.5)["runtime"] == pytest.approx(
        2.5
    )


def test_output_paths_are_shell_quoted(corsika_stage_mock_array_model, tmp_test_directory):
    stage = corsika_stage_mock_array_model
    stage.runner_service.directory = pathlib.Path(tmp_test_directory) / "space ; $(touch marker)"
    stage.runner_service.directory.mkdir(parents=True)
    stage.save_output = True

    lines = stage.prepare(run_number=1, output=StreamTo(("command",)))

    output = stage.output_file(1)
    assert lines[0] == f"export CORSIKA_DATA={shlex.quote(str(output.parent))}"
    config = stage.stream_service.get_file_name("multi_pipe_config", run_number=1)
    assert config.read_text(encoding="utf-8").splitlines()[-1] == (
        f"zstd > {shlex.quote(str(output))}"
    )


def test_output_directory_rejects_line_breaks(corsika_stage_mock_array_model, tmp_test_directory):
    stage = corsika_stage_mock_array_model
    stage.runner_service.directory = pathlib.Path(tmp_test_directory) / "invalid\npath"
    with pytest.raises(ValueError, match="output directory cannot contain line breaks"):
        stage.prepare(run_number=1)
