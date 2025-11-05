#!/usr/bin/python3

import logging
import pathlib
import re

import pytest

logger = logging.getLogger()


@pytest.fixture
def bin_bash():
    """Path to bash."""
    return "/usr/bin/env bash"


@pytest.fixture
def pfp_command():
    """Basic pfp command."""
    return "sim_telarray/bin/pfp"


def test_corsika_runner(corsika_runner_mock_array_model):
    cr = corsika_runner_mock_array_model
    assert "corsika_sim_telarray" not in str(cr._directory["data"])
    assert "corsika" in str(cr._directory["data"])
    assert isinstance(cr._directory["data"], pathlib.Path)


def test_prepare_run_script(corsika_runner_mock_array_model, bin_bash, pfp_command):
    # No run number is given
    script = corsika_runner_mock_array_model.prepare_run_script()

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert bin_bash in script_content
        assert "corsika_autoinputs" in script_content
        assert pfp_command in script_content

    # Run number is given
    script = corsika_runner_mock_array_model.prepare_run_script(run_number=3)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert bin_bash in script_content
        assert "corsika_autoinputs" in script_content
        assert pfp_command in script_content
        assert "-R 3" in script_content


def test_prepare_run_script_with_input_file(corsika_runner_mock_array_model, caplog):
    with caplog.at_level("WARNING"):
        corsika_runner_mock_array_model.prepare_run_script(input_file="test")
    assert any(
        "input_file parameter is not used in CorsikaRunner.prepare_run_script" in message
        for message in caplog.messages
    )


def test_prepare_run_script_with_invalid_run(corsika_runner_mock_array_model):
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        _ = corsika_runner_mock_array_model.prepare_run_script(run_number=-2)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        _ = corsika_runner_mock_array_model.prepare_run_script(run_number="test")


def test_prepare_run_script_with_extra(
    corsika_runner_mock_array_model, file_has_text, bin_bash, pfp_command
):
    extra = ["testing", "testing-extra-2"]
    script = corsika_runner_mock_array_model.prepare_run_script(run_number=3, extra_commands=extra)

    assert file_has_text(script, "testing-extra-2")
    with open(script) as f:
        script_content = f.read()
        assert bin_bash in script_content
        assert "corsika_autoinputs" in script_content
        assert pfp_command in script_content


def test_prepare_run_script_without_pfp(corsika_runner_mock_array_model, bin_bash, pfp_command):
    script = corsika_runner_mock_array_model.prepare_run_script(use_pfp=False)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert bin_bash in script_content
        assert "corsika_autoinputs" in script_content
        assert pfp_command not in script_content


def test_get_pfp_command(corsika_runner_mock_array_model, pfp_command):
    command = corsika_runner_mock_array_model._get_pfp_command(
        "input_tmp_file", "corsika_input_file"
    )
    assert pfp_command in command
    assert "> input_tmp_file" in command


def test_get_autoinputs_command(corsika_runner_mock_array_model, caplog):
    autoinputs_command = corsika_runner_mock_array_model._get_autoinputs_command(
        run_number=3, input_tmp_file="tmp_file"
    )
    assert "corsika_autoinputs" in autoinputs_command
    assert "-R 3" in autoinputs_command
    assert "--keep-seeds" not in autoinputs_command
    corsika_runner_mock_array_model._keep_seeds = True
    with caplog.at_level("WARNING"):
        autoinputs_command_with_seeds = corsika_runner_mock_array_model._get_autoinputs_command(
            run_number=3, input_tmp_file="tmp_file"
        )
        assert any(
            "Using --keep-seeds option in corsika_autoinputs is not recommended" in message
            for message in caplog.messages
        )
    assert "--keep-seeds" in autoinputs_command_with_seeds


def test_get_autoinputs_command_with_multipipe(corsika_runner_mock_array_model):
    # Enable multipipe and ensure log file name is prefixed accordingly
    corsika_runner_mock_array_model._use_multipipe = True
    cmd = corsika_runner_mock_array_model._get_autoinputs_command(
        run_number=7, input_tmp_file="tmp"
    )
    assert "multipipe_" in cmd


def test_get_resources(corsika_runner_mock_array_model):
    with pytest.raises(FileNotFoundError):
        corsika_runner_mock_array_model.get_resources()


def test_get_file_name(corsika_runner_mock_array_model):
    with pytest.raises(
        ValueError, match=re.escape("simulation_software (test) is not supported in CorsikaRunner")
    ):
        corsika_runner_mock_array_model.get_file_name(simulation_software="test")
