#!/usr/bin/python3

import logging
import pathlib
import re

import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_corsika_runner(corsika_runner):
    cr = corsika_runner
    assert "corsika_simtel" not in str(cr._directory["output"])
    assert "corsika" in str(cr._directory["output"])
    assert isinstance(cr._directory["data"], pathlib.Path)


def test_prepare_run_script(corsika_runner):
    # No run number is given
    script = corsika_runner.prepare_run_script()

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content

    # Run number is given
    script = corsika_runner.prepare_run_script(run_number=3)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content
        assert "-R 3" in script_content


def test_prepare_run_script_with_input_file(corsika_runner, caplog):
    with caplog.at_level("WARNING"):
        corsika_runner.prepare_run_script(input_file="test")
    assert any(
        "input_file parameter is not used in CorsikaRunner.prepare_run_script" in message
        for message in caplog.messages
    )


def test_prepare_run_script_with_invalid_run(corsika_runner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_runner.prepare_run_script(run_number=run)


def test_prepare_run_script_with_extra(corsika_runner, file_has_text):
    extra = ["testing", "testing-extra-2"]
    script = corsika_runner.prepare_run_script(run_number=3, extra_commands=extra)

    assert file_has_text(script, "testing-extra-2")
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content


def test_prepare_run_script_without_pfp(corsika_runner):
    script = corsika_runner.prepare_run_script(use_pfp=False)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content


def test_get_pfp_command(corsika_runner):
    pfp_command = corsika_runner._get_pfp_command("input_tmp_file", "corsika_input_file")
    assert "sim_telarray/bin/pfp" in pfp_command
    assert "> input_tmp_file" in pfp_command


def test_get_autoinputs_command(corsika_runner):
    autoinputs_command = corsika_runner._get_autoinputs_command(
        run_number=3, input_tmp_file="tmp_file"
    )
    assert "corsika_autoinputs" in autoinputs_command
    assert "-R 3" in autoinputs_command
    assert "--keep-seeds" not in autoinputs_command
    corsika_runner._keep_seeds = True
    autoinputs_command_with_seeds = corsika_runner._get_autoinputs_command(
        run_number=3, input_tmp_file="tmp_file"
    )
    assert "--keep-seeds" in autoinputs_command_with_seeds


def test_get_resources(corsika_runner):
    with pytest.raises(FileNotFoundError):
        corsika_runner.get_resources()


def test_get_file_name(corsika_runner):
    with pytest.raises(
        ValueError, match=re.escape("simulation_software (test) is not supported in CorsikaRunner")
    ):
        corsika_runner.get_file_name(simulation_software="test")
