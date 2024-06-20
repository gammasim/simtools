#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.runners.simtel_runner import SimtelExecutionError, SimtelRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def simtel_runner(simtel_path):
    simtel_runner = SimtelRunner(simtel_path=simtel_path, label="test")
    return simtel_runner


def test_repr(simtel_runner):
    assert repr(simtel_runner) == "SimtelRunner(label=test)\n"


def test_prepare_run_script(simtel_runner, tmp_test_directory):
    simtel_runner._base_directory = Path(tmp_test_directory)
    simtel_runner.prepare_run_script(
        test=True, input_file="test", run_number=1, extra_commands=None
    )
    assert simtel_runner._script_file.exists()
    with open(simtel_runner._script_file) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "RUNTIME" in script_content
        assert "test-1" in script_content

    simtel_runner.runs_per_set = 5
    simtel_runner.prepare_run_script(
        test=False, input_file="test", run_number=5, extra_commands="extra"
    )
    with open(simtel_runner._script_file) as f:
        script_content = f.read()
        assert "test-5" in script_content
        assert script_content.count("test-5") == 5
        assert "extra" in script_content


def test_run(simtel_runner, caplog):
    with caplog.at_level(logging.INFO):
        simtel_runner.run(test=False, force=False)
    assert "Skipping because output exists and force = False" in caplog.text
    with caplog.at_level(logging.INFO):
        with pytest.raises(SimtelExecutionError):
            simtel_runner.run(test=True, force=True, input_file="test", run_number=5)
    assert "Running (test) with command: test-5" in caplog.text
    simtel_runner.runs_per_set = 5
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(SimtelExecutionError):
            simtel_runner.run(test=False, force=True, input_file="test", run_number=5)
    assert "Running (5x) with command: test-5" in caplog.text


def test_run_simtel_and_check_output(simtel_runner):

    with pytest.raises(SimtelExecutionError):
        simtel_runner._run_simtel_and_check_output("test-5")
    simtel_runner._run_simtel_and_check_output("echo test")


def test_simtel_execution_error(simtel_runner):
    with pytest.raises(SimtelExecutionError):
        simtel_runner._raise_simtel_error()


def test_config_option(simtel_runner):
    assert simtel_runner._config_option("test", "value") == " -C test=value"
    assert simtel_runner._config_option("test", "value", weak_option=True) == " -W test=value"
    assert simtel_runner._config_option("test", "value", weak_option=False) == " -C test=value"
    assert simtel_runner._config_option("test") == " -C test"


def test_raise_simtel_error(simtel_runner):
    with pytest.raises(SimtelExecutionError, match="Simtel log file does not exist."):
        simtel_runner._raise_simtel_error()


def test_make_run_command(simtel_runner, caplog):
    with caplog.at_level(logging.DEBUG):
        assert simtel_runner._make_run_command("test", "5") == "test-5"
    assert "make_run_command is being called from the base class" in caplog.text


def test_shall_run(simtel_runner, caplog):
    with caplog.at_level(logging.DEBUG):
        assert not simtel_runner._shall_run()
    assert "shall_run is being called from the base class" in caplog.text
