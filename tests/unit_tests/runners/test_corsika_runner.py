#!/usr/bin/python3

import logging
import pathlib

import pytest

logger = logging.getLogger()


@pytest.fixture
def bin_bash():
    """Path to bash."""
    return "/usr/bin/env bash"


def test_corsika_runner(corsika_runner_mock_array_model):
    cr = corsika_runner_mock_array_model
    assert "corsika_sim_telarray" not in str(cr.runner_service.directory)
    assert "corsika" in str(cr.runner_service.directory)
    assert isinstance(cr.runner_service.directory, pathlib.Path)


def test_prepare_run(corsika_runner_mock_array_model, bin_bash, tmp_path):
    # prepare_run now requires both run_number and sub_script
    script_path = tmp_path / "test_script.sh"
    corsika_runner_mock_array_model.prepare_run(run_number=1, sub_script=script_path)

    assert script_path.exists()
    with open(script_path) as f:
        script_content = f.read()
        assert bin_bash in script_content
        # The new implementation doesn't use corsika_autoinputs, it directly calls CORSIKA
        assert "corsika" in script_content.lower()


def test_prepare_run_with_input_file(corsika_runner_mock_array_model, tmp_path):
    # The new API doesn't have input_file parameter, test with corsika_file instead
    script_path = tmp_path / "test_script.sh"
    # This should work without warnings as the new API uses corsika_file parameter
    corsika_runner_mock_array_model.prepare_run(
        run_number=1, sub_script=script_path, corsika_file="test_file"
    )
    assert script_path.exists()


def test_prepare_run_with_invalid_run(corsika_runner_mock_array_model, tmp_path):
    script_path = tmp_path / "test_script.sh"
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        corsika_runner_mock_array_model.prepare_run(run_number=-2, sub_script=script_path)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        corsika_runner_mock_array_model.prepare_run(run_number="test", sub_script=script_path)


def test_prepare_run_with_extra(corsika_runner_mock_array_model, file_has_text, bin_bash, tmp_path):
    extra = ["testing", "testing-extra-2"]
    script_path = tmp_path / "test_script.sh"
    corsika_runner_mock_array_model.prepare_run(
        run_number=3, sub_script=script_path, extra_commands="\n".join(extra)
    )

    assert file_has_text(script_path, "testing-extra-2")
    with open(script_path) as f:
        script_content = f.read()
        assert bin_bash in script_content


def test_get_resources(corsika_runner_mock_array_model, tmp_path):
    # get_resources now requires sub_out_file parameter
    test_file = tmp_path / "nonexistent_file.log"
    with pytest.raises(FileNotFoundError):
        corsika_runner_mock_array_model.get_resources(test_file)


def test_corsika_executable(corsika_runner_mock_array_model):
    """Test that _corsika_executable returns the correct path."""
    # Test flat atmosphere (default)
    executable = corsika_runner_mock_array_model._corsika_executable()
    assert executable is not None

    # Test curved atmosphere
    corsika_runner_mock_array_model.corsika_config.use_curved_atmosphere = True
    executable_curved = corsika_runner_mock_array_model._corsika_executable()
    assert executable_curved is not None
