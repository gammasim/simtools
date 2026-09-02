"""Tests for output-validation orchestration."""

from pathlib import Path

from simtools.testing.output_validation import runner


def test_versions_match_requires_an_intersection():
    """Match scalar and sequence model-version filters."""
    assert runner.versions_match(None, "7.0.0")
    assert runner.versions_match("7.0.0", ["6.0.0", "7.0.0"])
    assert not runner.versions_match("7.0.0", "6.0.0")


def test_runner_dispatches_validations_in_order(tmp_test_directory, mocker):
    """Dispatch configured validations in declaration order."""
    output = Path(tmp_test_directory) / "result.txt"
    output.touch()
    dispatch = mocker.patch("simtools.testing.output_validation.runner.run_validator")
    config = {
        "configuration": {"output_path": tmp_test_directory},
        "integration_tests": [
            {
                "test_outputs": [
                    {
                        "path_descriptor": "output_path",
                        "file": output.name,
                        "validations": [{"type": "format", "format": "txt"}, {"type": "log"}],
                    }
                ]
            }
        ],
    }

    runner.validate_application_output(config)

    assert [call.args[1]["type"] for call in dispatch.call_args_list] == ["format", "log"]


def test_runner_applies_model_version_filters(tmp_test_directory, mocker):
    """Skip output validations for non-matching model versions."""
    output = Path(tmp_test_directory) / "result.txt"
    output.touch()
    dispatch = mocker.patch("simtools.testing.output_validation.runner.run_validator")
    config = {
        "configuration": {"output_path": tmp_test_directory},
        "integration_tests": [
            {
                "test_outputs": [
                    {
                        "path_descriptor": "output_path",
                        "file": output.name,
                        "model_versions": ["7.0.0"],
                        "validations": [{"type": "format", "format": "txt"}],
                    }
                ]
            }
        ],
    }

    runner.validate_application_output(config, from_config_file="6.0.0")
    dispatch.assert_not_called()
    runner.validate_application_output(config, from_config_file="7.0.0")
    dispatch.assert_called_once()


def test_runner_creates_one_reader_for_model_validations(tmp_test_directory, mocker):
    """Reuse one selected reader for all model-parameter validations."""
    output = Path(tmp_test_directory) / "result.json"
    output.touch()
    reader = mocker.Mock()
    create_reader = mocker.patch.object(runner, "create_model_reader", return_value=reader)
    dispatch = mocker.patch.object(runner, "run_validator")
    config = {
        "configuration": {},
        "integration_tests": [
            {
                "test_outputs": [
                    {
                        "path_descriptor": "output_path",
                        "file": output.name,
                        "validations": [
                            {"type": "model_parameter"},
                            {"type": "model_parameter"},
                        ],
                    }
                ]
            }
        ],
    }
    config["configuration"]["output_path"] = tmp_test_directory

    runner.validate_application_output(config)

    create_reader.assert_called_once_with(None)
    assert all(
        call.args[2]["configuration"] is config["configuration"] for call in dispatch.call_args_list
    )
    assert all(call.args[2]["model_reader"] is reader for call in dispatch.call_args_list)


def test_runner_skips_when_top_level_versions_do_not_match(mocker):
    """Skip all validations when the requested model versions do not match."""
    dispatch = mocker.patch("simtools.testing.output_validation.runner.run_validator")

    runner.validate_application_output(
        {"configuration": {}, "integration_tests": []},
        from_command_line="7.0.0",
        from_config_file="6.0.0",
    )

    dispatch.assert_not_called()
