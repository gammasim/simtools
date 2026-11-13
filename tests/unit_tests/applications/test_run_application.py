#!/usr/bin/env python3
"""Tests for workflow context and command-line replacements."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from simtools.applications import run_application


def test_context_and_cli_precedence(tmp_test_directory):
    root = Path(tmp_test_directory)
    context = root / "context.yml"
    context.write_text(yaml.safe_dump({"__VALUE__": "context", "__NUMBER__": 3}), encoding="utf-8")
    assert run_application._load_replacements(
        {
            "context_file": str(context),
            "config_file": str(root / "workflow.yml"),
            "replace": ["__VALUE__=cli=value"],
        }
    ) == {"__VALUE__": "cli=value", "__NUMBER__": "3", "__CONFIG_DIRECTORY__": str(root)}


@pytest.mark.parametrize("replacement", ["missing-equals", " =value"])
def test_invalid_replacement(replacement):
    with pytest.raises(ValueError, match="Replacement"):
        run_application._load_replacements({"replace": [replacement]})


def test_invalid_context_mapping(tmp_test_directory):
    path = Path(tmp_test_directory) / "context.yml"
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        run_application._load_replacements({"context_file": str(path)})


def test_main_forwards_context(mocker):
    context = mocker.Mock(args={"replace": ["__VALUE__=value"]}, run_time=[])
    mocker.patch.object(run_application.APPLICATION.__class__, "start", return_value=context)
    execute = mocker.patch.object(run_application, "run_applications")
    run_application.main()
    execute.assert_called_once_with(
        context.args, run_time=[], replacements={"__VALUE__": "value"}, context=context
    )


def test_runtime_model_source_does_not_require_a_host_reader(
    tmp_test_directory, simtools_root_path
):
    root = Path(tmp_test_directory)
    engine = root / "fake-container-engine"
    engine.write_text("#!/bin/sh\nexit 0\n")
    engine.chmod(0o755)
    config = root / "workflow.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "applications": [{"application": "echo", "run_application": False}],
                "runtime_environment": {
                    "container_engine": str(engine),
                    "image": "example-image",
                    "simulation_models": {"git_path": "/only/in/container", "git_revision": "v1"},
                },
            }
        )
    )
    env_file = root / "empty.env"
    env_file.write_text("")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SIMTOOLS_SIMULATION_MODELS_")
    }
    env["PYTHONPATH"] = str(Path(simtools_root_path) / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simtools.applications.run_application",
            "--config_file",
            str(config),
            "--env_file",
            str(env_file),
            "--disable_log_file",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
