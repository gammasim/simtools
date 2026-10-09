#!/usr/bin/env python3
"""Tests for workflow context and command-line replacements."""

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
    execute.assert_called_once_with(context.args, run_time=[], replacements={"__VALUE__": "value"})
