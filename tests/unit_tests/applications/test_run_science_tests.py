import sys

from simtools.applications import run_science_tests

RELEASE_DIR = "release_dir"
CONTEXT_FILE = "context_file"
RELEASE_PATH = "release/science_tests"
BASELINE_VERSION_KEY = "baseline_version"
MODEL_REVISION_KEY = "simulation_models_git_revision"
APPLICATION_START = "simtools.application.definition.ApplicationDefinition.start"
PARSED_BASELINE_VERSION = "v0.37.0"
PARSED_MODEL_REVISION = "v0.19.0"
SETUP_KEY = "setup"


def test_main_forwards_optional_context(mocker):
    args = {
        RELEASE_DIR: RELEASE_PATH,
        CONTEXT_FILE: None,
    }
    mocker.patch(
        APPLICATION_START,
        return_value=mocker.Mock(args=args),
    )
    run_release = mocker.patch.object(run_science_tests, "run_release")

    run_science_tests.main()

    assert run_release.call_args.args == (RELEASE_PATH,)
    assert run_release.call_args.kwargs[CONTEXT_FILE] is None


def test_parser_allows_setup_release_settings(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_science_tests.py",
            "--release_dir",
            RELEASE_PATH,
            "--setup",
            "--baseline_version",
            PARSED_BASELINE_VERSION,
            "--simulation_models_git_revision",
            PARSED_MODEL_REVISION,
        ],
    )

    args = run_science_tests.APPLICATION._parse()

    assert args[CONTEXT_FILE] is None
    assert args[SETUP_KEY] is True
    assert args[BASELINE_VERSION_KEY] == PARSED_BASELINE_VERSION
    assert args[MODEL_REVISION_KEY] == PARSED_MODEL_REVISION


def test_parser_accepts_run_label(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_science_tests.py", "--release_dir", RELEASE_PATH, "--run_label", "non-linear"],
    )
    assert run_science_tests.APPLICATION._parse()["run_label"] == "non-linear"


def test_setup_forwards_release_settings(mocker):
    args = {
        RELEASE_DIR: RELEASE_PATH,
        SETUP_KEY: True,
        BASELINE_VERSION_KEY: "v0.38.0",
        MODEL_REVISION_KEY: "v0.18.0",
    }
    mocker.patch(
        APPLICATION_START,
        return_value=mocker.Mock(args=args),
    )
    setup = mocker.patch.object(run_science_tests, "setup_release")
    run_science_tests.main()
    setup.assert_called_once_with(
        args[RELEASE_DIR],
        template_dir=None,
        context_file=None,
        baseline_version=args[BASELINE_VERSION_KEY],
        simulation_models_git_revision=args[MODEL_REVISION_KEY],
    )


def test_parser_accepts_max_workers(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_science_tests.py", "--release_dir", RELEASE_PATH, "--max_workers", "8"]
    )
    assert run_science_tests.APPLICATION._parse()["max_workers"] == 8


def test_max_workers_rejects_one_for_separate_intensities(capsys):
    import argparse

    import pytest

    with pytest.raises(SystemExit, match="2"):
        run_science_tests._post_parse({"max_workers": 1}, {}, argparse.ArgumentParser())
    assert "must be at least 2" in capsys.readouterr().err
