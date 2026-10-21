import sys
from pathlib import Path

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


def test_main_uses_release_context_by_default(mocker):
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

    assert run_release.call_args.kwargs[CONTEXT_FILE] == str(
        Path(args[RELEASE_DIR]) / "context.yml"
    )


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
