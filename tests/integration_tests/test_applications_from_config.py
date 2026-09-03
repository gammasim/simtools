#!/usr/bin/python3
# Integration tests for applications from config file

import copy
import importlib
import logging
import os
import subprocess
from pathlib import Path

import pytest

from simtools.testing import configuration, helpers, log_inspector, validate_output

logger = logging.getLogger()
_MONGODB_ENVIRONMENT = (
    "SIMTOOLS_DB_SERVER",
    "SIMTOOLS_DB_API_USER",
    "SIMTOOLS_DB_API_PW",
    "SIMTOOLS_DB_API_PORT",
    "SIMTOOLS_DB_SIMULATION_MODEL",
)
_MONGODB_MODEL_TAG_ENVIRONMENT = (
    "SIMTOOLS_DB_SIMULATION_MODEL_TAG",
    "SIMTOOLS_DB_SIMULATION_MODEL_VERSION",
)


def _is_mongodb_application(config):
    """Return whether an integration application requires MongoDB."""
    return config.get("requires_mongodb") or config["application"].startswith("simtools-db-")


def _has_mongodb_configuration(configuration=None):
    """Return whether required MongoDB settings are available to the application."""
    configuration = configuration or {}
    values = [
        configuration.get(variable.removeprefix("SIMTOOLS_").lower()) or os.environ.get(variable)
        for variable in _MONGODB_ENVIRONMENT
    ]
    model_tag = (
        configuration.get("db_simulation_model_tag")
        or configuration.get("db_simulation_model_version")
        or next((os.environ.get(variable) for variable in _MONGODB_MODEL_TAG_ENVIRONMENT), None)
    )
    return all(values) and bool(model_tag)


def _get_simulation_model_source(config, request, simtools_root_path):
    """Return the configured model source or skip MongoDB-only applications."""
    simulation_models_path = request.config.getoption("simulation_models_path", default=None)
    git_path = request.config.getoption("simulation_models_git_path", default=None)
    git_revision = request.config.getoption("simulation_models_git_revision", default=None)
    if not simulation_models_path and not git_path:
        simulation_models_path = os.environ.get("SIMTOOLS_SIMULATION_MODELS_PATH")
        git_path = os.environ.get("SIMTOOLS_SIMULATION_MODELS_GIT_PATH")
        git_revision = os.environ.get("SIMTOOLS_SIMULATION_MODELS_GIT_REVISION")
    if not simulation_models_path and not git_path:
        return None, None
    if _is_mongodb_application(config):
        pytest.skip(f"{config['application']} requires MongoDB")

    if simulation_models_path:
        simulation_models_path = Path(simulation_models_path)
        if not simulation_models_path.is_absolute():
            simulation_models_path = Path(simtools_root_path) / simulation_models_path
        return simulation_models_path.resolve(), None

    git_path = Path(git_path)
    if not git_path.is_absolute():
        git_path = Path(simtools_root_path) / git_path
    return None, (git_path.resolve(), git_revision)


def _set_simulation_model_source_env(monkeypatch, simulation_models_path, git_source):
    """Set environment variables for the selected simulation-model source."""
    if simulation_models_path:
        monkeypatch.delenv("SIMTOOLS_SIMULATION_MODELS_GIT_PATH", raising=False)
        monkeypatch.delenv("SIMTOOLS_SIMULATION_MODELS_GIT_REVISION", raising=False)
        monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_PATH", str(simulation_models_path))
    if git_source:
        git_path, git_revision = git_source
        monkeypatch.delenv("SIMTOOLS_SIMULATION_MODELS_PATH", raising=False)
        monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_GIT_PATH", str(git_path))
        if git_revision:
            monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_GIT_REVISION", git_revision)


def _get_model_source_arguments(application):
    """Return the model-source argument names accepted by an application."""
    module_name = "simtools.applications." + application.removeprefix("simtools-").replace("-", "_")
    try:
        definition = importlib.import_module(module_name).APPLICATION
    except ImportError, AttributeError:
        return set()
    return {argument.name for argument in definition.all_arguments}


def _set_simulation_model_source_configuration(config, simulation_models_path, git_source):
    """Replace a workflow's configured source with the selected test source.

    The model-source options are only written to workflows of applications using
    the standard model-source arguments. Applications defining
    ``simulation_models_path`` as an application-specific argument are not
    modified; they read the selected source from the environment variables set
    by ``_set_simulation_model_source_env``.
    """
    if not simulation_models_path and not git_source:
        return
    source_config = config.get("configuration")
    if source_config is None:  # e.g. 'auto-no_config' tests running without any argument
        return
    if "simulation_models_git_path" not in _get_model_source_arguments(config["application"]):
        return
    source_config.pop("simulation_models_path", None)
    source_config.pop("simulation_models_git_path", None)
    source_config.pop("simulation_models_git_revision", None)
    if simulation_models_path:
        source_config["simulation_models_path"] = str(simulation_models_path)
        return
    git_path, git_revision = git_source
    source_config["simulation_models_git_path"] = str(git_path)
    if git_revision:
        source_config["simulation_models_git_revision"] = git_revision


def pytest_generate_tests(metafunc):
    """Parametrize application tests using the configured test-resources path."""
    if "config" not in metafunc.fixturenames:
        return

    config_files = sorted(Path(__file__).parent.glob("config/*.yml"))
    test_configs, test_ids = configuration.get_list_of_test_configurations(
        config_files,
        test_resources_path=metafunc.config.getoption("test_resources_path", default=None)
        or os.environ.get("SIMTOOLS_TEST_RESOURCES"),
    )
    test_parameters = []
    for config, test_id in zip(test_configs, test_ids):
        marks = []
        if config.get("test_requirement"):
            marks.append(pytest.mark.verifies_requirement(config["test_requirement"]))
        if config.get("test_use_case"):
            marks.append(pytest.mark.verifies_usecase(config["test_use_case"]))
        if config.get("xfail"):
            marks.append(pytest.mark.xfail(reason=config["xfail"]))
        test_parameters.append(pytest.param(config, id=test_id, marks=marks))

    metafunc.parametrize("config", test_parameters)


def test_applications_from_config(
    tmp_test_directory, config, request, simtools_root_path, monkeypatch
):
    """
    Test all applications from config files found in the config directory.

    Parameters
    ----------
    tmp_test_directory: str
        Temporary directory, into which test configuration and output is written.
    config: dict
        Dictionary with the configuration parameters for the test.

    """
    tmp_config = copy.deepcopy(config)
    skip_message = helpers.skip_camera_efficiency(tmp_config)
    if skip_message:
        pytest.skip(skip_message)

    model_version = request.config.getoption("--model_version", default=None)
    if model_version:
        model_version = model_version.split(",")
        model_version = model_version[0] if len(model_version) == 1 else model_version
    skip_message = helpers.skip_multiple_version_test(tmp_config, model_version)
    if skip_message:
        pytest.skip(skip_message)

    if tmp_config.get("skip_integration_test"):
        pytest.skip(tmp_config["skip_integration_test"])
    if _is_mongodb_application(tmp_config) and not _has_mongodb_configuration(
        tmp_config.get("configuration")
    ):
        pytest.skip(f"{tmp_config['application']} requires MongoDB configuration")

    simulation_models_path, git_source = _get_simulation_model_source(
        tmp_config, request, simtools_root_path
    )
    _set_simulation_model_source_env(monkeypatch, simulation_models_path, git_source)
    _set_simulation_model_source_configuration(tmp_config, simulation_models_path, git_source)

    logger.info(f"Test configuration from config file: {tmp_config}")
    logger.info(f"Model version: {model_version}")
    logger.info(f"Application configuration: {tmp_config}")
    logger.info(f"Test requirement: {config.get('test_requirement')}")
    logger.info(f"Test use case: {config.get('test_use_case')}")
    try:
        cmd, config_file_model_version = configuration.configure(
            tmp_config, tmp_test_directory, request
        )
    except (configuration.ProductionDBError, configuration.VersionError) as exc:
        pytest.skip(str(exc))

    logger.info(f"Running application: {cmd}")
    env = os.environ.copy()
    env["SIMTOOLS_OFFLINE_IERS"] = "1"
    result = subprocess.run(
        cmd,
        shell=True,
        input="y\n",
        capture_output=True,
        text=True,
        env=env,
        cwd=simtools_root_path,
    )
    msg = f"Command {cmd!r} failed. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    if result.returncode != 0 and config.get("xfail_network_error"):
        combined_output = result.stdout + result.stderr
        network_error_patterns = (
            "URLError",
            "Network is unreachable",
            "ConnectionError",
            "TimeoutError",
            "gaierror",
        )
        if any(pattern in combined_output for pattern in network_error_patterns):
            pytest.xfail(f"Network error: {msg}")
    assert result.returncode == 0, msg

    assert log_inspector.inspect([result.stdout, result.stderr])

    validate_output.validate_application_output(
        tmp_config,
        model_version,
        config_file_model_version or model_version,
    )


def test_get_simulation_model_source_from_filesystem(tmp_test_directory, mocker):
    """Resolve a relative simulation-model path against the simtools root."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = "../simulation-models"
    root_path = Path(tmp_test_directory) / "simtools"

    path, git_source = _get_simulation_model_source(
        {"application": "simtools-simulate-prod"}, request, root_path
    )

    assert path == (root_path / "../simulation-models").resolve()
    assert git_source is None


def test_get_simulation_model_source_from_git(tmp_test_directory, mocker):
    """Resolve Git source options and preserve their requested revision."""
    request = mocker.MagicMock()
    options = {
        "simulation_models_path": None,
        "simulation_models_git_path": "../simulation-models.git",
        "simulation_models_git_revision": "HEAD",
    }
    request.config.getoption.side_effect = lambda option, default=None: options.get(option, default)

    path, git_source = _get_simulation_model_source(
        {"application": "simtools-simulate-prod"}, request, tmp_test_directory
    )

    assert path is None
    assert git_source == ((Path(tmp_test_directory) / "../simulation-models.git").resolve(), "HEAD")


def test_get_simulation_model_source_from_git_environment(tmp_test_directory, mocker, monkeypatch):
    """Use the Git source configured in .env when no command-line option is given."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = None
    monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_GIT_PATH", "../simulation-models.git")
    monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_GIT_REVISION", "6.0.2")

    path, git_source = _get_simulation_model_source(
        {"application": "simtools-simulate-prod"}, request, tmp_test_directory
    )

    assert path is None
    assert git_source == (
        (Path(tmp_test_directory) / "../simulation-models.git").resolve(),
        "6.0.2",
    )


def test_git_source_replaces_filesystem_source_in_workflow_configuration(
    tmp_test_directory, mocker
):
    """A selected Git source overrides a filesystem source embedded in a workflow."""
    mocker.patch(
        "tests.integration_tests.test_applications_from_config._get_model_source_arguments",
        return_value={
            "simulation_models_path",
            "simulation_models_git_path",
            "simulation_models_git_revision",
        },
    )
    config = {
        "application": "simtools-simulate-prod",
        "configuration": {
            "simulation_models_path": "../simulation-models",
            "simulation_models_git_revision": "old-revision",
        },
    }

    _set_simulation_model_source_configuration(
        config,
        None,
        ((Path(tmp_test_directory) / "models.git").resolve(), "7.0.0"),
    )

    assert config["configuration"] == {
        "simulation_models_git_path": str((Path(tmp_test_directory) / "models.git").resolve()),
        "simulation_models_git_revision": "7.0.0",
    }


def test_unsupported_application_does_not_receive_model_source_configuration(
    tmp_test_directory, mocker
):
    """Do not write model-source options into workflows that reject those arguments."""
    mocker.patch(
        "tests.integration_tests.test_applications_from_config._get_model_source_arguments",
        return_value=set(),
    )
    config = {"application": "simtools-derive-photon-electron-spectrum", "configuration": {}}

    _set_simulation_model_source_configuration(
        config,
        None,
        ((Path(tmp_test_directory) / "models.git").resolve(), "7.0.0"),
    )

    assert config["configuration"] == {}


def test_filesystem_source_supported_arguments_only(tmp_test_directory, mocker):
    """Inject a filesystem source only for applications using the model-source arguments."""
    mocker.patch(
        "tests.integration_tests.test_applications_from_config._get_model_source_arguments",
        return_value={"simulation_models_path", "simulation_models_git_path"},
    )
    config = {"application": "simtools-simulate-prod", "configuration": {}}

    _set_simulation_model_source_configuration(
        config,
        Path(tmp_test_directory) / "simulation-models",
        None,
    )

    assert config["configuration"] == {
        "simulation_models_path": str(Path(tmp_test_directory) / "simulation-models")
    }


def test_application_specific_model_path_is_not_overridden(tmp_test_directory, mocker):
    """Leave an application-specific ``simulation_models_path`` argument untouched."""
    mocker.patch(
        "tests.integration_tests.test_applications_from_config._get_model_source_arguments",
        return_value={"simulation_models_path"},
    )
    config = {
        "application": "simtools-docs-produce-production-summary",
        "configuration": {"simulation_models_path": "../simulation-models"},
    }

    _set_simulation_model_source_configuration(
        config,
        None,
        ((Path(tmp_test_directory) / "models.git").resolve(), "7.0.0"),
    )

    assert config["configuration"] == {"simulation_models_path": "../simulation-models"}


def test_git_source_replaces_stale_revision(tmp_test_directory, mocker):
    """Replace a stale Git revision when injecting a Git source."""
    mocker.patch(
        "tests.integration_tests.test_applications_from_config._get_model_source_arguments",
        return_value={"simulation_models_git_path", "simulation_models_git_revision"},
    )
    config = {
        "application": "simtools-simulate-prod",
        "configuration": {"simulation_models_git_revision": "old-revision"},
    }

    _set_simulation_model_source_configuration(
        config,
        None,
        ((Path(tmp_test_directory) / "models.git").resolve(), "7.0.0"),
    )

    assert config["configuration"] == {
        "simulation_models_git_path": str((Path(tmp_test_directory) / "models.git").resolve()),
        "simulation_models_git_revision": "7.0.0",
    }


def test_git_source_does_not_skip_application_with_model_path_argument(
    tmp_test_directory, mocker, monkeypatch
):
    """Run workflows using an application-specific model path also for a Git source."""
    request = mocker.MagicMock()
    options = {
        "simulation_models_path": None,
        "simulation_models_git_path": "models.git",
        "simulation_models_git_revision": "HEAD",
    }
    request.config.getoption.side_effect = lambda option, default=None: options.get(option, default)
    config = {
        "application": "simtools-docs-produce-production-summary",
        "test_name": "run",
        "configuration": {"simulation_models_path": "../simulation-models"},
    }
    run = mocker.patch("subprocess.run")
    run.return_value = mocker.MagicMock(returncode=0, stdout="", stderr="")
    validate = mocker.patch("simtools.testing.validate_output.validate_application_output")

    test_applications_from_config(
        tmp_test_directory, config, request, Path(tmp_test_directory), monkeypatch
    )

    assert run.called
    validate.assert_called_once()
    assert config["configuration"]["simulation_models_path"] == "../simulation-models"


def test_mongodb_only_application_is_skipped(tmp_test_directory, mocker):
    """Skip MongoDB-only applications when filesystem model access is selected."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = "../simulation-models"
    config = {"application": "simtools-mongodb-operation", "requires_mongodb": True}

    with pytest.raises(pytest.skip.Exception, match="simtools-mongodb-operation requires MongoDB"):
        _get_simulation_model_source(config, request, tmp_test_directory)


def test_database_application_is_skipped_without_metadata(tmp_test_directory, mocker):
    """Recognize database applications even when old configs lack metadata."""
    request = mocker.MagicMock()
    options = {
        "simulation_models_path": None,
        "simulation_models_git_path": "models.git",
        "simulation_models_git_revision": "HEAD",
    }
    request.config.getoption.side_effect = lambda option, default=None: options.get(option, default)

    with pytest.raises(
        pytest.skip.Exception, match="simtools-db-get-array-layouts-from-db requires MongoDB"
    ):
        _get_simulation_model_source(
            {"application": "simtools-db-get-array-layouts-from-db"},
            request,
            tmp_test_directory,
        )


def test_database_application_is_skipped_without_database_configuration(
    tmp_test_directory, mocker, monkeypatch
):
    """Avoid launching DB applications when the test environment has no DB settings."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = None
    for variable in _MONGODB_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(
        pytest.skip.Exception,
        match="simtools-db-get-file-from-db requires MongoDB configuration",
    ):
        test_applications_from_config(
            tmp_test_directory,
            {"application": "simtools-db-get-file-from-db"},
            request,
            tmp_test_directory,
            monkeypatch,
        )


def test_get_simulation_model_source_is_optional(tmp_test_directory, mocker, monkeypatch):
    """Leave integration tests unchanged when no filesystem path is configured."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = None
    monkeypatch.delenv("SIMTOOLS_SIMULATION_MODELS_PATH", raising=False)
    monkeypatch.delenv("SIMTOOLS_SIMULATION_MODELS_GIT_PATH", raising=False)

    assert _get_simulation_model_source(
        {"application": "simtools-simulate-prod"}, request, tmp_test_directory
    ) == (None, None)
