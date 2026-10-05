#!/usr/bin/python3
# Integration tests for applications from config file

import copy
import importlib
import logging
import os
import subprocess
from pathlib import Path

import pytest

from simtools.testing import configuration, helpers, log_inspector, options, validate_output

logger = logging.getLogger()


def _get_simulation_model_source(config, request, simtools_root_path):
    """Return the configured simulation-model repository source."""
    simulation_models_path = options.get_mirrored_option(request.config, "simulation_models_path")
    git_path = options.get_mirrored_option(request.config, "simulation_models_git_path")
    git_revision = options.get_mirrored_option(request.config, "simulation_models_git_revision")
    if not simulation_models_path and not git_path:
        return None, None
    if simulation_models_path:
        simulation_models_path = Path(simulation_models_path)
        if not simulation_models_path.is_absolute():
            simulation_models_path = Path(simtools_root_path) / simulation_models_path
        return simulation_models_path.resolve(), None

    git_path = Path(git_path)
    if not git_path.is_absolute():
        git_path = Path(simtools_root_path) / git_path
    # An explicitly selected local repository represents the checkout used by
    # the integration run. Use its tip unless a revision was requested.
    return None, (git_path.resolve(), git_revision or "HEAD")


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
    return {argument.name for argument in _get_application_arguments(application)}


def _get_application_arguments(application):
    """Return the command-line arguments accepted by an application."""
    module_name = "simtools.applications." + application.removeprefix("simtools-").replace("-", "_")
    try:
        definition = importlib.import_module(module_name).APPLICATION
    except ImportError, AttributeError:
        return ()
    return definition.all_arguments


def _requires_simulation_model_source(config, simulation_models_path, git_source):
    """Return whether a workflow needs an unavailable simulation-model source."""
    source_config = config.get("configuration")
    if source_config is None:
        return False
    if simulation_models_path or source_config.get("simulation_models_path"):
        return False
    arguments = _get_application_arguments(config["application"])
    requires_local_source = any(
        argument.name == "simulation_models_path" and argument.kwargs.get("required")
        for argument in arguments
    )
    if git_source or source_config.get("simulation_models_git_path"):
        return requires_local_source
    return requires_local_source or any(
        argument.name == "simulation_models_git_path" for argument in arguments
    )


def _set_simulation_model_source_configuration(config, simulation_models_path, git_source):
    """Replace a workflow's configured source with the selected test source.

    A source is only written when the application accepts its corresponding
    command-line option. This also supports applications that accept a local
    repository path directly without using the standard Git-source arguments.
    """
    if not simulation_models_path and not git_source:
        return
    source_config = config.get("configuration")
    if source_config is None:  # e.g. 'auto-no_config' tests running without any argument
        return
    model_source_arguments = _get_model_source_arguments(config["application"])
    if simulation_models_path and "simulation_models_path" not in model_source_arguments:
        return
    if git_source and "simulation_models_git_path" not in model_source_arguments:
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


def _validate_preparation_output_file(output_file):
    """Reject preparation output files that escape the prepared-resource directory."""
    if output_file is None:
        return
    if not isinstance(output_file, str):
        raise ValueError("Preparation output_file must be a relative path string.")
    output_path = Path(output_file)
    if output_path.is_absolute() or ".." in output_path.parts:
        raise ValueError(
            "Preparation output_file must stay within the prepared-resource directory: "
            f"{output_file!r}."
        )


def _prepare_model_parameter_inputs(
    config, tmp_test_directory, request, simtools_root_path, simulation_models_path, git_source
):
    """Run configured model-parameter retrieval steps for an integration test."""
    preparation_steps = config.pop("preparation", [])
    if not preparation_steps:
        return

    prepared_resources_path = Path(tmp_test_directory) / "prepared-resources"
    prepared_resources_path.mkdir(parents=True, exist_ok=True)
    for index, preparation in enumerate(preparation_steps):
        preparation_config = copy.deepcopy(preparation)
        preparation_config["test_name"] = f"preparation-{index}"
        preparation_options = preparation_config["configuration"]
        _validate_preparation_output_file(preparation_options.get("output_file"))
        preparation_options["output_path"] = str(prepared_resources_path)
        _set_simulation_model_source_configuration(
            preparation_config, simulation_models_path, git_source
        )
        command, _ = configuration.configure(preparation_config, tmp_test_directory, request)
        result = subprocess.run(
            command,
            shell=True,
            input="y\n",
            capture_output=True,
            text=True,
            env={**os.environ, "SIMTOOLS_OFFLINE_IERS": "1"},
            cwd=simtools_root_path,
        )
        message = (
            f"Preparation command {command!r} failed. stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert result.returncode == 0, message
        assert log_inspector.inspect([result.stdout, result.stderr])

    config.update(configuration.resolve_prepared_resource_paths(config, prepared_resources_path))


def pytest_generate_tests(metafunc):
    """Parametrize application tests using the configured test-resources path."""
    if "config" not in metafunc.fixturenames:
        return

    config_files = sorted(Path(__file__).parent.glob("config/*.yml"))
    test_configs, test_ids = configuration.get_list_of_test_configurations(
        config_files,
        test_resources_path=metafunc.config.getoption("test_resources_path", default=None),
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
    model_version = request.config.getoption("--model_version", default=None)
    if model_version:
        model_version = model_version.split(",")
        model_version = model_version[0] if len(model_version) == 1 else model_version
    skip_message = helpers.skip_multiple_version_test(tmp_config, model_version)
    if skip_message:
        pytest.skip(skip_message)

    if tmp_config.get("skip_integration_test"):
        pytest.skip(tmp_config["skip_integration_test"])
    simulation_models_path, git_source = _get_simulation_model_source(
        tmp_config, request, simtools_root_path
    )
    if _requires_simulation_model_source(tmp_config, simulation_models_path, git_source):
        pytest.skip("No simulation-model repository is configured")
    _set_simulation_model_source_env(monkeypatch, simulation_models_path, git_source)
    _set_simulation_model_source_configuration(tmp_config, simulation_models_path, git_source)
    _prepare_model_parameter_inputs(
        tmp_config,
        tmp_test_directory,
        request,
        simtools_root_path,
        simulation_models_path,
        git_source,
    )

    logger.info(f"Test configuration from config file: {tmp_config}")
    logger.info(f"Model version: {model_version}")
    logger.info(f"Application configuration: {tmp_config}")
    logger.info(f"Test requirement: {config.get('test_requirement')}")
    logger.info(f"Test use case: {config.get('test_use_case')}")
    try:
        cmd, config_file_model_version = configuration.configure(
            tmp_config, tmp_test_directory, request
        )
    except configuration.VersionError as exc:
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
    monkeypatch.delenv("SIMTOOLS_SIMULATION_MODELS_PATH", raising=False)
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


def test_git_model_source_defaults_to_checkout_head(tmp_test_directory, mocker):
    """Use the selected checkout tip when no Git revision is configured."""
    request = mocker.MagicMock()
    options = {
        "simulation_models_path": None,
        "simulation_models_git_path": "../simulation-models.git",
        "simulation_models_git_revision": None,
    }
    request.config.getoption.side_effect = lambda option, default=None: options.get(option, default)

    path, git_source = _get_simulation_model_source(
        {"application": "simtools-simulate-prod"}, request, tmp_test_directory
    )

    assert path is None
    assert git_source == (
        (Path(tmp_test_directory) / "../simulation-models.git").resolve(),
        "HEAD",
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


def test_set_simulation_model_source_configuration_uses_local_path_argument():
    """Configure applications that accept only a local model repository path."""
    config = {
        "application": "simtools-docs-produce-production-summary",
        "configuration": {"output_file": "production_version_descriptions.md"},
    }
    model_path = Path("/models")
    _set_simulation_model_source_configuration(config, model_path, None)

    assert config["configuration"]["simulation_models_path"] == str(model_path)


def test_set_simulation_model_source_configuration_uses_git_source():
    """Configure applications that use the common model reader with a Git source."""
    config = {
        "application": "simtools-docs-produce-production-summary",
        "configuration": {"output_file": "production_version_descriptions.md"},
    }
    _set_simulation_model_source_configuration(config, None, (Path("/models.git"), "HEAD"))

    assert config["configuration"] == {
        "output_file": "production_version_descriptions.md",
        "simulation_models_git_path": "/models.git",
        "simulation_models_git_revision": "HEAD",
    }


def test_requires_simulation_model_source_for_path_only_application():
    """Recognize workflows that still require a filesystem model repository."""
    config = {
        "application": "simtools-maintain-simulation-model-verify-production-tables",
        "configuration": {},
    }

    assert _requires_simulation_model_source(config, None, None)


def test_requires_simulation_model_source_for_reader_application():
    """Skip reader-backed applications when no model source is selected."""
    config = {
        "application": "simtools-docs-produce-production-summary",
        "configuration": {"output_file": "production_version_descriptions.md"},
    }

    assert _requires_simulation_model_source(config, None, None)


def test_git_source_is_incompatible_with_required_filesystem_argument():
    """Skip path-only applications when integration selects a Git source."""
    config = {
        "application": "simtools-maintain-simulation-model-verify-production-tables",
        "configuration": {},
    }

    assert _requires_simulation_model_source(config, None, (Path("/models.git"), "HEAD"))


def test_local_simulation_model_source_is_not_required_when_configured():
    """Allow workflows that supply their required local repository explicitly."""
    config = {
        "application": "simtools-docs-produce-production-summary",
        "configuration": {
            "output_file": "production_version_descriptions.md",
            "simulation_models_path": "models",
        },
    }

    assert not _requires_simulation_model_source(config, None, None)


def test_local_simulation_model_source_is_not_required_without_configuration():
    """Allow automatic no-configuration checks to run application help."""
    config = {"application": "simtools-docs-produce-production-summary"}

    assert not _requires_simulation_model_source(config, None, None)


def test_prepare_model_parameter_inputs(tmp_test_directory, mocker):
    """Prepare model-parameter files and resolve their temporary references."""
    config = {
        "preparation": [
            {
                "application": "simtools-get-model-parameter",
                "configuration": {"parameter": "array_layouts", "site": "North"},
            }
        ],
        "configuration": {"array_layout_parameter_file": "${prepared:array_layouts.json}"},
        "integration_tests": [
            {
                "test_outputs": [
                    {
                        "validations": [
                            {"reference": "${prepared:array_layouts.json}", "type": "reference"}
                        ]
                    }
                ]
            }
        ],
    }
    request = mocker.MagicMock()
    mocker.patch.object(configuration, "configure", return_value=("prepare-command", None))
    result = mocker.MagicMock(returncode=0, stdout="", stderr="")
    mocker.patch("subprocess.run", return_value=result)
    mocker.patch.object(log_inspector, "inspect", return_value=True)

    _prepare_model_parameter_inputs(
        config,
        tmp_test_directory,
        request,
        tmp_test_directory,
        None,
        None,
    )

    prepared_file = tmp_test_directory / "prepared-resources/array_layouts.json"
    assert "preparation" not in config
    assert config["configuration"]["array_layout_parameter_file"] == str(prepared_file)
    validation = config["integration_tests"][0]["test_outputs"][0]["validations"][0]
    assert validation["reference"] == str(prepared_file)


@pytest.mark.parametrize("output_file", ["../outside.json", "absolute"])
def test_prepare_model_parameter_inputs_rejects_escaping_output_file(
    tmp_test_directory, output_file
):
    """Reject preparation output files outside the temporary resource directory."""
    if output_file == "absolute":
        output_file = str((Path(tmp_test_directory) / "outside.json").resolve())
    with pytest.raises(ValueError, match="must stay within"):
        _validate_preparation_output_file(output_file)
