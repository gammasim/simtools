"""Construct the configured simulation-model reader."""

import os

from simtools import dependency_versions
from simtools.model_repository.reader import SimulationModelReader
from simtools.settings import config


def create_model_reader(
    simulation_models_path=None,
    simulation_models_git_path=None,
    simulation_models_git_revision=None,
):
    """Create a reader for a filesystem or Git model source.

    Parameters
    ----------
    simulation_models_path : str or Path, optional
        Root of a checked-out simulation-model repository. When supplied, this
        source is selected. If omitted, ``SIMTOOLS_SIMULATION_MODELS_PATH`` is
        used when set.
    simulation_models_git_path : str or Path, optional
        Local normal, bare, or mirror Git repository.
    simulation_models_git_revision : str, optional
        Git tag, ref, or commit. The model-repository catalog revision is used when
        the Git path is set and this value is omitted.

    Returns
    -------
    SimulationModelReader
        Reader backed by the selected source.
    """
    if simulation_models_path is None and simulation_models_git_path is None:
        simulation_models_path = os.getenv("SIMTOOLS_SIMULATION_MODELS_PATH")
        simulation_models_git_path = os.getenv("SIMTOOLS_SIMULATION_MODELS_GIT_PATH")
        simulation_models_git_revision = os.getenv("SIMTOOLS_SIMULATION_MODELS_GIT_REVISION")

    if simulation_models_path and simulation_models_git_path:
        raise ValueError(
            "Filesystem and Git simulation-model sources cannot be configured together."
        )

    if simulation_models_path:
        return SimulationModelReader.from_files(simulation_models_path)

    if simulation_models_git_path:
        revision = simulation_models_git_revision
        if revision is None:
            catalog = dependency_versions.load_dependency_catalog()
            model = catalog["model-repository"]
            revision = model.get("git-revision") or model.get(
                "default-tag", model.get("default-version")
            )
        if not revision:
            raise ValueError("A Git simulation-model revision is required.")
        return SimulationModelReader.from_git(simulation_models_git_path, revision)

    raise ValueError("A filesystem path or Git simulation-model source is required.")


def create_model_reader_from_configuration(configuration):
    """Create a model reader from an application or test configuration.

    Parameters
    ----------
    configuration : dict
        Mapping that may contain filesystem or Git model-source options.

    Returns
    -------
    SimulationModelReader
        Reader selected by the supplied source options.
    """
    return create_model_reader(
        simulation_models_path=configuration.get("simulation_models_path"),
        simulation_models_git_path=configuration.get("simulation_models_git_path"),
        simulation_models_git_revision=configuration.get("simulation_models_git_revision"),
    )


def create_model_reader_from_source_config(source_config):
    """Recreate a model reader from a worker-serializable source configuration.

    Parameters
    ----------
    source_config : dict
        Configuration returned by ``SimulationModelReader.source_config``.

    Returns
    -------
    SimulationModelReader
        Reader backed by the configured source.

    Raises
    ------
    ValueError
        If the source configuration is missing or unsupported.
    """
    if not isinstance(source_config, dict):
        raise ValueError("Model source configuration must be a dictionary.")
    source_type = source_config.get("type")
    if source_type == "filesystem":
        path = source_config.get("path")
        if not path:
            raise ValueError("Filesystem model source configuration requires a path.")
        return create_model_reader(simulation_models_path=path)
    if source_type == "git":
        repository = source_config.get("repository")
        commit = source_config.get("commit")
        if not repository or not commit:
            raise ValueError("Git model source configuration requires repository and commit.")
        return create_model_reader(
            simulation_models_git_path=repository,
            simulation_models_git_revision=commit,
        )
    raise ValueError(f"Unsupported model source type: {source_type!r}.")


def require_model_reader(model_reader=None):
    """Return an explicit or application-configured model reader.

    Parameters
    ----------
    model_reader : SimulationModelReader, optional
        Reader explicitly supplied by a caller outside an application context.

    Returns
    -------
    SimulationModelReader
        The supplied reader or the reader selected during application startup.

    Raises
    ------
    RuntimeError
        If no simulation-model reader has been selected.
    """
    if model_reader is not None:
        return model_reader
    if config.model_reader is None:
        raise RuntimeError("No simulation model reader is configured.")
    return config.model_reader
