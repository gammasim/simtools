"""Construct the configured simulation-model reader."""

from simtools.model_repository import dependency_versions
from simtools.model_repository import reader as model_repository_reader


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
        Git tag, ref, or commit. The model-repository catalog ref is used when
        the Git path is set and this value is omitted.

    Returns
    -------
    SimulationModelReader
        Reader backed by the selected source.
    """
    if simulation_models_path is None and simulation_models_git_path is None:
        reader = model_repository_reader.resolve_model_reader()
        if reader is None:
            raise ValueError("A filesystem path or Git simulation-model source is required.")
        return reader

    if simulation_models_path and simulation_models_git_path:
        raise ValueError(
            "Filesystem and Git simulation-model sources cannot be configured together."
        )

    if simulation_models_path:
        return model_repository_reader.SimulationModelReader.from_files(simulation_models_path)

    if simulation_models_git_path:
        revision = simulation_models_git_revision
        if revision is None:
            revision = dependency_versions.model_repository_revision(
                dependency_versions.load_dependency_catalog()
            )
        if not revision:
            raise ValueError("A Git simulation-model revision is required.")
        return model_repository_reader.SimulationModelReader.from_git(
            simulation_models_git_path, revision
        )

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
    return model_repository_reader.create_model_reader_from_source_config(source_config)
