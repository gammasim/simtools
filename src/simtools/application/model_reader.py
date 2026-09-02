"""Construct the configured simulation-model reader."""

import os

from simtools.model_repository.reader import SimulationModelReader
from simtools.settings import config


def create_model_reader(simulation_models_path=None, database_handler=None):
    """Create a reader for a filesystem or MongoDB model source.

    Parameters
    ----------
    simulation_models_path : str or Path, optional
        Root of a checked-out simulation-model repository. When supplied, this
        source is selected and no database handler is constructed. If omitted,
        ``SIMTOOLS_SIMULATION_MODELS_PATH`` is used when set.
    database_handler : DatabaseHandler, optional
        Existing MongoDB handler to adapt when no filesystem path is supplied.

    Returns
    -------
    SimulationModelReader
        Reader backed by the selected source.
    """
    if simulation_models_path is None and database_handler is None:
        simulation_models_path = os.getenv("SIMTOOLS_SIMULATION_MODELS_PATH")

    if simulation_models_path:
        return SimulationModelReader.from_files(simulation_models_path)

    if database_handler is None:
        from simtools.db.db_handler import (  # pylint: disable=import-outside-toplevel
            DatabaseHandler,
        )

        database_handler = DatabaseHandler()
    from simtools.db.model_source import (  # pylint: disable=import-outside-toplevel
        MongoDBModelSource,
    )

    return SimulationModelReader(MongoDBModelSource(database_handler))


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
