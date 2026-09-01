"""Construct the configured simulation-model reader."""

from simtools.model_repository.reader import SimulationModelReader


def create_model_reader(simulation_models_path=None, database_handler=None):
    """Create a reader for a filesystem or MongoDB model source.

    Parameters
    ----------
    simulation_models_path : str or Path, optional
        Root of a checked-out simulation-model repository. When supplied, this
        source is selected and no database handler is constructed.
    database_handler : DatabaseHandler, optional
        Existing MongoDB handler to adapt when no filesystem path is supplied.

    Returns
    -------
    SimulationModelReader
        Reader backed by the selected source.
    """
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
