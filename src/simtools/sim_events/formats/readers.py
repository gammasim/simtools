"""Select simulation-file readers independently of simulation software."""

from typing import Protocol


class EventReader(Protocol):
    """Read run information, metadata, parameters, and reduced event records."""

    def read_run_number(self):
        """Return the run number, or None when absent."""

    def count_events(self):
        """Return shower and event counts."""

    def iter_records(self, file_id=0):
        """Yield reduced event table names and row dictionaries."""

    def read_metadata(self):
        """Return metadata collected while iterating reduced records."""

    def read_simulation_parameters(self):
        """Return physical shower settings."""


def _eventio_reader(file_name):
    """Construct the reader for CORSIKA IACT and sim_telarray eventio files."""
    # Keep native parsing imports confined to the selected format.
    # pylint: disable-next=import-outside-toplevel
    from simtools.sim_events.formats.eventio_reader import EventioReader

    return EventioReader(file_name)


READERS = {"eventio": _eventio_reader}


def available_formats():
    """Return the registered simulation-file format names."""
    return tuple(sorted(READERS))


def get_reader(file_name, file_format="eventio"):
    """Create a reader for a simulation file.

    Parameters
    ----------
    file_name : str or pathlib.Path
        Simulation input file.
    file_format : str
        Registered format name. The default supports current IACT and simtel files.

    Returns
    -------
    EventReader
        Reader supplied by the selected factory.

    Raises
    ------
    ValueError
        If the requested format is not registered.
    """
    try:
        factory = READERS[file_format]
    except KeyError as exc:
        raise ValueError(
            f"Unknown simulation-file format '{file_format}'. "
            f"Available formats: {', '.join(available_formats())}."
        ) from exc
    return factory(file_name)
