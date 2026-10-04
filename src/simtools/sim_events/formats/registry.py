"""Select simulation-file readers independently of simulation software."""


def _eventio_reader(file_name):
    """Construct the reader for CORSIKA IACT and sim_telarray eventio files."""
    # Keep native parsing imports confined to the selected format.
    # pylint: disable-next=import-outside-toplevel
    from simtools.sim_events.formats.eventio_reader import EventioReader

    return EventioReader(file_name)


_READERS = {"eventio": _eventio_reader}


def register_reader(file_format, factory):
    """Register a simulation-file reader factory.

    Parameters
    ----------
    file_format : str
        Unique format name, independent of file suffix and simulation software.
    factory : callable
        Callable accepting a file path and returning a reader. Readers supply
        ``iter_records(file_id)``, ``read_metadata()``, ``read_run_number()``,
        and ``count_events()``. See the simulation-file format documentation
        for record fields and units.

    Raises
    ------
    ValueError
        If the name is empty, already registered, or the factory is not callable.
    """
    if not file_format or not callable(factory):
        raise ValueError("A format name and callable reader factory are required.")
    if file_format in _READERS:
        raise ValueError(f"Simulation-file format '{file_format}' is already registered.")
    _READERS[file_format] = factory


def available_formats():
    """Return the registered simulation-file format names."""
    return tuple(sorted(_READERS))


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
    object
        Reader supplied by the selected factory.

    Raises
    ------
    ValueError
        If the requested format is not registered.
    """
    try:
        factory = _READERS[file_format]
    except KeyError as exc:
        raise ValueError(
            f"Unknown simulation-file format '{file_format}'. "
            f"Available formats: {', '.join(available_formats())}."
        ) from exc
    return factory(file_name)
