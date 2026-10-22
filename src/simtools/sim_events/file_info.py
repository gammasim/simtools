"""Read run information and event counts through simulation-file readers."""

from simtools.sim_events.formats.registry import get_reader


def get_run_number(file, file_format="eventio"):
    """Read the run number using the selected simulation-file format.

    Parameters
    ----------
    file : str or pathlib.Path
        Simulation file.
    file_format : str
        Registered reader name.

    Returns
    -------
    int or None
        Run number, or None when absent.
    """
    return get_reader(file, file_format).read_run_number()


def get_corsika_run_number(file, file_format="eventio"):
    """
    Return the CORSIKA run number from an eventio (CORSIKA IACT or sim_telarray) file.

    Parameters
    ----------
    file: str
        Path to the eventio file.
    file_format : str
        Registered simulation-file reader, by default "eventio".

    Returns
    -------
    int, None
        CORSIKA run number. Returns None if not found.
    """
    return get_run_number(file, file_format)


def get_combined_eventio_run_header(sim_telarray_file, file_format="eventio"):
    """
    Return the CORSIKA run header information from an eventio (sim_telarray) file.

    Reads both RunHeader and MCRunHeader object from file and returns a merged dictionary.
    Adds primary id from the first event.

    Parameters
    ----------
    sim_telarray_file: str
        Path to the sim_telarray file.
    file_format : str
        Registered simulation-file reader, by default "eventio".

    Returns
    -------
    dict, None
        CORSIKA run header. Returns None if not found.
    """
    return get_reader(sim_telarray_file, file_format).read_combined_run_header()


def get_corsika_run_and_event_headers(corsika_iact_file, file_format="eventio"):
    """
    Return the CORSIKA run and event headers from a CORSIKA IACT eventio file.

    Parameters
    ----------
    corsika_iact_file: str, Path
        Path to the CORSIKA IACT eventio file.
    file_format : str
        Registered simulation-file reader, by default "eventio".

    Returns
    -------
    tuple
        CORSIKA run header and event header as dictionaries.
    """
    return get_reader(corsika_iact_file, file_format).read_run_headers()


def get_simulated_events(event_io_file, file_format="eventio"):
    """
    Return the number of shower and MC events from a simulation (eventio) file.

    Counts are supplied by the selected reader. Shower reuse is counted
    separately where the file format provides reused events.

    Parameters
    ----------
    event_io_file: str, Path
        Path to the eventio file.
    file_format : str
        Registered simulation-file reader, by default "eventio".

    Returns
    -------
    tuple
        Number of showers and number of MC events (MC events for sim_telarray files only).
    """
    return get_reader(event_io_file, file_format).count_events()
