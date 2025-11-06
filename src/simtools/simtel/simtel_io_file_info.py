#!/usr/bin/python3
"""Read file info and run headers from sim_telarray files."""

from eventio import EventIOFile
from eventio.simtel import MCRunHeader, MCShower, RunHeader


def get_corsika_run_number(file):
    """
    Return the CORSIKA run number from a sim_telarray file.

    Parameters
    ----------
    file: str
        Path to the sim_telarray file.

    Returns
    -------
    int, None
        CORSIKA run number. Returns None if not found.
    """
    run_header = get_corsika_run_header(file)
    return run_header.get("run") if run_header else None


def get_corsika_run_header(file):
    """
    Return the CORSIKA run header information from a sim_telarray file.

    Reads both RunHeader and MCRunHeader object from file and
    returns a merged dictionary. Adds primary id from the first event.

    Parameters
    ----------
    file: str
        Path to the sim_telarray file.

    Returns
    -------
    dict, None
        CORSIKA run header. Returns None if not found.
    """
    run_header = None
    mc_run_header = None
    primary_id = None

    with EventIOFile(file) as f:
        for o in f:
            if isinstance(o, RunHeader) and run_header is None:
                run_header = o.parse()
            elif isinstance(o, MCRunHeader) and mc_run_header is None:
                mc_run_header = o.parse()
            elif isinstance(o, MCShower):  # get primary_id from first MCShower
                primary_id = o.parse().get("primary_id")
            if run_header and mc_run_header and primary_id is not None:
                break

    run_header = run_header or {}
    mc_run_header = mc_run_header or {}
    if primary_id is not None:
        mc_run_header["primary_id"] = primary_id
    return run_header | mc_run_header or None


def get_simulated_events(file):
    """
    Return the number of shower and MC events from a simulation file.

    For a sim_telarray file, the number of simulated showers and MC events is
    determined by counting the number of MCShower (type id 2020) and MCEvent
    objects (type id 2021). For a CORSIKA IACT file, the number of simulated
    showers is determined by counting the number of IACTShower (type id 1202).

    Parameters
    ----------
    file: str
        Path to the sim_telarray file.

    Returns
    -------
    tuple
        Number of showers and number of MC events (sim_telarray files only).
        Number of MC events for CORSIKA IACT files.
    """
    counts = {1202: 0, 2020: 0, 2021: 0}
    with EventIOFile(file) as f:
        for o in f:
            t = o.header.type
            if t in counts:
                counts[t] += 1
    return counts[2020], counts[2021], counts[1202]
