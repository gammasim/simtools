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
