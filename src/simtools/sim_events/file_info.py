#!/usr/bin/python3
"""Read file info and run headers from eventio (CORSIKA IACT, sim_telarray) files."""

import warnings

from eventio import EventIOFile, iact
from eventio.simtel import MCRunHeader, MCShower, RunHeader

# Suppress all UserWarnings from corsikaio - no CORSIKA versions <7.7 are supported anyway
warnings.filterwarnings("ignore", category=UserWarning, module=r"corsikaio\.subblocks\..*")


def get_corsika_run_number(file):
    """
    Return the CORSIKA run number from an eventio (CORSIKA IACT or sim_telarray) file.

    Parameters
    ----------
    file: str
        Path to the eventio file.

    Returns
    -------
    int, None
        CORSIKA run number. Returns None if not found.
    """
    run_header = get_combined_eventio_run_header(file)
    if run_header and "run" in run_header:
        return run_header["run"]
    run_header, _ = get_corsika_run_and_event_headers(file)
    try:
        return int(run_header["run_number"])
    except (TypeError, KeyError, ValueError):
        return None


def get_combined_eventio_run_header(sim_telarray_file):
    """
    Return the CORSIKA run header information from an eventio (sim_telarray) file.

    Reads both RunHeader and MCRunHeader object from file and returns a merged dictionary.
    Adds primary id from the first event.

    Parameters
    ----------
    sim_telarray_file: str
        Path to the sim_telarray file.

    Returns
    -------
    dict, None
        CORSIKA run header. Returns None if not found.
    """
    run_header = mc_run_header = None
    primary_id = None

    with EventIOFile(sim_telarray_file) as f:
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


def get_corsika_run_and_event_headers(corsika_iact_file):
    """
    Return the CORSIKA run and event headers from a CORSIKA IACT eventio file.

    Parameters
    ----------
    corsika_iact_file: str, Path
        Path to the CORSIKA IACT eventio file.

    Returns
    -------
    tuple
        CORSIKA run header and event header as dictionaries.
    """
    run_header = event_header = None

    with EventIOFile(corsika_iact_file) as f:
        for o in f:
            if isinstance(o, iact.RunHeader) and run_header is None:
                run_header = o.parse()
            elif isinstance(o, iact.EventHeader) and event_header is None:
                event_header = o.parse()
            if run_header and event_header:
                break

    return run_header, event_header


def get_simulated_events(event_io_file):
    """
    Return the number of shower and MC events from a simulation (eventio) file.

    For a sim_telarray file, the number of simulated showers and MC events is
    determined by counting the number of MCShower (type id 2020) and MCEvent
    objects (type id 2021). For a CORSIKA IACT file, the number of simulated
    showers is determined by counting the number of IACTShower (type id 1202).

    Parameters
    ----------
    event_io_file: str, Path
        Path to the eventio file.

    Returns
    -------
    tuple
        Number of showers and number of MC events (MC events for sim_telarray files only).
    """
    counts = {1202: 0, 2020: 0, 2021: 0}
    with EventIOFile(event_io_file) as f:
        for o in f:
            t = o.header.type
            if t in counts:
                counts[t] += 1
    return counts[2020] if counts[2020] else counts[1202], counts[2021]
