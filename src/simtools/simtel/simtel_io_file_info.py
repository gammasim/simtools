#!/usr/bin/python3
"""Read file info and run headers from sim_telarray files."""

from eventio import EventIOFile
from eventio.simtel import RunHeader


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
    run_header = None
    with EventIOFile(file) as f:
        found_run_header = False
        for o in f:
            if isinstance(o, RunHeader):
                found_run_header = True
            else:
                if found_run_header:
                    break
                continue
            run_header = o.parse()

    return run_header.get("run") if run_header else None
