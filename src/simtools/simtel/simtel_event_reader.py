#!/usr/bin/python3
"""Event reader for sim_telarray."""

import logging

from eventio import SimTelFile

from simtools.simtel.simtel_io_metadata import get_sim_telarray_telescope_id

_logger = logging.getLogger(__name__)


def read_events(file_name, telescope, event_id, max_events=1):
    """
    Read events from a sim_telarray file for a given telescope.

    Parameters
    ----------
    file_name : str or Path
        Path to the sim_telarray file.
    telescope : str
        Telescope type to filter events.
    event_id : int
        ID of the event to read.
    max_events : int
        Maximum number of events to read (starting from event_id).

    Returns
    -------
    tuple
        A 3-tuple containing:
        - event_ids (list of int): List of event indices that were read.
        - tel_desc (dict): Telescope description dictionary.
        - events (list): List of telescope events.
        Returns (None, None, None) if telescope not found or no events available.
    """
    tel_id = get_sim_telarray_telescope_id(telescope, file_name)
    tel_id = 1 if tel_id is None else tel_id  # TODO
    if tel_id is None:
        _logger.warning(f"A Telescope type '{telescope}' not found in file '{file_name}'.")
        return None, None, None

    event_id = event_id or 0

    events = []
    event_ids = []
    with SimTelFile(file_name, skip_calibration=False) as f:
        try:
            tel_desc = f.telescope_descriptions[tel_id]
        except KeyError:
            _logger.warning(f"Telescope ID '{tel_id}' not found in file '{file_name}'.")
            return None, None, None

        for i, event in enumerate(f):
            if i >= event_id:
                events.append(event["telescope_events"][tel_id])
                event_ids.append(i)
            if len(events) >= max_events:
                break

    return event_ids, tel_desc, events
