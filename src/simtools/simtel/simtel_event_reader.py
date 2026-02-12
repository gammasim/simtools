#!/usr/bin/python3
"""Event reader for sim_telarray."""

import logging

from eventio import SimTelFile

from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)
from simtools.utils import general as gen

_logger = logging.getLogger(__name__)


def read_events(file_name, telescope, event_ids, max_events=1, verbose=False):
    """
    Read events from a sim_telarray file for a given telescope.

    Parameters
    ----------
    file_name : str or Path
        Path to the sim_telarray file.
    telescope : str
        Telescope type to filter events.
    event_ids : int or list of int
        ID(s) of the event(s) to read.
    max_events : int
        Maximum number of events to read (starting from event_id).
    verbose : bool
        If True, print detailed information about the reading process.

    Returns
    -------
    tuple
        A 3-tuple containing:
        - ids_with_data (list of int): List of event indices that were read.
        - tel_desc (dict): Telescope description dictionary.
        - events (list): List of telescope events.
        Returns (None, None, None) if telescope not found or no events available.
    """
    tel_id_map = get_sim_telarray_telescope_id_to_telescope_name_mapping(file_name)
    tel_id = next((k for k, v in tel_id_map.items() if v == telescope), None)
    if tel_id is None:
        _logger.warning(f"Telescope type '{telescope}' not found in file '{file_name}'.")
        return None, None, None

    event_ids = gen.ensure_iterable(event_ids) if event_ids is not None else []
    ids_with_data, events = [], []

    with SimTelFile(file_name, skip_calibration=False) as f:
        tel_desc = f.telescope_descriptions.get(tel_id)
        if tel_desc is None:
            _logger.warning(f"Telescope ID '{tel_id}' not found in file '{file_name}'.")
            return None, None, None

        for event in f:
            if max_events and len(events) >= max_events:
                break
            if event_ids and event["event_id"] not in event_ids:
                continue
            if tel_id in event["telescope_events"]:
                events.append(event["telescope_events"][tel_id])
                ids_with_data.append(event["event_id"])
            elif verbose:
                triggered = event["trigger_information"]["triggered_telescopes"]
                triggered_names = [tel_id_map.get(tid, f"ID {tid}") for tid in triggered]
                _logger.debug(
                    f"event {event['event_id']} with {len(event['telescope_events'])} "
                    f"telescope events (triggered telescopes: {triggered_names})"
                )

    _logger.info(f"Read {len(events)} events for telescope '{telescope}' from file '{file_name}'.")

    return ids_with_data, tel_desc, events
