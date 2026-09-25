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
    event_ids : int or list of int, optional
        Specific event ID or list of event IDs to include. If None, all events
        are considered (subject to ``max_events``).
    max_events : int, optional
        Maximum total number of events to read. If ``event_ids`` is provided,
        only those IDs are included, up to this limit.
    verbose : bool, optional
        If True, log detailed information about the reading process.

    Returns
    -------
    tuple
        A 3-tuple containing:
        - ids_with_data (list of int): List of event IDs that were read.
        - tel_desc (dict): Telescope description dictionary.
        - events (list): List of telescope events.
        Returns (None, None, None) if telescope not found or no events available.
    """
    tel_id_map = get_sim_telarray_telescope_id_to_telescope_name_mapping(file_name)
    tel_id = next((k for k, v in tel_id_map.items() if v == telescope), None)
    if tel_id is None:
        _logger.warning(f"Telescope type '{telescope}' not found in file '{file_name}'.")
        return None, None, None

    with SimTelFile(file_name, skip_calibration=False) as f:
        tel_desc = f.telescope_descriptions.get(tel_id)
        if tel_desc is None:
            _logger.warning(f"Telescope ID '{tel_id}' not found in file '{file_name}'.")
            return None, None, None
        ids_with_data, events = _collect_events_for_telescope(
            f, tel_id, tel_id_map, gen.ensure_list(event_ids), max_events, verbose
        )

    _logger.info(f"Read {len(events)} events for telescope '{telescope}' from file '{file_name}'.")

    return ids_with_data, tel_desc, events


def _collect_events_for_telescope(simtel_file, tel_id, tel_id_map, event_ids, max_events, verbose):
    """Collect matching events for one telescope during a file scan."""
    ids_with_data, events = [], []
    for event in simtel_file:
        if event_ids and event["event_id"] not in event_ids:
            continue
        if tel_id in event["telescope_events"]:
            events.append(event["telescope_events"][tel_id])
            ids_with_data.append(event["event_id"])
            if max_events and len(events) >= max_events:
                break
        elif verbose:
            _log_event_without_telescope_data(event, tel_id_map)
    return ids_with_data, events


def _log_event_without_telescope_data(event, tel_id_map):
    """Log trigger information for an event without selected telescope data."""
    triggered = event["trigger_information"]["triggered_telescopes"]
    triggered_names = [tel_id_map.get(tid, f"ID {tid}") for tid in triggered]
    _logger.debug(
        f"event {event['event_id']} with {len(event['telescope_events'])} "
        f"telescope events (triggered telescopes: {triggered_names})"
    )


def read_events_for_telescopes(
    file_name, telescopes, event_ids=None, max_events=None, verbose=False
):
    """Read events for several telescopes while scanning a file once.

    Parameters
    ----------
    file_name : str or Path
        Path to the sim_telarray file.
    telescopes : list[str]
        Telescope names to read.
    event_ids : int or list of int, optional
        Specific event IDs to include.
    max_events : int, optional
        Maximum number of events to read for each telescope. ``None`` reads all.
    verbose : bool, optional
        If True, log detailed information about events without selected telescope data.

    Returns
    -------
    tuple
        Event IDs, telescope descriptions by name, and events by telescope name.
        Returns ``(None, None, None)`` if a requested telescope is unavailable.
    """
    telescope_names = gen.ensure_list(telescopes)
    telescope_ids = _telescope_ids_for_names(file_name, telescope_names)
    if telescope_ids is None:
        return None, None, None

    event_ids = gen.ensure_list(event_ids)

    with SimTelFile(file_name, skip_calibration=False) as simtel_file:
        descriptions = _telescope_descriptions(simtel_file, telescope_ids, file_name)
        if descriptions is None:
            return None, None, None

        ids_with_data, events_by_telescope = _collect_events_for_telescope_ids(
            simtel_file, telescope_ids, event_ids, max_events, verbose
        )

    for telescope, events in events_by_telescope.items():
        _logger.info(
            f"Read {len(events)} events for telescope '{telescope}' from file '{file_name}'."
        )
    return ids_with_data, descriptions, events_by_telescope


def _telescope_ids_for_names(file_name, telescope_names):
    """Return telescope IDs for requested names, logging missing names."""
    tel_id_map = get_sim_telarray_telescope_id_to_telescope_name_mapping(file_name)
    telescope_ids = {
        telescope: next((tel_id for tel_id, name in tel_id_map.items() if name == telescope), None)
        for telescope in telescope_names
    }
    missing = [name for name, tel_id in telescope_ids.items() if tel_id is None]
    for telescope in missing:
        _logger.warning(f"Telescope type '{telescope}' not found in file '{file_name}'.")
    return None if missing else telescope_ids


def _telescope_descriptions(simtel_file, telescope_ids, file_name):
    """Return descriptions for selected telescopes, logging missing descriptions."""
    descriptions = {
        telescope: simtel_file.telescope_descriptions.get(tel_id)
        for telescope, tel_id in telescope_ids.items()
    }
    missing = [name for name, description in descriptions.items() if description is None]
    for telescope in missing:
        _logger.warning(
            f"Telescope ID '{telescope_ids[telescope]}' not found in file '{file_name}'."
        )
    return None if missing else descriptions


def _collect_events_for_telescope_ids(simtel_file, telescope_ids, event_ids, max_events, verbose):
    """Collect selected telescope events during one sim_telarray file scan."""
    ids_with_data = []
    events_by_telescope = {telescope: [] for telescope in telescope_ids}
    if max_events == 0:
        return ids_with_data, events_by_telescope
    for event in simtel_file:
        if _requested_event(event, event_ids):
            _collect_event_data(
                event, telescope_ids, events_by_telescope, ids_with_data, max_events, verbose
            )
        if _all_limits_reached(events_by_telescope, max_events):
            break
    return ids_with_data, events_by_telescope


def _requested_event(event, event_ids):
    """Return True if the event matches the requested IDs."""
    return not event_ids or event["event_id"] in event_ids


def _collect_event_data(
    event, telescope_ids, events_by_telescope, ids_with_data, max_events, verbose
):
    """Append matching telescope data for one event and track its ID."""
    telescope_events = event.get("telescope_events", {})
    found_telescope_data = False
    for telescope, tel_id in telescope_ids.items():
        if tel_id in telescope_events and (
            max_events is None or len(events_by_telescope[telescope]) < max_events
        ):
            events_by_telescope[telescope].append(telescope_events[tel_id])
            found_telescope_data = True
    if found_telescope_data:
        ids_with_data.append(event["event_id"])
    elif verbose:
        _logger.debug(
            f"event {event['event_id']} has no data for selected telescopes {list(telescope_ids)}"
        )


def _all_limits_reached(events_by_telescope, max_events):
    """Return True once every telescope reached its event limit."""
    return max_events is not None and all(
        len(events) >= max_events for events in events_by_telescope.values()
    )
