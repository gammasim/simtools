"""Read photon bunches from CORSIKA IACT EventIO files.

Photon bunches (EventIO objects 1204 containing 1205 in long format) are read
per event and telescope. Values are kept in the units of the file (cm, ns, nm).
"""

import logging

import numpy as np
from eventio import EventIOFile
from eventio.iact import EventHeader, Photons, TelescopeData

_logger = logging.getLogger(__name__)

_PHOTONS_HEADER_BYTES = 12
_EVENT_ID_SHIFT = 32
_SKIPPED_WAVELENGTHS_NM = (1.0,)
_WAVELENGTH_LIMIT_NM = 9000.0
_FLOAT_COLUMNS = {
    "x": "x",
    "y": "y",
    "cx": "cx",
    "cy": "cy",
    "time_ns": "time",
    "zem": "zem",
    "size": "photons",
    "wavelength_nm": "wavelength",
}


def read_photon_bunches(file_name):
    """
    Read photon bunches of each event and telescope from a CORSIKA IACT EventIO file.

    Bunches with a wavelength of 1 nm or of 9000 nm and above are skipped and counted.
    The bunch id combines the event number and the index of the bunch within the
    telescope's bunch array (before skipping).

    Parameters
    ----------
    file_name: str or Path
        CORSIKA IACT EventIO file with photon bunches in long format.

    Yields
    ------
    dict
        Photon bunches of one telescope in one event with keys ``event_number``,
        ``telescope_id``, ``x``, ``y``, ``cx``, ``cy``, ``time_ns``, ``zem``, ``size``,
        ``wavelength_nm``, ``bunch_id``, ``skipped_bunches``, ``wavelength_min_nm``
        and ``wavelength_max_nm`` (event-header wavelength limits, None if absent).

    Raises
    ------
    NotImplementedError
        If the file contains photon bunches in compact format.
    ValueError
        If the file contains photo-electron bunches (negative wavelength).
    """
    event_number = None
    wavelength_limits = {"wavelength_min_nm": None, "wavelength_max_nm": None}
    n_events = n_telescopes = n_skipped = 0
    with EventIOFile(str(file_name)) as f:
        for obj in f:
            if isinstance(obj, EventHeader):
                header = obj.parse()
                event_number = int(header["event_number"])
                wavelength_limits = _wavelength_limits(header)
                n_events += 1
            elif isinstance(obj, TelescopeData):
                for photons in obj:
                    if not isinstance(photons, Photons):
                        continue
                    telescope = _read_telescope_bunches(
                        photons, file_name, event_number, wavelength_limits
                    )
                    n_telescopes += 1
                    n_skipped += telescope["skipped_bunches"]
                    yield telescope
    _logger.debug(
        "Read photon bunches from %s: %d events, %d telescopes, %d skipped bunches",
        file_name,
        n_events,
        n_telescopes,
        n_skipped,
    )


def _wavelength_limits(header):
    """Return the Cherenkov wavelength limits [nm] from an event header."""
    return {
        "wavelength_min_nm": float(header["cherenkov_wavelength_min"]),
        "wavelength_max_nm": float(header["cherenkov_wavelength_max"]),
    }


def _read_telescope_bunches(photons, file_name, event_number, wavelength_limits):
    """Return the photon bunches of one telescope in one event, without skipped bunches."""
    if photons.compact:
        raise NotImplementedError(f"Compact photon bunch format is not supported: {file_name}")
    photons.seek(_PHOTONS_HEADER_BYTES)
    bunches = photons.read_data()
    wavelength = bunches["wavelength"]
    if np.any(wavelength < 0):
        raise ValueError(
            f"Photo-electron bunches (negative wavelength) are not supported: {file_name}"
        )
    skipped = np.isin(wavelength, _SKIPPED_WAVELENGTHS_NM) | (wavelength >= _WAVELENGTH_LIMIT_NM)
    kept = ~skipped
    index = np.arange(len(bunches), dtype=np.uint64)
    bunch_id = (np.uint64(event_number) << np.uint64(_EVENT_ID_SHIFT)) | index

    telescope = {
        "event_number": event_number,
        "telescope_id": int(photons.telescope),
        "bunch_id": bunch_id[kept],
        "skipped_bunches": int(np.count_nonzero(skipped)),
        **wavelength_limits,
    }
    for key, column in _FLOAT_COLUMNS.items():
        telescope[key] = bunches[column][kept].astype(np.float32)
    return telescope
