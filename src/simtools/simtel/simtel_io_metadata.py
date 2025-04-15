#!/usr/bin/python3
"""Read metadata from sim_telarray files."""

from functools import cache

from eventio import EventIOFile
from eventio.simtel import HistoryMeta


@cache
def read_sim_telarray_metadata(file, encoding="utf8"):
    """
    Return global and per-telescope metadata from sim_telarray file.

    Cached to avoid re-reading the file multiple times.

    Parameters
    ----------
    file: str
        Path to the sim_telarray file.
    encoding: str
        Encoding to use for decoding metadata. Default is 'utf8'.

    Returns
    -------
    global_meta: dict
        Global metadata dictionary.
    telescope_meta: dict
        Dictionary of telescope metadata, keyed by telescope ID.
    """

    def decode(meta):
        return {k.decode(encoding): v.decode(encoding) for k, v in meta.items()}

    global_meta = None
    telescope_meta = {}

    with EventIOFile(file) as f:
        found_meta = False
        for o in f:
            if isinstance(o, HistoryMeta):
                found_meta = True
            else:
                if found_meta:
                    break
                continue

            meta = decode(o.parse())
            if o.header.id == -1:
                global_meta = meta
            else:
                telescope_meta[o.header.id] = meta

    def clean_meta(meta):
        """Clean metadata dictionary."""
        meta = {k.lower().lstrip("*"): v for k, v in meta.items()}
        return {k: v.strip() if isinstance(v, str) else v for k, v in meta.items()}

    # keys to lower case and strip leading '*', trailing spaces
    return clean_meta(global_meta), {
        tel_id: clean_meta(meta) for tel_id, meta in telescope_meta.items()
    }


def get_sim_telarray_telescope_id(telescope_name, file):
    """
    Return the telescope ID for a given telescope name in a sim_telarray file.

    Translates e.g. 'LSTN-01' to the corresponding telescope ID.

    Parameters
    ----------
    telescope_name: str
        Name of the telescope.
    file: str
        Path to the sim_telarray file.

    Returns
    -------
    int, None
        Telescope ID. Returns None if not found.
    """
    _, telescope_meta = read_sim_telarray_metadata(file)
    telescope_name_to_sim_telarray_id = {}
    for tel_id in telescope_meta.keys():
        _optics_name = telescope_meta[tel_id].get("optics_config_name", None)
        _camera_name = telescope_meta[tel_id].get("camera_config_name", None)
        if _optics_name == _camera_name and _optics_name == telescope_name:
            telescope_name_to_sim_telarray_id[telescope_name] = tel_id

    return telescope_name_to_sim_telarray_id.get(telescope_name, None)
