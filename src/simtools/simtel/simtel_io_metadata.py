#!/usr/bin/python3
"""Read metadata from sim_telarray files."""

from eventio import EventIOFile
from eventio.simtel import HistoryMeta


def read_sim_telarray_metadata(file, encoding="utf8"):
    """
    Return global and per-telescope metadata from sim_telarray file.

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
        for o in f:
            if not isinstance(o, HistoryMeta):
                if global_meta is not None:
                    break
                continue

            meta = decode(o.parse())
            if o.header.id == -1:
                global_meta = meta
            else:
                telescope_meta[o.header.id] = meta

    return global_meta, telescope_meta
