"""Read photon bunches from CORSIKA IACT EventIO files.

Photon bunches (EventIO objects 1204 containing 1205 in long format) are read
per event and telescope. Values are kept in the units of the file (cm, ns, nm).
Files are read from a path or from an uncompressed binary stream (e.g. a pipe).
"""

import gzip
import io
import logging
import os

import numpy as np
import zstandard
from eventio.base import KNOWN_OBJECTS, EventIOObject, read_header, read_sync_marker
from eventio.iact import EventHeader, Photons, TelescopeData

_logger = logging.getLogger(__name__)

_PHOTONS_HEADER_BYTES = 12
_EVENT_ID_SHIFT = 32
_SKIPPED_WAVELENGTHS_NM = (1.0,)
_WAVELENGTH_LIMIT_NM = 9000.0
_STREAM_CHUNK_BYTES = 1 << 20
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


class _ForwardStream:
    """
    Read-only view of a binary stream with the seek semantics eventio needs.

    Reads return the requested number of bytes unless the stream ends (short reads
    of raw pipes are completed). Seeking is forward only; skipped bytes are read and
    discarded, so the stream is consumed once and no sniffing or rewinding is needed.
    """

    def __init__(self, stream):
        self._stream = stream
        self._pos = 0

    def read(self, size):
        """Read up to size bytes, fewer only at the end of the stream."""
        data = bytearray()
        while len(data) < size:
            chunk = self._stream.read(size - len(data))
            if not chunk:
                break
            data += chunk
        self._pos += len(data)
        return bytes(data)

    def seek(self, offset, whence=0):
        """Seek forward by reading and discarding bytes; backward seeks are rejected."""
        if whence not in (0, 1):
            raise ValueError("Only absolute and relative seeks are supported on a stream")
        target = offset if whence == 0 else self._pos + offset
        if target < self._pos:
            raise OSError("Only forward seeking is possible on a stream")
        to_skip = target - self._pos
        while to_skip > 0:
            chunk = self._stream.read(min(to_skip, _STREAM_CHUNK_BYTES))
            if not chunk:
                break
            to_skip -= len(chunk)
            self._pos += len(chunk)
        return self._pos

    def tell(self):
        """Return the number of bytes consumed so far."""
        return self._pos


def _iter_stream_objects(stream):
    """Yield top-level objects and their exact bytes from a forward-only stream."""
    handle = _ForwardStream(stream)
    while True:
        start = handle.tell()
        captured = bytearray()

        class RecordingStream:
            """Stream wrapper that keeps the exact bytes of the object being read."""

            def read(self, size):
                """Read from the shared handle and record the bytes."""
                data = handle.read(size)
                captured.extend(data)
                return data

            def tell(self):
                """Return the position of the shared handle."""
                return handle.tell()

        recording = RecordingStream()
        try:
            read_sync_marker(recording)
        except StopIteration:
            return
        header = read_header(recording, offset=start, toplevel=True)
        remaining = header.total_size - len(captured)
        captured.extend(handle.read(remaining))
        if len(captured) != header.total_size:
            raise EOFError("Truncated EventIO object")
        raw = bytes(captured)
        buffer = io.BytesIO(raw)
        read_sync_marker(buffer)
        local_header = read_header(buffer, offset=0, toplevel=True)
        obj = KNOWN_OBJECTS.get(local_header.type, EventIOObject)(local_header, filehandle=buffer)
        yield obj, raw


def _iter_path_objects(path):
    """Yield objects from an uncompressed, gzip, or zstd-compressed file."""
    with open(path, "rb") as source:
        magic = source.read(4)
        source.seek(0)
        if magic.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=source) as stream:
                yield from _iter_stream_objects(stream)
        elif magic == b"\x28\xb5\x2f\xfd":
            with zstandard.ZstdDecompressor().stream_reader(source) as stream:
                yield from _iter_stream_objects(stream)
        else:
            yield from _iter_stream_objects(source)


def _is_path(file_name):
    return isinstance(file_name, str | os.PathLike)


def _check_source(file_name):
    """Raise TypeError unless ``file_name`` is a path or a binary stream."""
    if _is_path(file_name):
        return
    if isinstance(file_name, io.TextIOBase):
        raise TypeError(f"Photon bunches need a binary stream, not a text stream: {file_name!r}")
    if not callable(getattr(file_name, "read", None)):
        raise TypeError(f"Expected a file path or a binary stream, got {type(file_name).__name__}")


def read_photon_bunches(file_name, on_object=None):
    """
    Read photon bunches of each event and telescope from a CORSIKA IACT EventIO file or stream.

    Bunches with a wavelength of 1 nm or of 9000 nm and above are skipped and counted.
    The bunch id combines the event number and the index of the bunch within the
    telescope's bunch array (before skipping).

    Parameters
    ----------
    file_name: str, Path or binary stream
        CORSIKA IACT EventIO file with photon bunches in long format, or an
        uncompressed binary stream of such data (e.g. ``sys.stdin.buffer`` or a pipe).
        Streams are read once, forward only.
    on_object: callable or None
        Called with each top-level EventIO object after its photon bunches have been yielded
        and the consumer has asked for the next item, i.e. once the object is processed.
        Objects of other types are passed too.

    Returns
    -------
    Iterator[dict]
        Photon bunches of one telescope in one event with keys ``event_number``,
        ``telescope_id``, ``array_id`` (array index of the telescope block, 0 for a single
        array), ``x``, ``y``, ``cx``, ``cy``, ``time_ns``, ``zem``, ``size``,
        ``wavelength_nm``, ``bunch_id``, ``skipped_bunches``, ``wavelength_min_nm``
        and ``wavelength_max_nm`` (event-header wavelength limits, None if absent).

    Raises
    ------
    TypeError
        If ``file_name`` is neither a path nor a binary stream (e.g. a text stream).
    NotImplementedError
        If the file contains photon bunches in compact format.
    ValueError
        If the file contains photo-electron bunches (negative wavelength).
    """
    _check_source(file_name)
    return (
        bunches
        for _, _, telescopes in iter_photon_objects(file_name, on_object=on_object)
        if telescopes is not None
        for bunches in telescopes
    )


def iter_photon_objects(file_name, on_object=None):
    """Yield each input object, its raw bytes, and parsed telescope blocks if present.

    The callback runs after the consumer has processed each top-level object.
    """
    _check_source(file_name)
    if _is_path(file_name):
        name, objects = str(file_name), _iter_path_objects(file_name)
    else:
        name = getattr(file_name, "name", "<stream>")
        objects = _iter_stream_objects(file_name)
    event_number = None
    wavelength_limits = {"wavelength_min_nm": None, "wavelength_max_nm": None}
    for obj, raw in objects:
        telescopes = None
        if isinstance(obj, EventHeader):
            header = obj.parse()
            event_number = int(header["event_number"])
            wavelength_limits = _wavelength_limits(header)
        elif isinstance(obj, TelescopeData):
            telescopes = [
                _read_telescope_bunches(photons, name, event_number, wavelength_limits)
                for photons in obj
                if isinstance(photons, Photons)
            ]
        yield obj, raw, telescopes
        if on_object is not None:
            on_object(obj)


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
        "array_id": int(photons.array_id),
        "bunch_id": bunch_id[kept],
        "skipped_bunches": int(np.count_nonzero(skipped)),
        **wavelength_limits,
    }
    for key, column in _FLOAT_COLUMNS.items():
        telescope[key] = bunches[column][kept].astype(np.float32)
    return telescope
