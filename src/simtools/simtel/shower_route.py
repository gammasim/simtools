"""Route CORSIKA photon bunches through a ray-tracing backend to sim_telarray, as a stream.

The route process reads the IACT photon bunches of a CORSIKA run from a binary stream, traces
the bunches of each telescope through the optics of its telescope model and writes them as
camera photon bunches for sim_telarray with its optics bypassed (see ``photon_bunch_file``):

    corsika ... | shower_route --telescope-map map.json --telescope-ids 1 | sim_telarray ...

where shower_route is ``python -m simtools.simtel.shower_route`` and sim_telarray reads the
stream with ``-C input_file=-``. The input and the output can also be files (``--input``,
``--output``), e.g. for a CORSIKA IACT file kept on disk.

The telescope map holds all telescopes of the array. ``--telescope-ids`` selects the
traced telescopes. Each input event and core reuse is retained, with unselected telescope
photon blocks omitted. All other top-level objects are passed through unchanged.
"""

import argparse
import json
import logging
import math
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
from astropy.table import Table

from simtools import constants
from simtools.application.model_reader import create_model_reader
from simtools.corsika.photon_bunches import iter_photon_objects
from simtools.io.output_paths import OutputPaths
from simtools.model.model_utils import export_model_files
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing.backends.interface import create_shower_tracer
from simtools.runtime.context import RunContext
from simtools.simtel.photon_bunch_file import write_shower_photon_stream
from simtools.simtel.shower_photons import (
    camera_bunches_of_event,
    derive_seed,
    minimum_input_plane_height_m,
    shower_bunch_frame,
)
from simtools.simtel.shower_transmission import SimtelTransmission, transmission_ratio
from simtools.simulation.configuration import get_model_writer

_logger = logging.getLogger(__name__)

TELESCOPE_MAP_KEYS = (
    "model",
    "model_version",
    "site",
    "dual_mirror",
    "radius_m",
    "input_plane_height_m",
    "theta_deg",
    "phi_deg",
)
SITES = ("North", "South")
DEFAULT_SEED = 4242
# EventIO types of the CORSIKA IACT stream (sim_telarray's IO_TYPE_MC_* values).
_IO_TYPE_ATMOSPHERIC_PROFILE = 1216
_IO_TYPE_EVENT_HEADER = 1202
# Objects before the first event header that multipipe_corsika does not wait for (run header,
# input card, telescope positions); sim_telarray skips their wakeups in the same way.
_SEQUENTIAL_SKIPPED_OBJECTS = 3


@dataclass(frozen=True)
class TelescopeRoute:
    """One entry of the telescope map.

    Parameters
    ----------
    telescope_id: int
        Telescope id in the IACT stream.
    name: str
        Telescope model name.
    model_version: str
        Model version.
    site: str
        Site name (North or South).
    dual_mirror: bool
        Dual-mirror telescope.
    radius_m: float
        Retained for compatibility; the input stream supplies telescope positions.
    input_plane_height_m: float
        Height of the input plane above the telescope sphere centre [m].
    theta_deg: float
        Telescope zenith angle [deg].
    phi_deg: float
        Telescope azimuth in the array frame [deg].
    """

    telescope_id: int
    name: str
    model_version: str
    site: str
    dual_mirror: bool
    radius_m: float
    input_plane_height_m: float
    theta_deg: float
    phi_deg: float


@dataclass(frozen=True)
class RouteInputs:
    """Tracers and site inputs of a telescope map.

    Parameters
    ----------
    tracers: dict
        Shower-photon tracer of each telescope id.
    observation_level_m: float
        Retained for compatibility; the input stream supplies this value.
    atmospheric_profile: astropy.table.Table
        Retained for compatibility; the input stream supplies the profile.
    transmission: SimtelTransmission or None
        Atmospheric transmission lookup.
    pathlen_m_by_telescope: dict or None
        sim_telarray instrument path lengths [m].
    """

    tracers: dict
    observation_level_m: float
    atmospheric_profile: Table
    transmission: object = None
    pathlen_m_by_telescope: dict = None


class MultipipeWakeup:
    """
    Wakeup signals of the sequential mode of multipipe_corsika for the route process.

    In sequential mode multipipe_corsika writes one top-level object after the other to its
    consumer, and waits for a wakeup signal from the consumer after each object before it
    continues CORSIKA. The consumer registers itself with an init signal sent to the
    multipipe process; the sender of all signals must be the same process. The rules follow
    sim_telarray: the first objects before the first event header get no wakeup, except that an
    atmospheric profile does not count among them, and every other object gets one wakeup
    after it has been processed.

    Parameters
    ----------
    pid: int
        Process id of multipipe_corsika (MULTIPIPE_PID).
    signal: int
        Wakeup signal (MULTIPIPE_SIGNAL).
    signal_init: int
        Init signal of the registration (MULTIPIPE_SIGNAL_INIT).
    send: callable
        Sends a signal, called as ``send(pid, signal)`` (default: ``os.kill``).
    """

    def __init__(self, pid, signal, signal_init, send=os.kill):
        self._pid = pid
        self._signal = signal
        self._signal_init = signal_init
        self._send = send
        self._skipped = _SEQUENTIAL_SKIPPED_OBJECTS

    @classmethod
    def from_environment(cls, environ=None, send=os.kill):
        """
        Return the wakeup of a route started by multipipe_corsika in sequential mode.

        Parameters
        ----------
        environ: mapping or None
            Environment (default: ``os.environ``).
        send: callable
            Signal sender (see ``MultipipeWakeup``).

        Returns
        -------
        MultipipeWakeup or None
            None unless MULTIPIPE_SEQUENTIAL is set and the process id and both signals are
            valid (as in sim_telarray).
        """
        environ = os.environ if environ is None else environ
        if "MULTIPIPE_SEQUENTIAL" not in environ:
            return None
        pid, signal, signal_init = (
            _environment_int(environ, name)
            for name in ("MULTIPIPE_PID", "MULTIPIPE_SIGNAL", "MULTIPIPE_SIGNAL_INIT")
        )
        if pid <= 1 or signal <= 0 or signal_init <= 0:
            return None
        return cls(pid, signal, signal_init, send=send)

    def start(self):
        """Register this process with multipipe_corsika (init signal)."""
        self._send_to_multipipe(self._signal_init)

    def on_object(self, obj):
        """
        Send the wakeup after a top-level object of the input has been processed.

        Parameters
        ----------
        obj: EventIO object
            Top-level object of the input stream (its header holds the object type).
        """
        object_type = obj.header.type
        if object_type == _IO_TYPE_ATMOSPHERIC_PROFILE and self._skipped > 0:
            return
        if object_type == _IO_TYPE_EVENT_HEADER:
            self._skipped = 0
        if self._skipped > 0:
            self._skipped -= 1
            return
        self._send_to_multipipe(self._signal)

    def _send_to_multipipe(self, signal_number):
        try:
            self._send(self._pid, signal_number)
        except OSError as exc:
            _logger.warning(
                "Signal %d to multipipe_corsika (pid %d) failed: %s", signal_number, self._pid, exc
            )


def _environment_int(environ, name):
    """Return an environment variable as integer like C's atoi (0 if missing or invalid)."""
    try:
        return int(environ.get(name, "0"))
    except ValueError:
        return 0


@dataclass(frozen=True)
class _TracedTelescope:
    """Optical model, input frame and dual-mirror flag of one telescope in the stream."""

    model: object
    frame: object
    dual_mirror: bool


def load_telescope_map(path):
    """
    Read and validate a telescope map from a JSON file.

    Parameters
    ----------
    path: str or Path
        JSON file: ``{"<telescope id>": {"model": ..., "model_version": ..., "site": ...,
        "dual_mirror": ..., "radius_m": ..., "input_plane_height_m": ..., "theta_deg": ...,
        "phi_deg": ...}}``.

    Returns
    -------
    dict
        ``TelescopeRoute`` of each telescope id.
    """
    try:
        mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Telescope map {path} is not valid JSON: {exc}") from exc
    return parse_telescope_map(mapping)


def parse_telescope_map(mapping):
    """
    Validate a telescope map (parsed JSON) and return its routes.

    Parameters
    ----------
    mapping: dict
        Telescope map (see ``load_telescope_map``).

    Returns
    -------
    dict
        ``TelescopeRoute`` of each telescope id.

    Raises
    ------
    ValueError
        If the map is empty, a telescope id is not an integer or occurs twice, or an entry
        lacks a key or has an invalid value.
    """
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Telescope map must be a non-empty object keyed by telescope id")
    routes = {}
    for key, entry in mapping.items():
        telescope_id = _telescope_id(key)
        if telescope_id in routes:
            raise ValueError(f"Telescope id {telescope_id} occurs twice in the telescope map")
        routes[telescope_id] = _telescope_route(telescope_id, entry)
    return routes


def _telescope_id(key):
    try:
        return int(key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Telescope id {key!r} in the telescope map is not an integer") from exc


def _telescope_route(telescope_id, entry):
    if not isinstance(entry, dict):
        raise ValueError(f"Entry of telescope {telescope_id} in the telescope map is not an object")
    missing = [key for key in TELESCOPE_MAP_KEYS if key not in entry]
    if missing:
        raise ValueError(
            f"Telescope {telescope_id} in the telescope map lacks {', '.join(missing)}"
        )
    if entry["site"] not in SITES:
        raise ValueError(f"Telescope {telescope_id}: site must be one of {', '.join(SITES)}")
    if not isinstance(entry["dual_mirror"], bool):
        raise ValueError(f"Telescope {telescope_id}: dual_mirror must be true or false")
    for key in ("model", "model_version"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise ValueError(f"Telescope {telescope_id}: {key} must be a non-empty string")
    return TelescopeRoute(
        telescope_id=telescope_id,
        name=entry["model"],
        model_version=entry["model_version"],
        site=entry["site"],
        dual_mirror=entry["dual_mirror"],
        radius_m=_positive_number(telescope_id, entry, "radius_m"),
        input_plane_height_m=_positive_number(telescope_id, entry, "input_plane_height_m"),
        theta_deg=_number(telescope_id, entry, "theta_deg"),
        phi_deg=_number(telescope_id, entry, "phi_deg"),
    )


def _is_finite_number(value):
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _number(telescope_id, entry, key):
    value = entry[key]
    if not _is_finite_number(value):
        raise ValueError(f"Telescope {telescope_id}: {key} must be a number")
    return float(value)


def _positive_number(telescope_id, entry, key):
    value = entry[key]
    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f"Telescope {telescope_id}: {key} must be a positive number")
    return float(value)


def _single_value(routes, attribute):
    """Return the value of an attribute shared by all telescopes of a map."""
    values = {getattr(route, attribute) for route in routes.values()}
    if len(values) != 1:
        raise ValueError(
            f"All telescopes of one route stream must share {attribute}; found {sorted(values)}"
        )
    return values.pop()


def create_route_inputs(routes, *, simulation_models_path=None, backend=None, context=None):
    """
    Create the shower-photon tracers and the site inputs of the telescopes of a map.

    All telescopes of a map share the site and the model version (the site inputs are
    read once). The telescope models are read from the simulation models.

    Parameters
    ----------
    routes: dict
        ``TelescopeRoute`` of each telescope id (see ``load_telescope_map``).
    simulation_models_path: str or Path or None
        Root of the simulation models; default: ``SIMTOOLS_SIMULATION_MODELS_PATH``.
    backend: str or None
        Ray-tracing backend; default: the shower-photon backend.
    context: RunContext or None
        Run context of the model exports (default: the environment context).

    Returns
    -------
    RouteInputs
        Tracers, observation level and atmospheric profile.
    """
    site = _single_value(routes, "site")
    model_version = _single_value(routes, "model_version")
    reader = create_model_reader(simulation_models_path=simulation_models_path)
    site_model = SiteModel(
        site=site, model_version=model_version, model_reader=reader, context=context
    )
    export_model_files(site_model, update_if_necessary=True)
    profile_name = site_model.get_parameter_value("atmospheric_profile")
    observation_level = site_model.get_parameter_value_with_unit(
        "corsika_observation_level"
    ).to_value(u.m)
    parameter = site_model.parameters["atmospheric_transmission"]
    table_name = parameter["value"]
    if str(table_name).lower().endswith(".ecsv"):
        table_name = get_model_writer(site_model).export_model_parameter_as_simtel_file(
            "atmospheric_transmission",
            site_model.config_file_directory,
            "atmospheric_transmission",
            "shower-route-transmission.dat",
        )
    transmission = SimtelTransmission(
        site_model.config_file_directory / table_name, observation_level
    )
    tracers = {}
    path_lengths = {}
    for telescope_id, route in routes.items():
        telescope_model = TelescopeModel(
            site=route.site,
            telescope_name=route.name,
            model_version=route.model_version,
            model_reader=reader,
            context=context,
        )
        # sim_config.c zeros the constant polynomial coefficients before calculating
        # pathlen_tel for mirror_class 2; single-mirror telescopes use focal_length.
        if route.dual_mirror:
            path_lengths[telescope_id] = 0.0
        else:
            path_lengths[telescope_id] = telescope_model.get_parameter_value_with_unit(
                "focal_length"
            ).to_value(u.m)
        tracers[telescope_id] = create_shower_tracer(
            telescope_model, name=backend or constants.SHOWER_PHOTON_RAY_TRACING_BACKEND
        )
    return RouteInputs(
        tracers=tracers,
        observation_level_m=observation_level,
        atmospheric_profile=Table.read(site_model.config_file_directory / profile_name),
        transmission=transmission,
        pathlen_m_by_telescope=path_lengths,
    )


def route_stream(
    in_stream,
    out_stream,
    tracers,
    *,
    theta_deg_by_telescope,
    phi_deg,
    input_plane_height_m_by_telescope,
    seed,
    dual_mirror_by_telescope=None,
    skipped_telescope_ids=(),
    threads=0,
    tracing_seed=None,
    on_object=None,
    transmission=None,
    pathlen_m_by_telescope=None,
):
    """
    Trace the photon bunches of a CORSIKA IACT stream and write camera photon bunches.

    The optical models and input planes of all telescopes are checked before the stream
    is read, since a stream cannot be rolled back. Each telescope block of the input is
    traced with the tracer of its telescope id in its original event and array.

    Parameters
    ----------
    in_stream: binary stream
        CORSIKA IACT photon bunches (uncompressed stream, e.g. ``sys.stdin.buffer``).
    out_stream: binary stream
        Output stream of photon bunches for sim_telarray.
    tracers: dict
        Shower-photon tracer (``optical_model(theta_deg, seed)``) of each telescope id.
    theta_deg_by_telescope: dict
        Telescope zenith angle [deg] of each telescope id.
    phi_deg: float
        Telescope azimuth in the array frame [deg].
    input_plane_height_m_by_telescope: dict
        Input plane height above the sphere centre [m] of each telescope id.
    seed: int
        Instrument seed; each telescope derives its own mirror realisation from it and its id.
    dual_mirror_by_telescope: dict or None
        Dual-mirror flag of each telescope id (default: single mirror).
    skipped_telescope_ids: iterable of int
        Telescope ids of the stream whose photon blocks are omitted.
    threads: int
        Number of threads used for tracing (0: default of the optical model).
    tracing_seed: int or None
        Seed of the random tracing draws (default: ``seed``). Each telescope block derives its
        own tracing seed from it, the event number, the array id and the telescope id.
    on_object: callable or None
        Called with each top-level input object once it is processed (see
        ``read_photon_bunches``); used for the wakeups of sequential multipipe_corsika.
    transmission: SimtelTransmission or None
        Lookup for correcting the atmospheric transmission; None disables the correction.
    pathlen_m_by_telescope: dict or None
        sim_telarray instrument path lengths [m], required with the lookup.

    Returns
    -------
    int
        Number of events written.

    Raises
    ------
    ValueError
        If a telescope id of the stream is not in ``tracers``, a telescope has no parameters,
        or an input plane height is below the minimum for its telescope.
    """
    dual_mirror_by_telescope = dual_mirror_by_telescope or {}
    tracing_seed = seed if tracing_seed is None else tracing_seed
    telescopes = {
        telescope_id: _traced_telescope(
            telescope_id,
            tracer,
            theta_deg_by_telescope=theta_deg_by_telescope,
            phi_deg=phi_deg,
            input_plane_height_m_by_telescope=input_plane_height_m_by_telescope,
            dual_mirror=dual_mirror_by_telescope.get(telescope_id, False),
            seed=seed,
        )
        for telescope_id, tracer in tracers.items()
    }
    correction = (
        None
        if transmission is None
        else _TransmissionCorrection(transmission, pathlen_m_by_telescope or {})
    )
    count = write_shower_photon_stream(
        out_stream,
        _camera_events(
            in_stream,
            telescopes,
            skipped_telescope_ids=frozenset(skipped_telescope_ids),
            seed=tracing_seed,
            threads=threads,
            on_object=on_object,
            correction=correction,
        ),
    )
    if correction is not None:
        _logger.info("Dropped %d bunches with zero bypass transmission", correction.dropped)
    _logger.info("Routed %d events of photon bunches for %d telescopes", count, len(telescopes))
    return count


def _traced_telescope(
    telescope_id,
    tracer,
    *,
    theta_deg_by_telescope,
    phi_deg,
    input_plane_height_m_by_telescope,
    dual_mirror,
    seed,
):
    """Optical model, input frame and dual-mirror flag of one telescope; checks the input plane."""
    theta_deg = _parameter(theta_deg_by_telescope, telescope_id, "zenith angle")
    height_m = _parameter(input_plane_height_m_by_telescope, telescope_id, "input plane height")
    model = tracer.optical_model(theta_deg, derive_seed(seed, telescope_id))
    minimum_m = minimum_input_plane_height_m(model, theta_deg)
    if height_m < minimum_m:
        raise ValueError(
            f"input_plane_height_m = {height_m:.2f} m of telescope {telescope_id} is below the "
            f"minimum {minimum_m:.2f} m: the input plane must lie above the incoming obscurers"
        )
    return _TracedTelescope(
        model=model,
        frame=shower_bunch_frame(theta_deg, phi_deg, input_plane_height_m=height_m),
        dual_mirror=dual_mirror,
    )


def _parameter(values, telescope_id, name):
    if telescope_id not in values:
        raise ValueError(f"No {name} for telescope id {telescope_id}")
    return values[telescope_id]


def _zero_traced_telescope_heights(raw, telescopes):
    """
    Zero the 1201 z coordinates of the traced telescopes, preserving all other bytes.

    The telescope height z of the 1201 object is the height above the observation level, as
    sim_telarray reads it (see ``_TransmissionCorrection``). Bypass_Optics shifts bunches by
    ztel * airmass along their direction; camera-frame bunches must not receive that shift
    (sim_telarray.c:2740-2790).
    """
    data = bytearray(raw)
    offset = 16  # Sync marker and standard top-level EventIO header.
    count = struct.unpack_from("<i", data, offset)[0]
    for telescope_id in telescopes:
        if 0 <= telescope_id < count:
            struct.pack_into("<f", data, offset + 4 + 4 * (2 * count + telescope_id), 0.0)
    return bytes(data)


class _TransmissionCorrection:
    """
    Apply the native-to-bypass transmission ratio to traced bunch weights.

    Telescope height convention: the z of each telescope in the 1201 object is the CORSIKA z
    of the elevation axis (ground altitude + axis height - observation level). The atmosphere
    height is the observation level plus z, as sim_telarray uses it. The reference-point
    altitude is not involved.
    """

    def __init__(self, table, path_lengths):
        self.table = table
        self.path_lengths = path_lengths
        self.observation_m = None
        self.heights = None
        self.dropped = 0

    def read_header(self, object_type, raw):
        """Read the observation level (type 1200) and telescope heights (type 1201)."""
        if object_type == 1200:
            nlevels = int(struct.unpack_from("<f", raw, 20 + 4 * 4)[0])
            if not 0 < nlevels <= 10:
                raise ValueError("Invalid observation levels in CORSIKA run header")
            self.observation_m = struct.unpack_from("<f", raw, 20 + 4 * (4 + nlevels))[0] / 100
        elif object_type == 1201:
            count = struct.unpack_from("<i", raw, 16)[0]
            self.heights = struct.unpack_from(f"<{count}f", raw, 20 + 8 * count)

    def for_telescope(self, telescope_id):
        """Return the weight correction function for one telescope."""

        def apply(camera, source, indices):
            if self.observation_m is None or self.heights is None:
                raise ValueError("Transmission correction needs run and telescope headers")
            height = self.heights[telescope_id] / 100
            pathlen = self.path_lengths[telescope_id]
            keep = []
            for i, index in enumerate(indices):
                ground = _airmass(source["cx"][index], source["cy"][index])
                bypass = _airmass(camera["cx"][i], camera["cy"][i])
                # read_tel_pos supplies array.ztel directly from 1201. The route writes
                # zero there, so sim_telarray uses observation level + 0 for bypass.
                ratio = transmission_ratio(
                    self.table,
                    camera["wavelength"][i],
                    camera["zem"][i] / 100,
                    ground,
                    bypass,
                    height,
                    0.0,
                    pathlen,
                    self.observation_m,
                )
                if ratio is None:
                    self.dropped += 1
                else:
                    camera["photons"][i] *= ratio
                    keep.append(i)
            return camera[keep]

        return apply


def _airmass(cx, cy):
    # sim_telarray reconstructs the downward cz from the stored 2D direction.
    return 1.0 / math.sqrt(1.0 - float(cx) ** 2 - float(cy) ** 2)


def _camera_events(
    in_stream, telescopes, *, skipped_telescope_ids, seed, threads, on_object=None, correction=None
):
    """Yield input objects with their telescope blocks traced to the camera."""
    for obj, raw, blocks in iter_photon_objects(in_stream, on_object=on_object):
        if blocks is None:
            if correction is not None:
                correction.read_header(obj.header.type, raw)
            if obj.header.type == 1201:
                raw = _zero_traced_telescope_heights(raw, telescopes)
            yield obj, raw, None
            continue
        cameras = _camera_blocks(
            blocks,
            telescopes,
            skipped_telescope_ids=skipped_telescope_ids,
            seed=seed,
            threads=threads,
            correction=correction,
        )
        yield obj, raw, cameras


def _camera_blocks(blocks, telescopes, *, skipped_telescope_ids, seed, threads, correction):
    """Trace the telescope blocks of one event; blocks of skipped telescopes are omitted."""
    cameras = []
    for bunches in blocks:
        telescope_id = bunches["telescope_id"]
        telescope = telescopes.get(telescope_id)
        if telescope is None and telescope_id in skipped_telescope_ids:
            continue
        if telescope is None:
            raise ValueError(
                f"Telescope id {telescope_id} of event {bunches['event_number']} is not in the "
                "telescope map"
            )
        block_seed = derive_seed(seed, bunches["event_number"], bunches["array_id"], telescope_id)
        camera = camera_bunches_of_event(
            telescope.model,
            bunches,
            telescope.frame,
            seed=block_seed,
            threads=threads,
            dual_mirror=telescope.dual_mirror,
            camera_rotation_deg=0.0,
            correction=None if correction is None else correction.for_telescope(telescope_id),
        )
        cameras.append((telescope_id, camera))
    return cameras


def select_routes(routes, telescope_ids=None):
    """
    Return the routes of the telescopes traced by this process.

    Parameters
    ----------
    routes: dict
        ``TelescopeRoute`` of each telescope id of the telescope map.
    telescope_ids: iterable of int or None
        Telescope ids to trace; None selects all telescopes of the map.

    Returns
    -------
    dict
        ``TelescopeRoute`` of each selected telescope id.

    Raises
    ------
    ValueError
        If a selected telescope id is not in the telescope map.
    """
    if telescope_ids is None:
        return dict(routes)
    missing = sorted(set(telescope_ids) - set(routes))
    if missing:
        raise ValueError(f"Telescope ids {missing} are not in the telescope map")
    return {telescope_id: routes[telescope_id] for telescope_id in telescope_ids}


def _telescope_ids_argument(value):
    """Parse a comma-separated list of telescope ids (e.g. ``1,2,5``)."""
    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Telescope ids must be integers: {value!r}") from exc


def core_reuses(card_lines):
    """
    Return the number of core reuses of a CORSIKA run from its input card (``CSCAT`` line).

    Parameters
    ----------
    card_lines: list of str
        Lines of the CORSIKA input card.

    Returns
    -------
    int
        Number of core reuses (1 if the card has no ``CSCAT`` line).
    """
    for line in card_lines:
        tokens = line.split()
        if tokens and tokens[0] == "CSCAT" and len(tokens) > 1:
            return max(1, int(tokens[1]))
    return 1


def parse_args(argv=None):
    """Parse the command line of the route process."""
    parser = argparse.ArgumentParser(
        description=(
            "Trace CORSIKA IACT photon bunches (stdin or --input) and write camera photon bunches "
            "for sim_telarray (stdout or --output)."
        )
    )
    parser.add_argument(
        "--telescope-map", type=Path, required=True, help="JSON telescope map (see docs)."
    )
    parser.add_argument(
        "--telescope-ids",
        type=_telescope_ids_argument,
        default=None,
        help="Comma-separated telescope ids traced by this process (default: all of the map).",
    )
    parser.add_argument(
        "--input", default="-", help="CORSIKA IACT file, or '-' for stdin (default: stdin)."
    )
    parser.add_argument(
        "--output", default="-", help="Camera photon-bunch file, or '-' for stdout (default)."
    )
    parser.add_argument(
        "--corsika-card",
        type=Path,
        default=None,
        help="Accepted for compatibility; the card is copied from the input stream.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Instrument seed.")
    parser.add_argument(
        "--tracing-seed",
        type=int,
        default=None,
        help="Seed of the random tracing draws (default: same as --seed).",
    )
    parser.add_argument(
        "--no-atmospheric-correction",
        action="store_true",
        help="Disable the atmospheric transmission correction for debugging.",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Tracing threads (0: backend default)."
    )
    parser.add_argument(
        "--simulation-models-path",
        type=Path,
        default=None,
        help="Simulation models root (default: SIMTOOLS_SIMULATION_MODELS_PATH).",
    )
    parser.add_argument(
        "--ray-tracing-backend",
        default=constants.SHOWER_PHOTON_RAY_TRACING_BACKEND,
        help="Ray-tracing backend of the route.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Work directory for exported model files (default: a temporary directory).",
    )
    return parser.parse_args(argv)


def main(argv=None, *, inputs_factory=create_route_inputs):
    """
    Run the route process: photon bunches from the input to camera photon bunches on the output.

    Parameters
    ----------
    argv: list of str or None
        Command-line arguments (default: ``sys.argv``).
    inputs_factory: callable
        Creates the ``RouteInputs`` of the selected telescopes (keyword arguments as
        ``create_route_inputs``).

    Returns
    -------
    int
        Exit status.
    """
    args = parse_args(argv)
    wakeup = MultipipeWakeup.from_environment()
    if wakeup is not None:
        wakeup.start()
    work_path = args.output_path or Path(tempfile.gettempdir()) / "simtools-shower-route"
    context = RunContext.from_arguments(
        {"output_path": str(work_path)}, paths=OutputPaths(output_path=str(work_path))
    )
    routes = load_telescope_map(args.telescope_map)
    selected = select_routes(routes, args.telescope_ids)
    inputs = inputs_factory(
        selected,
        simulation_models_path=args.simulation_models_path,
        backend=args.ray_tracing_backend,
        context=context,
    )
    route_stream(
        sys.stdin.buffer if args.input == "-" else args.input,
        sys.stdout.buffer if args.output == "-" else args.output,
        inputs.tracers,
        theta_deg_by_telescope={
            telescope_id: route.theta_deg for telescope_id, route in selected.items()
        },
        phi_deg=_single_value(selected, "phi_deg"),
        input_plane_height_m_by_telescope={
            telescope_id: route.input_plane_height_m for telescope_id, route in selected.items()
        },
        dual_mirror_by_telescope={
            telescope_id: route.dual_mirror for telescope_id, route in selected.items()
        },
        seed=args.seed,
        tracing_seed=args.tracing_seed,
        skipped_telescope_ids=set(routes) - set(selected),
        threads=args.threads,
        on_object=None if wakeup is None else wakeup.on_object,
        transmission=None if args.no_atmospheric_correction else inputs.transmission,
        pathlen_m_by_telescope=inputs.pathlen_m_by_telescope,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    sys.exit(main())
