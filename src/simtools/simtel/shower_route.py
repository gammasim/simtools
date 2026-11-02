"""Route CORSIKA photon bunches through a ray-tracing backend to sim_telarray, as a stream.

The route process reads the IACT photon bunches of a CORSIKA run from a binary stream, traces
the bunches of each telescope through the optics of its telescope model and writes them as
camera photon bunches for sim_telarray with its optics bypassed (see ``photon_bunch_file``):

    corsika ... | shower_route --telescope-map map.json --telescope-ids 1 | sim_telarray ...

where shower_route is ``python -m simtools.simtel.shower_route`` and sim_telarray reads the
stream with ``-C input_file=-``. The input and the output can also be files (``--input``,
``--output``), e.g. for a CORSIKA IACT file kept on disk.

The telescope map holds all telescopes of the array. ``--telescope-ids`` selects the telescopes
traced by this process (default: all); the telescope blocks of the other telescopes are written
as events without telescope blocks, so that sim_telarray processes of other telescope types see
the same event numbers. Each telescope id of the stream must be in the telescope map; an id that
is not in the map is an error. An event without photon bunches of a traced telescope is written
as an empty camera event, and it is not traced.

Output events: each telescope block of the input is written as one output event, in input order,
with the event number and the telescope id of its input block. Several telescope blocks of one
input event therefore give several output events with the same event number.
"""

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools.corsika.photon_bunches import read_photon_bunches
from simtools.model.model_utils import export_model_files
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.model_repository.reader import SimulationModelReader, resolve_model_reader
from simtools.ray_tracing.backends.interface import create_shower_tracer
from simtools.runtime.context import RunContext
from simtools.simtel.photon_bunch_file import PhotonBunchEvent, write_photon_bunches
from simtools.simtel.shower_photons import (
    camera_bunches_of_event,
    minimum_input_plane_height_m,
    shower_bunch_frame,
)
from simtools.utils import names

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
DEFAULT_SEED = 4242
# EventIO types of the CORSIKA IACT stream (sim_telarray's IO_TYPE_MC_* values).
_IO_TYPE_ATMOSPHERIC_PROFILE = 1216
_IO_TYPE_EVENT_HEADER = 1202
# Objects before the first event header that multipipe_corsika does not wait for (run header,
# input card, telescope positions); sim_telarray skips their wakeups in the same way.
_SEQUENTIAL_SKIPPED_OBJECTS = 3


def derive_seed(seed, *keys):
    """
    Return a 32-bit seed derived from a seed and integer keys (e.g. telescope id).

    Parameters
    ----------
    seed: int
        Seed the derived seed is based on.
    keys: int
        Integers identifying the derived stream (non-negative).

    Returns
    -------
    int
        Derived seed in [0, 2**32).
    """
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1, np.uint32)[0])


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
        Radius of the telescope's fiducial sphere [m].
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
        Observation level a.s.l. [m].
    atmospheric_profile: astropy.table.Table
        Atmospheric profile written into the output.
    """

    tracers: dict
    observation_level_m: float
    atmospheric_profile: Table


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
    try:
        site = names.validate_site_name(entry["site"])
    except ValueError as exc:
        raise ValueError(f"Telescope {telescope_id}: {exc}") from exc
    if not isinstance(entry["dual_mirror"], bool):
        raise ValueError(f"Telescope {telescope_id}: dual_mirror must be true or false")
    for key in ("model", "model_version"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise ValueError(f"Telescope {telescope_id}: {key} must be a non-empty string")
    return TelescopeRoute(
        telescope_id=telescope_id,
        name=entry["model"],
        model_version=entry["model_version"],
        site=site,
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


def create_route_inputs(routes, *, simulation_models_path=None, backend=None):
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
        Ray-tracing backend; default: obdeect.

    Returns
    -------
    RouteInputs
        Tracers, observation level and atmospheric profile.
    """
    return _create_route_inputs(routes, simulation_models_path, backend, None)


def _create_route_inputs(routes, simulation_models_path, backend, output_path):
    site = _single_value(routes, "site")
    model_version = _single_value(routes, "model_version")
    reader = (
        SimulationModelReader.from_files(simulation_models_path)
        if simulation_models_path
        else resolve_model_reader()
    )
    if reader is None:
        raise ValueError("A simulation-model source is required for shower photon routing")
    output_path = output_path or Path(tempfile.gettempdir()) / "simtools-shower-route"
    context = RunContext.from_arguments({"output_path": output_path}, model_reader=reader)
    site_model = SiteModel(site=site, model_version=model_version, context=context)
    export_model_files(site_model, update_if_necessary=True)
    profile_name = site_model.get_parameter_value("atmospheric_profile")
    tracers = {}
    for telescope_id, route in routes.items():
        telescope_model = TelescopeModel(
            site=route.site,
            telescope_name=route.name,
            model_version=route.model_version,
            context=context,
        )
        tracers[telescope_id] = create_shower_tracer(telescope_model, name=backend or "obdeect")
    return RouteInputs(
        tracers=tracers,
        observation_level_m=site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        ).to_value(u.m),
        atmospheric_profile=Table.read(site_model.config_file_directory / profile_name),
    )


def route_stream(
    in_stream,
    out_stream,
    inputs,
    *,
    theta_deg_by_telescope,
    phi_deg,
    input_plane_height_m_by_telescope,
    seed,
    telescope_radius_m,
    dual_mirror_by_telescope=None,
    skipped_telescope_ids=(),
    corsika_card=None,
    reuses=1,
    threads=0,
    tracing_seed=None,
    on_object=None,
):
    """
    Trace the photon bunches of a CORSIKA IACT stream and write camera photon bunches.

    The optical models and input planes of all telescopes are checked before the stream
    is read, since a stream cannot be rolled back. Each telescope block of the input is
    traced with the tracer of its telescope id and written as one output event.

    Parameters
    ----------
    in_stream: binary stream
        CORSIKA IACT photon bunches (uncompressed stream, e.g. ``sys.stdin.buffer``).
    out_stream: binary stream
        Output stream of photon bunches for sim_telarray.
    inputs: RouteInputs
        Shower-photon tracer (``optical_model(theta_deg, seed)``) of each telescope id, observation
        level and atmospheric profile.
    theta_deg_by_telescope: dict
        Telescope zenith angle [deg] of each telescope id.
    phi_deg: float
        Telescope azimuth in the array frame [deg].
    input_plane_height_m_by_telescope: dict
        Input plane height above the sphere centre [m] of each telescope id.
    seed: int
        Instrument seed; each telescope derives its own mirror realisation from it and its id.
    telescope_radius_m: float
        Radius of the telescope's fiducial sphere [m].
    dual_mirror_by_telescope: dict or None
        Dual-mirror flag of each telescope id (default: single mirror).
    skipped_telescope_ids: iterable of int
        Telescope ids of the stream that are not traced (their blocks are written as events
        without telescope blocks).
    corsika_card: list of str or None
        Lines of the CORSIKA input card of the run (sim_telarray reads the number of showers
        from it); None writes no card.
    threads: int
        Number of threads used for tracing (0: default of the optical model).
    tracing_seed: int or None
        Seed of the random tracing draws (default: ``seed``). Each telescope block derives its
        own tracing seed from it, the event number, the array id and the telescope id.
    on_object: callable or None
        Called with each top-level input object once it is processed (see
        ``read_photon_bunches``); used for the wakeups of sequential multipipe_corsika.

    Returns
    -------
    int
        Number of events written.

    Raises
    ------
    ValueError
        If a telescope id of the stream is not in ``inputs.tracers``, a telescope has no parameters,
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
        for telescope_id, tracer in inputs.tracers.items()
    }
    count = write_photon_bunches(
        out_stream,
        _camera_events(
            in_stream,
            telescopes,
            skipped_telescope_ids=frozenset(skipped_telescope_ids),
            seed=tracing_seed,
            threads=threads,
            on_object=on_object,
        ),
        observation_level_m=inputs.observation_level_m,
        telescope_radius_m=telescope_radius_m,
        atmospheric_profile=inputs.atmospheric_profile,
        corsika_card=corsika_card,
        reuses=reuses,
    )
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


def _camera_events(in_stream, telescopes, *, skipped_telescope_ids, seed, threads, on_object=None):
    """Yield one output event per telescope block of the input stream (same event number).

    Blocks of skipped telescopes (in the telescope map, but not traced by this process) are
    written as events without telescope blocks. Each traced block uses its own tracing seed,
    derived from the event number, the array id and the telescope id, so that the random draws
    differ between telescopes and between core reuses of one event.
    """
    for bunches in read_photon_bunches(in_stream, on_object=on_object):
        telescope_id = bunches["telescope_id"]
        telescope = telescopes.get(telescope_id)
        if telescope is None and telescope_id in skipped_telescope_ids:
            yield PhotonBunchEvent(number=bunches["event_number"], blocks=())
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
        )
        yield PhotonBunchEvent(number=bunches["event_number"], blocks=((telescope_id, camera),))


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
        help="CORSIKA input card of the run (its lines are written into the output).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Instrument seed.")
    parser.add_argument(
        "--tracing-seed",
        type=int,
        default=None,
        help="Seed of the random tracing draws (default: same as --seed).",
    )
    parser.add_argument("--threads", type=int, default=0, help="Tracing threads (0: default).")
    parser.add_argument(
        "--simulation-models-path",
        type=Path,
        default=None,
        help="Simulation models root (default: SIMTOOLS_SIMULATION_MODELS_PATH).",
    )
    parser.add_argument(
        "--ray-tracing-backend",
        default="obdeect",
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
    routes = load_telescope_map(args.telescope_map)
    selected = select_routes(routes, args.telescope_ids)
    card_lines = None if args.corsika_card is None else args.corsika_card.read_text().splitlines()
    if inputs_factory is create_route_inputs:
        inputs = _create_route_inputs(
            selected, args.simulation_models_path, args.ray_tracing_backend, work_path
        )
    else:
        inputs = inputs_factory(
            selected,
            simulation_models_path=args.simulation_models_path,
            backend=args.ray_tracing_backend,
        )
    route_stream(
        sys.stdin.buffer if args.input == "-" else args.input,
        sys.stdout.buffer if args.output == "-" else args.output,
        inputs,
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
        telescope_radius_m=_single_value(selected, "radius_m"),
        skipped_telescope_ids=set(routes) - set(selected),
        corsika_card=card_lines,
        reuses=core_reuses(card_lines or []),
        threads=args.threads,
        on_object=None if wakeup is None else wakeup.on_object,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    sys.exit(main())
