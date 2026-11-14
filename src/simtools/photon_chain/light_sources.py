"""Calibration light sources of photon-stream simulation chains.

A light source type turns the simulation models of a calibration device and a telescope into
a request (a frozen dataclass without reference to any program) and the request into the
command that makes the photons. The request does not depend on the command line of a
photon-source program: replacing the program means replacing the ``command`` of a source type.

Source types are looked up by name in ``SOURCE_TYPES``, which maps each name to a factory
returning the source-type class; factories defer imports until the type is used. Further types
are added with :func:`register_source_type`; no existing code changes.

A source-type class provides:

- ``name`` and ``light_source_types`` (the ``light_source_type`` model values it handles),
- ``build_request(calibration_model, telescope_model, site_model, config, seed)``,
- ``command(request, work_directory, output=None, file_token="")``: the command tokens; files
  the program reads are written to ``work_directory``.

The models are used through ``get_parameter_value`` and ``get_parameter_value_with_unit``.
``config`` holds the settings of a light-emission run: ``number_of_events``, ``flasher_photons``,
``wavelength``, ``light_source_position``, ``light_source_pointing`` and ``run_number``.
"""

import json
import sys
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.simulation.light_source_profiles import angular_profile, pulse_profile
from simtools.utils import value_conversion
from simtools.utils.geometry import fiducial_radius_from_shape

FLASHER_MODULE = "simtools.photon_chain.flasher_stream"
ILLUMINATOR_EXECUTABLE = "obdeect-sources-point-source"
WAVELENGTH_TOLERANCE_NM = 0.5
#: Illuminator pointing of the artificial setup with the telescope at the origin.
POINTING_DOWN = (0.0, 0.0, -1.0)
_CM = u.cm


@dataclass(frozen=True)
class AngularDistribution:
    """Angular emission profile: shape name (as the model parameter) and width [deg]."""

    shape: str
    width_deg: float


@dataclass(frozen=True)
class PulseShape:
    """Time profile of a pulse: shape name (as the model parameter), width and decay [ns]."""

    shape: str
    width_ns: float
    decay_ns: float


@dataclass(frozen=True)
class FlasherRequest:
    """
    Flasher inside a telescope, lighting the camera directly.

    Parameters
    ----------
    position_m : tuple of float
        Flasher position in the telescope frame [m] (+z along the optical axis).
    camera_radius_m : float
        Fiducial radius of the camera [m].
    events : tuple of int
        Events (pulses) per photon level.
    photons : tuple of int
        Photons emitted into 4 pi per pulse, one value per photon level.
    wavelength_nm : float
        Wavelength [nm].
    angular : AngularDistribution
        Angular emission profile.
    pulse : PulseShape
        Time profile of the pulse.
    observation_level_m : float
        Observation level [m].
    telescope, site, model_version : str
        Telescope model that traces the light.
    run_number : int
        Run number.
    seed : int
        Simulation seed.
    instrument_seed : int
        Seed of the random instrument realisation.
    """

    position_m: tuple
    camera_radius_m: float
    events: tuple
    photons: tuple
    wavelength_nm: float
    angular: AngularDistribution
    pulse: PulseShape
    observation_level_m: float
    telescope: str = ""
    site: str = ""
    model_version: str = ""
    run_number: int = 1
    seed: int = 1
    instrument_seed: int = 1

    @property
    def photons_per_pulse(self):
        """Return the photons emitted into 4 pi in each pulse (one entry per event)."""
        return tuple(n for count, n in zip(self.events, self.photons) for _ in range(count))


def request_to_json(request, path):
    """Write a ``FlasherRequest`` as JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(request)), encoding="utf-8")
    return path


def flasher_request_from_json(path):
    """Return the ``FlasherRequest`` written by :func:`request_to_json`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("position_m", "events", "photons"):
        data[name] = tuple(data[name])
    data["angular"] = AngularDistribution(**data["angular"])
    data["pulse"] = PulseShape(**data["pulse"])
    return FlasherRequest(**data)


@dataclass(frozen=True)
class IlluminatorRequest:
    """
    Illuminator outside the telescope, lighting it from a distance (sky-frame photons).

    Parameters
    ----------
    position_m : tuple of float
        Illuminator position in the ground frame [m] (array frame: x north, y west, z up).
    pointing : tuple of float
        Unit vector of the illuminator axis (ground frame).
    telescope_position_m : tuple of float
        Telescope position in the ground frame [m], corrected for the axes offsets.
    telescope_radius_m : float
        Radius of the telescope sphere [m].
    telescope_pointing_deg : tuple of float
        Zenith and azimuth angle of the telescope pointing towards the illuminator [deg].
    events : int
        Events per photon level.
    photons : tuple of int
        Photons emitted into 4 pi per event, one value per photon level.
    bunch_size : float
        Photons per bunch.
    wavelength_nm : float
        Wavelength [nm].
    angular : AngularDistribution
        Angular emission profile.
    pulse : PulseShape
        Time profile of the pulse.
    observation_level_m : float
        Observation level [m].
    run_number : int
        Run number.
    seed : int
        Random seed.
    """

    position_m: tuple
    pointing: tuple
    telescope_position_m: tuple
    telescope_radius_m: float
    telescope_pointing_deg: tuple
    events: int
    photons: tuple
    bunch_size: float
    wavelength_nm: float
    angular: AngularDistribution
    pulse: PulseShape
    observation_level_m: float
    run_number: int = 1
    seed: int = 1


# ---------------------------------------------------------------------------
# Quantities of the models and the run configuration (pure functions)
# ---------------------------------------------------------------------------


def _meters(value):
    """Return a length (quantity, or a number in m) as float [m]."""
    return float(value.to_value(u.m)) if hasattr(value, "to_value") else float(value)


def event_photon_sequences(number_of_events, photons):
    """
    Return the events and photons per photon level.

    One event value applies to all photon levels; otherwise there is one value per level.

    Parameters
    ----------
    number_of_events : int, list of int or str
        Events (one value, or one value per photon level).
    photons : int, list of int or str
        Photons per event (one value per photon level).

    Returns
    -------
    tuple of tuple of int
        Events and photons, one value each per photon level.
    """
    photons = tuple(value_conversion.parse_typed_sequence(photons, int))
    events = tuple(value_conversion.parse_typed_sequence(number_of_events, int))
    if not photons:
        raise ValueError("No photon yield: set flasher_photons in the model or the configuration.")
    if len(events) == 1:
        events *= len(photons)
    if len(events) != len(photons):
        raise ValueError(
            "Invalid number_of_events list length. Use one value or one value per photon intensity."
        )
    return events, photons


def select_wavelength(calibration_model, requested=None):
    """
    Return the wavelength [nm] of the light source.

    Parameters
    ----------
    calibration_model : CalibrationModel
        Calibration device model (parameter ``flasher_wavelength``).
    requested : astropy.units.Quantity, optional
        Requested wavelength; it must match a model wavelength within 0.5 nm.

    Returns
    -------
    float
        Wavelength of the model closest to the requested wavelength [nm].
    """
    allowed = [
        float(w.to_value(u.nm))
        for w in value_conversion.ensure_list(
            calibration_model.get_parameter_value_with_unit("flasher_wavelength")
        )
    ]
    if requested is None:
        if len(allowed) != 1:
            raise ValueError(
                f"The light source has several wavelengths {allowed} nm; select one with "
                "the wavelength option."
            )
        return allowed[0]
    requested_nm = float(requested.to_value(u.nm))
    if not any(abs(requested_nm - wavelength) < WAVELENGTH_TOLERANCE_NM for wavelength in allowed):
        raise ValueError(
            f"Wavelength {requested_nm} nm is not supported by the light source. "
            f"Allowed wavelengths are: {allowed} nm"
        )
    return min(allowed, key=lambda wavelength: abs(wavelength - requested_nm))


def angular_distribution(calibration_model):
    """Return the angular emission profile of the calibration device model."""
    width = calibration_model.get_parameter_value_with_unit("flasher_angular_distribution_width")
    return AngularDistribution(
        shape=str(calibration_model.get_parameter_value("flasher_angular_distribution")),
        width_deg=0.0 if width is None else float(width.to_value(u.deg)),
    )


def pulse_shape(calibration_model):
    """Return the pulse shape of the calibration device model."""
    shape, width_ns, decay_ns = calibration_model.get_parameter_value("flasher_pulse_shape")
    return PulseShape(shape=str(shape), width_ns=float(width_ns), decay_ns=float(decay_ns))


def _photon_settings(calibration_model, config):
    """Return (events, photons) per photon level, the configuration overriding the model."""
    photons = config.get("flasher_photons")
    if photons is None:
        photons = calibration_model.get_parameter_value("flasher_photons")
    return event_photon_sequences(config.get("number_of_events", 1), photons)


def _common_settings(calibration_model, site_model, config):
    """Return the settings shared by all request types."""
    return {
        "wavelength_nm": select_wavelength(calibration_model, config.get("wavelength")),
        "angular": angular_distribution(calibration_model),
        "pulse": pulse_shape(calibration_model),
        "observation_level_m": _meters(
            site_model.get_parameter_value_with_unit("corsika_observation_level")
        ),
        "run_number": int(config.get("run_number") or 1),
    }


def flasher_request(calibration_model, telescope_model, site_model, config, seed=1):
    """
    Return the request of a flasher (flat-fielding light source) in a telescope.

    The flasher position is given in the primary-mirror frame and converted to the telescope
    frame (``mirror_offset``).

    Parameters
    ----------
    calibration_model : CalibrationModel
        Flasher model.
    telescope_model : TelescopeModel
        Telescope model.
    site_model : SiteModel
        Site model.
    config : dict
        Settings of the light-emission run; ``telescope``, ``site``, ``model_version`` and
        ``instrument_seed`` identify the telescope model and its random realisation.
    seed : int
        Simulation seed.

    Returns
    -------
    FlasherRequest
        Flasher request.
    """
    x_m, y_m, z_m = (
        _meters(c) for c in calibration_model.get_parameter_value_with_unit("flasher_position")[:3]
    )
    mirror_offset_m = _meters(telescope_model.get_parameter_value_with_unit("mirror_offset"))
    camera_radius_m = fiducial_radius_from_shape(
        _meters(telescope_model.get_parameter_value_with_unit("camera_body_diameter")),
        telescope_model.get_parameter_value("camera_body_shape"),
    )
    events, photons = _photon_settings(calibration_model, config)
    return FlasherRequest(
        position_m=(x_m, y_m, z_m - mirror_offset_m),
        camera_radius_m=camera_radius_m,
        events=events,
        photons=photons,
        telescope=str(config.get("telescope") or ""),
        site=str(config.get("site") or ""),
        model_version=str(config.get("model_version") or ""),
        seed=int(seed),
        instrument_seed=int(config.get("instrument_seed") or 1),
        **_common_settings(calibration_model, site_model, config),
    )


# ---------------------------------------------------------------------------
# Illuminator geometry (ground frame)
# ---------------------------------------------------------------------------


def uses_telescope_position(pointing_vector):
    """Return False for the downward beam used with an artificial telescope at the origin."""
    try:
        vector = np.asarray(pointing_vector, dtype=float)
    except TypeError, ValueError:
        return True
    if vector.size < 3 or not np.all(np.isfinite(vector[:3])):
        return True
    return not np.allclose(vector[:3], POINTING_DOWN, atol=1e-6)


def illuminator_position_m(calibration_model, config):
    """Return the illuminator emission position in the ground frame [m] (array)."""
    explicit = config.get("light_source_position")
    if explicit is not None:
        return np.array([_meters(c) for c in explicit], dtype=float)
    x_m, y_m, z_m = (
        _meters(c)
        for c in calibration_model.get_parameter_value_with_unit("array_element_position_ground")
    )
    tower_m = _meters(calibration_model.get_parameter_value_with_unit("illuminator_tower_height"))
    return np.array([x_m, y_m, z_m + tower_m], dtype=float)


def _axes_offset_position(telescope_m, offsets_m, illuminator_m):
    """Return the telescope position corrected for the azimuth and altitude axes offsets."""
    delta = np.asarray(illuminator_m, dtype=float) - telescope_m
    phi = np.arctan2(delta[0], delta[1])  # From North (y-axis) towards East (x-axis).
    elevation = np.arctan2(delta[2], np.hypot(delta[0], delta[1]))
    pointing = np.array(
        [
            np.cos(elevation) * np.sin(phi),
            np.cos(elevation) * np.cos(phi),
            np.sin(elevation),
        ]
    )
    horizontal = np.array([np.sin(phi), np.cos(phi), 0.0])
    altitude_axis = np.array([np.cos(phi), -np.sin(phi), 0.0])
    perpendicular = np.cross(altitude_axis, pointing)
    perpendicular /= np.linalg.norm(perpendicular)
    return telescope_m + offsets_m[0] * horizontal + offsets_m[1] * perpendicular


def telescope_position_m(telescope_model, illuminator_m, config):
    """
    Return the telescope position in the ground frame [m], corrected for the axes offsets.

    The telescope is at the origin for an illuminator at a configured position with the
    downward beam (artificial setup).
    """
    if config.get("light_source_position") is not None and not uses_telescope_position(
        config.get("light_source_pointing")
    ):
        return np.zeros(3)
    telescope = np.array(
        [
            _meters(c)
            for c in telescope_model.get_parameter_value_with_unit("array_element_position_ground")
        ],
        dtype=float,
    )
    offsets = telescope_model.get_parameter_value_with_unit("axes_offsets")
    offsets_m = [_meters(offset) for offset in value_conversion.ensure_list(offsets)]
    offsets_m = [*offsets_m, 0.0, 0.0][:2]
    if np.allclose(offsets_m, 0.0):
        return telescope
    return _axes_offset_position(telescope, offsets_m, illuminator_m)


def pointing_towards_telescope(illuminator_m, telescope_m):
    """
    Return the illuminator pointing to the telescope and the telescope pointing angles.

    Returns
    -------
    tuple
        Unit vector from the illuminator to the telescope (rounded to 1e-6), and the
        telescope zenith and azimuth angle [deg] when pointing to the illuminator.
    """
    direction = np.asarray(telescope_m, dtype=float) - np.asarray(illuminator_m, dtype=float)
    norm = np.linalg.norm(direction)
    if np.isclose(norm, 0.0):
        raise ValueError("The illuminator is at the telescope position.")
    zenith = 180.0 - np.round(np.rad2deg(np.arccos(direction[2] / norm)), 6)
    azimuth = 180.0 - np.round(np.rad2deg(np.arctan2(direction[1], direction[0])), 6)
    return np.round(direction / norm, 6), (float(zenith), float(azimuth))


def _illuminator_geometry(calibration_model, telescope_model, config):
    """Return position, pointing, telescope position and telescope pointing (ground frame)."""
    position = illuminator_position_m(calibration_model, config)
    telescope = telescope_position_m(telescope_model, position, config)
    pointing, angles = pointing_towards_telescope(position, telescope)
    configured = config.get("light_source_pointing")
    if configured is not None:
        pointing = np.asarray(configured, dtype=float)
    if config.get("light_source_position") is not None:
        angles = (0.0, 0.0)  # Fixed (vertical) telescope pointing.
    return position, pointing, telescope, angles


def illuminator_request(calibration_model, telescope_model, site_model, config, seed=1):
    """
    Return the request of an illuminator (external light source) and a telescope.

    Parameters
    ----------
    calibration_model : CalibrationModel
        Illuminator model.
    telescope_model : TelescopeModel
        Telescope model.
    site_model : SiteModel
        Site model.
    config : dict
        Settings of the light-emission run (``light_source_position`` in m and
        ``light_source_pointing`` override the model).
    seed : int
        Random seed.

    Returns
    -------
    IlluminatorRequest
        Illuminator request.
    """
    position, pointing, telescope, angles = _illuminator_geometry(
        calibration_model, telescope_model, config
    )
    events, photons = _photon_settings(calibration_model, config)
    if len(set(events)) != 1:
        raise ValueError("An illuminator takes one number of events for all photon levels.")
    return IlluminatorRequest(
        position_m=tuple(float(c) for c in position),
        pointing=tuple(float(c) for c in pointing),
        telescope_position_m=tuple(float(c) for c in telescope),
        telescope_radius_m=_meters(
            telescope_model.get_parameter_value_with_unit("telescope_sphere_radius")
        ),
        telescope_pointing_deg=angles,
        events=events[0],
        photons=photons,
        seed=int(seed),
        bunch_size=float(calibration_model.get_parameter_value("flasher_bunch_size") or 1),
        **_common_settings(calibration_model, site_model, config),
    )


# ---------------------------------------------------------------------------
# Command tokens
# ---------------------------------------------------------------------------


def _number(value):
    """Return a number as command text."""
    return format(float(value), ".10g")


def _numbers(values, scale=1.0):
    """Return comma-separated numbers as command text."""
    return ",".join(_number(float(value) * scale) for value in values)


def _write_table(path, columns, header):
    """Write a two-column table (comment header, one row per node) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack(columns), header=header, fmt="%.10g")
    return path


def illuminator_angular_token(angular, work_directory, file_token=""):
    """Return the angular distribution as point-source option value (a table when needed)."""
    shape = angular.shape.lower()
    if shape == "isotropic":
        return "isotropic"
    if shape in ("gauss", "filledcone"):
        return f"{'Gauss' if shape == 'gauss' else 'FilledCone'}:{_number(angular.width_deg)}"
    angles, values = angular_profile(angular.shape, angular.width_deg)
    path = Path(work_directory) / f"light_source_angular_distribution{file_token}.dat"
    return str(_write_table(path, (angles, values), "angle [deg], relative intensity"))


def illuminator_pulse_token(pulse, work_directory, file_token=""):
    """Return the pulse shape as point-source option value (a table when needed)."""
    shape = pulse.shape.lower()
    if shape == "gauss":
        return f"Gauss:{_number(pulse.width_ns)}"
    if shape == "tophat":
        return f"Simple:{_number(pulse.width_ns)}"
    if shape == "exponential":
        return f"Exp:{_number(pulse.decay_ns)}"
    profile = pulse_profile(pulse.shape, pulse.width_ns, pulse.decay_ns)
    if profile is None:
        return "Simple:0"
    path = Path(work_directory) / f"light_source_pulse_shape{file_token}.dat"
    return str(_write_table(path, profile, "time [ns], relative rate"))


def _output_tokens(output):
    """Return the output option (none: the program writes to stdout)."""
    return [] if output is None else ["-o", str(output)]


class FlasherSource:
    """Flasher traced in process through the telescope model (shadowing by its structure)."""

    name = "flasher"
    light_source_types = ("flat_fielding",)
    build_request = staticmethod(flasher_request)

    @staticmethod
    def command(request, work_directory, output=None, file_token="", executable=None):
        """
        Return the command tokens that make the photons of a flasher request.

        The command runs ``simtools.photon_chain.flasher_stream``, which traces the light of
        the flasher to the camera plane (obscurers of the telescope tested, mirrors bypassed).

        Parameters
        ----------
        request : FlasherRequest
            Flasher request; it is written to ``work_directory``.
        work_directory : str or Path
            Directory of the request file and the exported model files.
        output : str or Path, optional
            Output file; without it the program writes the photon stream to stdout.
        file_token : str
            Suffix of the request file name.
        executable : str, optional
            Python interpreter (default: the running one).

        Returns
        -------
        list of str
            Command tokens.
        """
        request_file = request_to_json(
            request, Path(work_directory) / f"light_source_request{file_token}.json"
        )
        return [
            executable or sys.executable,
            "-m",
            FLASHER_MODULE,
            "--request",
            str(request_file),
            "--output-path",
            str(work_directory),
            *(["--output", str(output)] if output is not None else []),
        ]


class IlluminatorSource:
    """Illuminator (external light source) made with the obdeect point-source program."""

    name = "illuminator"
    light_source_types = ("illuminator",)
    build_request = staticmethod(illuminator_request)

    @staticmethod
    def command(request, work_directory, output=None, file_token="", executable=None):
        """
        Return the command tokens that make the photons of an illuminator request.

        The telescope position file (``x y z r`` in cm) and the tables of a profile without
        a program token are written to ``work_directory``.

        Parameters
        ----------
        request : IlluminatorRequest
            Illuminator request.
        work_directory : str or Path
            Directory for the telescope position file and tables.
        output : str or Path, optional
            Output file; without it the program writes the photon stream to stdout.
        file_token : str
            Suffix of the table file names.
        executable : str, optional
            Program.

        Returns
        -------
        list of str
            Command tokens.
        """
        telescope_file = Path(work_directory) / f"light_source_telescope_position{file_token}.dat"
        telescope_file.parent.mkdir(parents=True, exist_ok=True)
        telescope_file.write_text(
            f"{_numbers(request.telescope_position_m, 100.0).replace(',', ' ')} "
            f"{_number(request.telescope_radius_m * 100.0)}\n",
            encoding="utf-8",
        )
        return [
            executable or ILLUMINATOR_EXECUTABLE,
            "-R",
            str(request.run_number),
            "-N",
            str(request.events),
            "-x",
            _number(request.position_m[0] * 100.0),
            "-y",
            _number(request.position_m[1] * 100.0),
            "-z",
            _number(request.position_m[2] * 100.0),
            "-d",
            _numbers(request.pointing),
            "-n",
            ",".join(str(n) for n in request.photons),
            "-b",
            _number(request.bunch_size),
            "-s",
            _number(request.wavelength_nm),
            "-p",
            illuminator_pulse_token(request.pulse, work_directory, file_token),
            "-a",
            illuminator_angular_token(request.angular, work_directory, file_token),
            "-h",
            _number(request.observation_level_m),
            "--telpos-file",
            str(telescope_file),
            "--seed",
            str(request.seed),
            *_output_tokens(output),
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _lazy_source_type(module, class_name):
    """Return a factory that imports a source-type class when it is first used."""

    def factory():
        return getattr(import_module(module), class_name)

    return factory


SOURCE_TYPES = {
    "flasher": _lazy_source_type(__name__, "FlasherSource"),
    "illuminator": _lazy_source_type(__name__, "IlluminatorSource"),
}


def register_source_type(name, factory):
    """
    Register a source type under a name, replacing any type registered under it.

    Parameters
    ----------
    name : str
        Source-type name.
    factory : callable
        Function without arguments returning the source-type class.
    """
    SOURCE_TYPES[name] = factory


def get_source_type(name):
    """Return the source-type class registered under a name."""
    try:
        return SOURCE_TYPES[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown light source type {name!r}; known: {sorted(SOURCE_TYPES)}"
        ) from exc


def source_type_of_model(light_source_type):
    """Return the source-type class that handles a ``light_source_type`` model value."""
    for name in SOURCE_TYPES:
        source_type = get_source_type(name)
        if light_source_type in source_type.light_source_types:
            return source_type
    raise ValueError(
        f"No light source type handles light_source_type {light_source_type!r}; "
        f"known: {sorted(SOURCE_TYPES)}"
    )
