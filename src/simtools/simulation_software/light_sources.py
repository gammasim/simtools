"""Build flasher and illuminator requests from calibration and telescope models."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.utils import value_conversion
from simtools.utils.geometry import fiducial_radius_from_shape

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


def build_request(name, calibration_model, telescope_model, site_model, config, seed=1):
    """Build the request for a flasher or illuminator."""
    builders = {"flasher": flasher_request, "illuminator": illuminator_request}
    try:
        builder = builders[name]
    except KeyError as exc:
        raise ValueError(f"Unknown light source type {name!r}; known: {sorted(builders)}") from exc
    return builder(calibration_model, telescope_model, site_model, config, seed)


def source_name_for_model(light_source_type):
    """Return the source name for a model ``light_source_type`` value."""
    for name, model_types in (("flasher", ("flat_fielding",)), ("illuminator", ("illuminator",))):
        if light_source_type in model_types:
            return name
    raise ValueError(
        f"No light source handles light_source_type {light_source_type!r}; "
        "known: ['flasher', 'illuminator']"
    )


def source_command(name, request, work_directory, output=None, file_token="", executable=None):
    """Return command tokens for a source request."""
    from simtools.simulation_software.obdeect import sources  # pylint: disable=C0415

    commands = {"flasher": sources.flasher_command, "illuminator": sources.illuminator_command}
    try:
        command = commands[name]
    except KeyError as exc:
        raise ValueError(f"Unknown light source type {name!r}; known: {sorted(commands)}") from exc
    return command(request, work_directory, output, file_token, executable)
