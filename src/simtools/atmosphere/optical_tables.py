"""Atmospheric optical tables for the photon-propagation code obdeect.

Radiative-transfer backends (6S, MODTRAN, ...) produce
:class:`AtmosphericOpticalProperties` on a (wavelength, altitude) grid. The
writers convert them to obdeect input tables; the ECSV exchange format keeps
them losslessly. Altitudes are a.s.l.; optical depths are vertical, measured
from the altitude up to the top of the atmosphere.
"""

import abc
import copy
import importlib
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

OBDEECT_MAX_WAVELENGTH_NM = 1000
OBDEECT_MAX_HEIGHTS = 99
OBDEECT_ANGLE_RANGE_DEG = (0.0, 180.0)
ECSV_FORMAT = "ascii.ecsv"
ECSV_METADATA_KEY = "metadata"
ECSV_DEPOLARIZATION_KEY = "depolarization"
ECSV_PHASE_ANGLE_KEY = "phase_angle_deg"
PHASE_COLUMN = "aerosol_phase"
SIXS_BACKEND_NAME = "6s"
SIXS_BACKEND_MODULE = "simtools.atmosphere.sixs_backend"
ANGLE_TOLERANCE_DEG = 1.0e-9


class AtmosphericOpticalProperties:
    """
    Vertical optical properties of the atmosphere on a (wavelength, altitude) grid.

    Parameters
    ----------
    wavelength: astropy.units.Quantity
        Strictly ascending wavelengths, shape (n_wl,), at least two values.
    altitude: astropy.units.Quantity
        Strictly ascending positive altitudes a.s.l., shape (n_h,), at least two values.
    tau_molecular, tau_aerosol, tau_absorption: array
        Vertical optical depths from the altitude to the top, shape (n_wl, n_h).
        Non-negative and non-increasing with altitude.
    aerosol_ssa: array
        Aerosol single-scattering albedo in [0, 1], shape (n_wl, n_h).
    aerosol_asymmetry: array
        Aerosol asymmetry parameter in (-1, 1), shape (n_wl, n_h).
    phase_angle: astropy.units.Quantity, optional
        Scattering angles in [0, 180] deg, ascending, shape (n_angle,).
    aerosol_phase: array, optional
        Aerosol phase function in 1/sr, shape (n_wl, n_h, n_angle). Given together with
        ``phase_angle`` or not at all.
    depolarization: float
        Rayleigh depolarization ratio in [0, 1].
    metadata: dict, optional
        Backend name, version and settings; written as comments to text tables.
    """

    def __init__(
        self,
        wavelength,
        altitude,
        tau_molecular,
        tau_aerosol,
        tau_absorption,
        aerosol_ssa,
        aerosol_asymmetry,
        phase_angle=None,
        aerosol_phase=None,
        depolarization=0.0,
        metadata=None,
    ):
        self.wavelength = _grid("wavelength", wavelength, u.nm)
        self.altitude = _grid("altitude", altitude, u.m)
        shape = (self.wavelength.size, self.altitude.size)
        self.tau_molecular = _array("tau_molecular", tau_molecular, shape)
        self.tau_aerosol = _array("tau_aerosol", tau_aerosol, shape)
        self.tau_absorption = _array("tau_absorption", tau_absorption, shape)
        self.aerosol_ssa = _array("aerosol_ssa", aerosol_ssa, shape)
        self.aerosol_asymmetry = _array("aerosol_asymmetry", aerosol_asymmetry, shape)
        for name, tau in (
            ("tau_molecular", self.tau_molecular),
            ("tau_aerosol", self.tau_aerosol),
            ("tau_absorption", self.tau_absorption),
        ):
            _check_optical_depth(name, tau)
        _check_ranges(self.aerosol_ssa, self.aerosol_asymmetry)
        self.phase_angle, self.aerosol_phase = _phase(
            phase_angle, aerosol_phase, self.wavelength.size, self.altitude.size
        )
        self.depolarization = _depolarization(depolarization)
        self.metadata = dict(metadata or {})

    def total_optical_depth(self):
        """Total vertical optical depth (molecular + aerosol + absorption), shape (n_wl, n_h)."""
        return self.tau_molecular + self.tau_aerosol + self.tau_absorption

    def interpolate_wavelength(self, wavelength):
        """
        Return new properties at the requested wavelengths, linear in wavelength.

        Raises ValueError outside the tabulated wavelength range (no extrapolation).
        """
        return _resampled(self, 0, u.Quantity(wavelength, u.nm))

    def with_depolarization(self, depolarization):
        """
        Return a copy with the Rayleigh depolarization ratio set (0 = none).

        The original object is not changed. Arrays and metadata are shared with the original.
        """
        copied = copy.copy(self)
        copied.depolarization = _depolarization(depolarization)
        return copied

    def vertical_transmission(self, observation_level):
        """
        Vertical transmission from the top of the altitude grid down to observation_level.

        Returns exp(-depth) per wavelength, shape (n_wl,). The depth is the difference of the
        total optical depth at the observation level and at the top of the grid, with the
        total interpolated linearly in log(altitude). The observation level must lie within
        the altitude grid.
        """
        level_m = u.Quantity(observation_level, u.m).value
        total = self.total_optical_depth()
        index, weight = _interpolation_weights(
            self.altitude.value, np.array([level_m]), log_axis=True
        )
        at_level = _lerp(total, 1, index, weight)[:, 0]
        return np.exp(-(at_level - total[:, -1]))


def write_obdeect_transmission_table(properties, file_name, observation_level, heights=None):
    """
    Write an obdeect transmission table (``# H2=`` and ``H1=`` header).

    One row per integer wavelength from ceil(first) to floor(last) wavelength, below
    1000 nm. Each row gives the optical depth between the observation level H2 and
    each height H1: depth(h) = total(H2) - total(h), with total interpolated linearly
    in log(altitude). Heights default to the table altitudes above the observation
    level; at most 99 heights are allowed.

    Parameters
    ----------
    properties: AtmosphericOpticalProperties
        Optical properties on the source grid.
    file_name: str or pathlib.Path
        Output file.
    observation_level: astropy.units.Quantity
        Observation level (H2), a.s.l.
    heights: astropy.units.Quantity, optional
        Heights (H1) above the observation level, strictly ascending.
    """
    level_m = u.Quantity(observation_level, u.m).value
    grid = _integer_wavelength_grid(properties.wavelength)
    if grid[-1] >= OBDEECT_MAX_WAVELENGTH_NM:
        raise ValueError(
            f"obdeect transmission tables are limited to wavelengths below "
            f"{OBDEECT_MAX_WAVELENGTH_NM} nm, got {grid[-1]} nm"
        )
    heights_m = _transmission_heights(properties, level_m, heights)
    total = properties.interpolate_wavelength(grid * u.nm).total_optical_depth()
    index, weight = _interpolation_weights(
        properties.altitude.value, np.concatenate(([level_m], heights_m)), log_axis=True
    )
    at_levels = _lerp(total, 1, index, weight)
    depth = at_levels[:, :1] - at_levels[:, 1:]
    with Path(file_name).open("w", encoding="utf-8") as file:
        _write_metadata(file, properties.metadata)
        file.write(f"# H2= {level_m / 1000:.3f}, H1=")
        file.write("".join(f"{height / 1000:10.3f}" for height in heights_m) + "\n")
        for wavelength, row in zip(grid, depth, strict=True):
            file.write(f"{wavelength:d}" + "".join(f"{value:12.6f}" for value in row) + "\n")


def write_obdeect_scattering_table(properties, file_name):
    """
    Write an obdeect scattering table (one row per integer wavelength and altitude).

    Columns: ``wavelength_nm altitude_km tau_mol tau_aer ssa g``. If present, the aerosol
    phase function follows as ``phase <wavelength_nm> <altitude_km> p_1 ... p_K`` lines,
    preceded by an ``angles_deg`` line (ascending, 0 to 180). A non-zero depolarization
    is written as ``depolarization <rho>``.

    Parameters
    ----------
    properties: AtmosphericOpticalProperties
        Optical properties on the source grid.
    file_name: str or pathlib.Path
        Output file.
    """
    grid = _integer_wavelength_grid(properties.wavelength)
    resolved = properties.interpolate_wavelength(grid * u.nm)
    if resolved.depolarization >= 0.5:
        raise ValueError("obdeect requires depolarization below 0.5")
    if resolved.aerosol_phase is not None:
        _check_angle_span(resolved.phase_angle)
        if np.any(resolved.aerosol_phase <= 0):
            raise ValueError("obdeect phase values must be positive")
        mu = np.cos(np.radians(resolved.phase_angle.value))[::-1]
        norm = 2 * np.pi * np.trapezoid(resolved.aerosol_phase[..., ::-1], mu, axis=-1)
        if np.any(np.abs(norm - 1) > 0.05):
            raise ValueError("obdeect phase normalization must be within 5% of 1")
    with Path(file_name).open("w", encoding="utf-8") as file:
        _write_metadata(file, properties.metadata)
        file.write("# wavelength_nm altitude_km tau_mol tau_aer ssa g\n")
        if resolved.depolarization > 0:
            file.write(f"depolarization {resolved.depolarization:.8g}\n")
        if resolved.aerosol_phase is not None:
            angles = " ".join(f"{angle:.8g}" for angle in resolved.phase_angle.value)
            file.write(f"angles_deg {angles}\n")
        for line in _scattering_lines(resolved):
            file.write(line + "\n")


def write_optical_properties(properties, file_name):
    """
    Write optical properties to an ECSV file (lossless, one row per wavelength and altitude).

    Rows are ordered by wavelength, then altitude. Depolarization, phase angles and
    metadata are stored in the table metadata.
    """
    n_wl, n_h = properties.tau_molecular.shape
    columns = {
        "wavelength": u.Quantity(np.repeat(properties.wavelength.value, n_h), u.nm),
        "altitude": u.Quantity(np.tile(properties.altitude.value, n_wl), u.m),
        "tau_molecular": properties.tau_molecular.reshape(-1),
        "tau_aerosol": properties.tau_aerosol.reshape(-1),
        "tau_absorption": properties.tau_absorption.reshape(-1),
        "aerosol_ssa": properties.aerosol_ssa.reshape(-1),
        "aerosol_asymmetry": properties.aerosol_asymmetry.reshape(-1),
    }
    table = Table(columns)
    table.meta[ECSV_METADATA_KEY] = properties.metadata
    table.meta[ECSV_DEPOLARIZATION_KEY] = properties.depolarization
    if properties.aerosol_phase is not None:
        n_angle = properties.phase_angle.size
        table[PHASE_COLUMN] = u.Quantity(
            properties.aerosol_phase.reshape(n_wl * n_h, n_angle), 1 / u.sr
        )
        table.meta[ECSV_PHASE_ANGLE_KEY] = properties.phase_angle.to_value(u.deg).tolist()
    table.write(file_name, format=ECSV_FORMAT, overwrite=True)


def read_optical_properties(file_name):
    """Read optical properties written by :func:`write_optical_properties`."""
    table = Table.read(file_name, format=ECSV_FORMAT)
    wavelength_values = np.asarray(table["wavelength"])
    n_wl = np.unique(wavelength_values).size
    if len(table) % n_wl != 0:
        raise ValueError(f"ECSV table {file_name} is not a complete (wavelength, altitude) grid")
    n_h = len(table) // n_wl
    shape = (n_wl, n_h)
    wavelengths = wavelength_values.reshape(shape)
    altitudes = np.asarray(table["altitude"]).reshape(shape)
    if not (np.all(wavelengths == wavelengths[:, :1]) and np.all(altitudes == altitudes[:1, :])):
        raise ValueError(f"ECSV table {file_name} is not a complete ordered grid")

    def column(name):
        return np.asarray(table[name]).reshape(shape)

    phase_angle = None
    aerosol_phase = None
    if ECSV_PHASE_ANGLE_KEY in table.meta:
        phase_angle = u.Quantity(table.meta[ECSV_PHASE_ANGLE_KEY], u.deg)
        aerosol_phase = np.asarray(table[PHASE_COLUMN]).reshape(n_wl, n_h, phase_angle.size)
    return AtmosphericOpticalProperties(
        wavelength=u.Quantity(table["wavelength"][::n_h], u.nm),
        altitude=u.Quantity(table["altitude"][:n_h], u.m),
        tau_molecular=column("tau_molecular"),
        tau_aerosol=column("tau_aerosol"),
        tau_absorption=column("tau_absorption"),
        aerosol_ssa=column("aerosol_ssa"),
        aerosol_asymmetry=column("aerosol_asymmetry"),
        phase_angle=phase_angle,
        aerosol_phase=aerosol_phase,
        depolarization=table.meta.get(ECSV_DEPOLARIZATION_KEY, 0.0),
        metadata=table.meta.get(ECSV_METADATA_KEY, {}),
    )


class RadiativeTransferBackend(abc.ABC):
    """Base class of radiative-transfer backends that produce optical properties."""

    name = None

    @abc.abstractmethod
    def compute(self, wavelength, altitude):
        """Return :class:`AtmosphericOpticalProperties` on the requested grid."""


class EcsvBackend(RadiativeTransferBackend):
    """Optical properties from an ECSV exchange file (e.g. converted MODTRAN output)."""

    name = "ecsv"

    def __init__(self, file_name):
        self.file_name = Path(file_name)

    def compute(self, wavelength, altitude):
        """Read the file and interpolate to the requested wavelengths and altitudes."""
        properties = read_optical_properties(self.file_name)
        resolved = properties.interpolate_wavelength(wavelength)
        return _resampled(resolved, 1, u.Quantity(altitude, u.m))


_BACKENDS = {EcsvBackend.name: EcsvBackend}


def register_backend(name, factory):
    """Register a backend factory under ``name``, replacing any earlier registration."""
    _BACKENDS[name] = factory


def get_backend(name, **kwargs):
    """
    Return a backend instance by name.

    The 6S backend module is imported on first use and registers itself.
    Raises ValueError for unknown names.
    """
    if name == SIXS_BACKEND_NAME and name not in _BACKENDS:
        importlib.import_module(SIXS_BACKEND_MODULE)
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown atmospheric backend '{name}'. Known backends: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name](**kwargs)


def _grid(name, values, unit):
    """Return a strictly ascending, positive, one-dimensional grid as a Quantity."""
    grid = _ascending(name, values, unit)
    if grid.value[0] <= 0:
        raise ValueError(f"{name} must be positive, got {grid[0]}")
    return grid


def _ascending(name, values, unit):
    """Return a strictly ascending, non-empty, one-dimensional grid."""
    grid = u.Quantity(values, unit)
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError(f"{name} must be a non-empty 1-D grid")
    if np.any(np.diff(grid.value) <= 0):
        raise ValueError(f"{name} must be strictly ascending")
    return grid


def _array(name, values, shape):
    """Return a finite float array of the expected shape."""
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _depolarization(value):
    """Rayleigh depolarization ratio in [0, 1] as float."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"depolarization must be in [0, 1], got {value}")
    return float(value)


def _check_optical_depth(name, tau):
    """Optical depths must be non-negative and non-increasing with altitude."""
    if np.any(tau < 0) or np.any(np.diff(tau, axis=1) > 0):
        raise ValueError(f"{name} must be non-negative and non-increasing with altitude")


def _check_ranges(ssa, asymmetry):
    """Check aerosol single-scattering albedo in [0, 1] and asymmetry in (-1, 1)."""
    if np.any((ssa < 0) | (ssa > 1)):
        raise ValueError("aerosol_ssa must be in [0, 1]")
    if np.any(np.abs(asymmetry) >= 1):
        raise ValueError("aerosol_asymmetry must be in (-1, 1)")


def _phase(phase_angle, aerosol_phase, n_wl, n_h):
    """Validate the optional phase function; both inputs or neither."""
    if (phase_angle is None) != (aerosol_phase is None):
        raise ValueError("phase_angle and aerosol_phase must be given together")
    if phase_angle is None:
        return None, None
    angle = _ascending("phase_angle", phase_angle, u.deg)
    low, high = OBDEECT_ANGLE_RANGE_DEG
    if angle.value[0] < low or angle.value[-1] > high:
        raise ValueError(f"phase_angle must lie within [{low}, {high}] deg")
    phase = _array("aerosol_phase", aerosol_phase, (n_wl, n_h, angle.size))
    if np.any(phase < 0):
        raise ValueError("aerosol_phase must be non-negative")
    return angle, phase


def _integer_wavelength_grid(wavelength):
    """Integer nm from ceil(first) to floor(last) wavelength."""
    first = int(np.ceil(wavelength.value[0]))
    last = int(np.floor(wavelength.value[-1]))
    if first > last:
        raise ValueError(f"No integer wavelength in the range {wavelength[0]} to {wavelength[-1]}")
    return np.arange(first, last + 1)


def _transmission_heights(properties, level_m, heights):
    """Return the H1 heights in m: explicit, or the table altitudes above the level."""
    if heights is None:
        heights_m = properties.altitude.value[properties.altitude.value > level_m]
    else:
        heights_m = u.Quantity(heights, u.m).value
    if heights_m.size == 0 or heights_m.size > OBDEECT_MAX_HEIGHTS:
        raise ValueError(
            f"Transmission table needs 1 to {OBDEECT_MAX_HEIGHTS} heights above the "
            f"observation level, got {heights_m.size}"
        )
    if np.any(heights_m <= level_m) or np.any(np.diff(heights_m) <= 0):
        raise ValueError(
            "Transmission table heights must be above the observation level and strictly ascending"
        )
    return heights_m


def _interpolation_weights(grid, targets, log_axis):
    """Lower grid index and fractional weight for linear interpolation of targets."""
    if grid.size < 2:
        raise ValueError("Interpolation needs at least two tabulated values")
    inside = (targets >= grid[0]) & (targets <= grid[-1])
    if not np.all(inside):
        raise ValueError(
            f"Values {targets[~inside]} lie outside the tabulated range "
            f"[{grid[0]}, {grid[-1]}]; extrapolation is not supported"
        )
    if log_axis:
        grid, targets = np.log(grid), np.log(targets)
    index = np.clip(np.searchsorted(grid, targets, side="right") - 1, 0, grid.size - 2)
    weight = (targets - grid[index]) / (grid[index + 1] - grid[index])
    return index, weight


def _lerp(values, axis, index, weight):
    """Linear interpolation along ``axis`` between the nodes ``index`` and ``index + 1``."""
    lower = np.take(values, index, axis=axis)
    upper = np.take(values, index + 1, axis=axis)
    shape = [1] * values.ndim
    shape[axis] = -1
    return lower + weight.reshape(shape) * (upper - lower)


def _resampled(properties, axis, targets):
    """
    Return properties at new grid points along one axis.

    Axis 0 is wavelength (linear); axis 1 is altitude (linear in log altitude).
    """
    if axis == 0:
        grid, log_axis = properties.wavelength.value, False
    else:
        grid, log_axis = properties.altitude.value, True
    index, weight = _interpolation_weights(grid, targets.value, log_axis)

    def lerp(values):
        return None if values is None else _lerp(values, axis, index, weight)

    return AtmosphericOpticalProperties(
        wavelength=targets if axis == 0 else properties.wavelength,
        altitude=properties.altitude if axis == 0 else targets,
        tau_molecular=lerp(properties.tau_molecular),
        tau_aerosol=lerp(properties.tau_aerosol),
        tau_absorption=lerp(properties.tau_absorption),
        aerosol_ssa=lerp(properties.aerosol_ssa),
        aerosol_asymmetry=lerp(properties.aerosol_asymmetry),
        phase_angle=properties.phase_angle,
        aerosol_phase=lerp(properties.aerosol_phase),
        depolarization=properties.depolarization,
        metadata=properties.metadata,
    )


def _check_angle_span(angles):
    """Obdeect phase tables cover the full angular range 0 to 180 deg."""
    low, high = OBDEECT_ANGLE_RANGE_DEG
    values = angles.to_value(u.deg)
    if abs(values[0] - low) > ANGLE_TOLERANCE_DEG or abs(values[-1] - high) > ANGLE_TOLERANCE_DEG:
        raise ValueError(f"obdeect phase functions need angles from {low} to {high} deg")


def _scattering_lines(properties):
    """Yield the scattering-table rows, then the phase-function rows if present."""
    altitude_km = properties.altitude.to_value(u.km)
    wavelength_nm = properties.wavelength.value
    for i, j in np.ndindex(properties.tau_molecular.shape):
        yield (
            f"{wavelength_nm[i]:.0f} {altitude_km[j]:.8g} "
            f"{properties.tau_molecular[i, j]:.8e} {properties.tau_aerosol[i, j]:.8e} "
            f"{properties.aerosol_ssa[i, j]:.8g} {properties.aerosol_asymmetry[i, j]:.8g}"
        )
    if properties.aerosol_phase is None:
        return
    for i, j in np.ndindex(properties.tau_molecular.shape):
        values = " ".join(f"{value:.8e}" for value in properties.aerosol_phase[i, j])
        yield f"phase {wavelength_nm[i]:.0f} {altitude_km[j]:.8g} {values}"


def _write_metadata(file, metadata):
    """Write metadata as comment lines."""
    for key, value in metadata.items():
        file.write(f"# {key} = {value}\n")
