"""6S radiative transfer backend for atmospheric optical properties.

Runs the 6S (6SV1.1) radiative transfer code through Py6S. Py6S and the 6S
executable are optional: Py6S is imported when a computation starts.

Per wavelength and target altitude, one 6S run at nadir with the sun at zenith gives
the Rayleigh and aerosol vertical optical depths, the aerosol single-scattering albedo
and the upward total gaseous transmittance. 6S reports the aerosol phase function only
at the scattering angle of a run. For each wavelength, runs at geometries with the
requested scattering angles sample the phase function. The phase function is assumed
independent of altitude (one predefined aerosol profile), so it is sampled at sea level.

Scattering-angle convention (6S, checked against a 6S output with solar zenith 32 deg,
view zenith 23 deg and azimuth difference 74 deg, which reports 146.90 deg)::

    cos(scattering angle) = -(cos(sun zenith) cos(view zenith)
                              + sin(sun zenith) sin(view zenith) cos(azimuth difference))

With an azimuth difference of 180 deg the scattering angle is 180 deg - (sun zenith +
view zenith). Sun and view zenith are therefore both (180 deg - scattering angle) / 2.
Zenith angles are restricted to below 89 deg, so scattering angles up to 2 deg are not
reachable. Unreachable angles below the smallest reachable angle are extended with a
Henyey-Greenstein function (see ``_complete_phase``).
"""

import concurrent.futures
import importlib
from collections import namedtuple

import astropy.units as u
import numpy as np

from simtools.atmosphere.optical_tables import (
    AtmosphericOpticalProperties,
    RadiativeTransferBackend,
    register_backend,
)

BACKEND_NAME = "6s"
WAVELENGTH_RANGE_UM = (0.25, 4.0)
ALTITUDE_RANGE_KM = (0.0, 100.0)
DEFAULT_PHASE_ANGLE_DEG = np.arange(2.0, 181.0, 1.0)
MAX_ZENITH_DEG = 89.0
SCATTERING_ANGLE_TOLERANCE_DEG = 0.05
MAX_ASYMMETRY = 0.99
OPTICAL_KEYS = ("tau_molecular", "tau_aerosol", "tau_absorption", "aerosol_ssa")

_Settings = namedtuple(
    "_Settings", "atmosphere_profile aerosol_profile aot550 ozone water sixs_path"
)
_Run = namedtuple(
    "_Run",
    "wavelength_um altitude_km solar_zenith_deg view_zenith_deg azimuth_difference_deg",
)


class SixSBackend(RadiativeTransferBackend):
    """
    Atmospheric optical properties from 6S runs.

    Parameters
    ----------
    atmosphere_profile: str
        Predefined Py6S atmosphere profile name (for example 'MidlatitudeSummer').
        Ignored when ``ozone`` and ``water`` are given.
    aerosol_profile: str
        Predefined Py6S aerosol profile name (for example 'Continental').
    aot550: float
        Aerosol optical thickness at 550 nm.
    ozone: float, optional
        Ozone content in cm-atm. Requires ``water``.
    water: float, optional
        Water vapour content in g/cm^2. Requires ``ozone``.
    phase_angle: array-like or astropy.units.Quantity, optional
        Scattering angles in deg, within [0, 180]. Default 2 to 180 deg in 1 deg steps.
        0 and 180 deg are always included.
    sixs_path: str, optional
        Path to the 6S executable. Default: searched on PATH by Py6S.
    processes: int
        Number of worker processes for the 6S runs. Results do not depend on it.
    """

    name = BACKEND_NAME

    def __init__(
        self,
        atmosphere_profile="MidlatitudeSummer",
        aerosol_profile="Continental",
        aot550=0.1,
        ozone=None,
        water=None,
        phase_angle=None,
        sixs_path=None,
        processes=1,
    ):
        if (ozone is None) != (water is None):
            raise ValueError(
                "Give 'ozone' and 'water' together to use a user atmosphere with "
                "water vapour and ozone."
            )
        if processes < 1:
            raise ValueError(f"'processes' must be at least 1, got {processes}")
        self.atmosphere_profile = atmosphere_profile
        self.aerosol_profile = aerosol_profile
        self.aot550 = float(aot550)
        self.ozone = ozone
        self.water = water
        self.sixs_path = sixs_path
        self.processes = processes
        self.phase_angle_deg = _phase_angle_grid(phase_angle)
        self._settings = _Settings(
            atmosphere_profile=atmosphere_profile,
            aerosol_profile=aerosol_profile,
            aot550=self.aot550,
            ozone=ozone,
            water=water,
            sixs_path=sixs_path,
        )

    def compute(self, wavelength, altitude):
        """
        Compute the optical properties on the (wavelength, altitude) grid.

        Parameters
        ----------
        wavelength: astropy.units.Quantity or float
            Wavelengths in um (plain numbers are in um), 0.25 to 4.0 um.
        altitude: astropy.units.Quantity or float
            Target altitudes a.s.l. in km (plain numbers are in km), above 0 and below 100 km.
            Sea level is given as a small positive altitude, e.g. 1 m.
        """
        wavelength_um = _as_array(wavelength, u.um)
        altitude_km = _as_array(altitude, u.km)
        _check_wavelength(wavelength_um)
        _check_altitude(altitude_km)
        py6s = _load_py6s()
        phase_angles = self.phase_angle_deg[_reachable(self.phase_angle_deg)]
        phase_results = _map_runs(
            _execute_run,
            [
                (self._settings, _phase_run(wl, angle))
                for wl in wavelength_um
                for angle in phase_angles
            ],
            self.processes,
        )
        optical_results = _map_runs(
            _execute_run,
            [(self._settings, _optical_run(wl, h)) for wl in wavelength_um for h in altitude_km],
            self.processes,
        )
        n_wl, n_h = wavelength_um.size, altitude_km.size
        phase, asymmetry = _phase_tables(
            _stack(phase_results, "phase", (n_wl, phase_angles.size)),
            self.phase_angle_deg,
            _reachable(self.phase_angle_deg),
        )
        optical = {key: _stack(optical_results, key, (n_wl, n_h)) for key in OPTICAL_KEYS}
        return AtmosphericOpticalProperties(
            wavelength=u.Quantity(wavelength_um, u.um),
            altitude=u.Quantity(altitude_km, u.km),
            tau_molecular=optical["tau_molecular"],
            tau_aerosol=optical["tau_aerosol"],
            tau_absorption=optical["tau_absorption"],
            aerosol_ssa=optical["aerosol_ssa"],
            aerosol_asymmetry=np.broadcast_to(asymmetry[:, None], (n_wl, n_h)).copy(),
            phase_angle=u.Quantity(self.phase_angle_deg, u.deg),
            aerosol_phase=np.broadcast_to(
                phase[:, None, :], (n_wl, n_h, self.phase_angle_deg.size)
            ).copy(),
            metadata=self._metadata(py6s.SixS.__version__, optical_results[0]["sixs_version"]),
        )

    def _metadata(self, py6s_version, sixs_version):
        """Backend name, versions and settings."""
        return {
            "backend": BACKEND_NAME,
            "py6s_version": py6s_version,
            "sixs_version": sixs_version,
            "atmosphere_profile": self.atmosphere_profile,
            "aerosol_profile": self.aerosol_profile,
            "aot550": self.aot550,
            "ozone": self.ozone,
            "water": self.water,
            "sixs_path": self.sixs_path,
            "processes": self.processes,
            "phase_angle_deg": self.phase_angle_deg.tolist(),
        }


register_backend(BACKEND_NAME, SixSBackend)


def scattering_angle_deg(solar_zenith_deg, view_zenith_deg, azimuth_difference_deg):
    """Scattering angle in deg for the 6S geometry convention (see module docstring)."""
    sun = np.radians(solar_zenith_deg)
    view = np.radians(view_zenith_deg)
    difference = np.radians(azimuth_difference_deg)
    cos_angle = -(np.cos(sun) * np.cos(view) + np.sin(sun) * np.sin(view) * np.cos(difference))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _load_py6s():
    """Import Py6S when needed (optional dependency)."""
    try:
        return importlib.import_module("Py6S")
    except ImportError as exc:
        raise ImportError(
            "The 6S backend needs the optional Py6S package. Install it (pip install Py6S) "
            "and make the 6S executable (6SV1.1, 'sixsV1.1') available on PATH or set "
            "'sixs_path'."
        ) from exc


def _phase_angle_grid(phase_angle):
    """Scattering angles in deg on 0 to 180 deg, including both end points."""
    if phase_angle is None:
        requested = DEFAULT_PHASE_ANGLE_DEG
    else:
        requested = u.Quantity(phase_angle, u.deg).to_value(u.deg)
    requested = np.asarray(requested, dtype=float).ravel()
    if np.any((requested < 0.0) | (requested > 180.0)):
        raise ValueError("phase_angle values must be within [0, 180] deg")
    grid = np.unique(np.concatenate(([0.0, 180.0], requested)))
    if np.count_nonzero(_reachable(grid)) < 2:
        raise ValueError(
            "phase_angle needs at least two angles that 6S can reach "
            f"(above {180.0 - 2.0 * MAX_ZENITH_DEG:.0f} deg scattering angle, i.e. "
            f"zenith angles below {MAX_ZENITH_DEG} deg)"
        )
    return grid


def _reachable(angles_deg):
    """Scattering angles reachable with sun and view zenith below MAX_ZENITH_DEG."""
    return _phase_zenith_deg(angles_deg) < MAX_ZENITH_DEG


def _phase_zenith_deg(scattering_angle_deg_values):
    """Sun and view zenith angle for a scattering angle at an azimuth difference of 180 deg."""
    return (180.0 - np.asarray(scattering_angle_deg_values, dtype=float)) / 2.0


def _as_array(value, unit):
    """Values as a one-dimensional float array in the given unit."""
    values = np.atleast_1d(u.Quantity(value, unit).to_value(unit)).astype(float).ravel()
    if values.size == 0:
        raise ValueError(f"At least one value in {unit} is required")
    return values


def _check_wavelength(wavelength_um):
    """6S wavelength range."""
    low, high = WAVELENGTH_RANGE_UM
    if not np.all((wavelength_um >= low) & (wavelength_um <= high)):
        raise ValueError(f"6S wavelengths must be between {low} and {high} um, got {wavelength_um}")


def _check_altitude(altitude_km):
    """Target altitudes a.s.l. above 0 km and below 100 km."""
    low, high = ALTITUDE_RANGE_KM
    if not np.all((altitude_km > low) & (altitude_km < high)):
        raise ValueError(
            f"Altitudes must be above {low} km and below {high} km, got {altitude_km}. "
            "Give sea level as a small positive altitude, for example 0.001 km (1 m): "
            "obdeect tables interpolate in log(altitude)."
        )


def _optical_run(wavelength_um, altitude_km):
    """Nadir view with the sun at zenith, target at altitude_km."""
    return _Run(wavelength_um, altitude_km, 0.0, 0.0, 0.0)


def _phase_run(wavelength_um, scattering_angle):
    """Sea-level run with the scattering angle given by the zenith angles (see module docstring)."""
    zenith = float(_phase_zenith_deg(scattering_angle))
    return _Run(wavelength_um, 0.0, zenith, zenith, 180.0)


def _predefined(profile_class, name, label):
    """Integer code of a predefined Py6S profile name."""
    code = getattr(profile_class, name, None) if isinstance(name, str) else None
    if not isinstance(code, int):
        raise ValueError(f"Unknown {label} profile {name!r}; use a predefined Py6S profile name.")
    return code


def _atmosphere_profile(py6s, settings):
    """Predefined atmosphere, or user water vapour and ozone when given."""
    if settings.ozone is not None:
        return py6s.AtmosProfile.UserWaterAndOzone(settings.water, settings.ozone)
    code = _predefined(py6s.AtmosProfile, settings.atmosphere_profile, "atmosphere")
    return py6s.AtmosProfile.PredefinedType(code)


def _geometry(py6s, run):
    """Sun at azimuth 0, view at the azimuth difference of the run."""
    geometry = py6s.Geometry.User()
    geometry.solar_z = run.solar_zenith_deg
    geometry.solar_a = 0.0
    geometry.view_z = run.view_zenith_deg
    geometry.view_a = run.azimuth_difference_deg
    return geometry


def _altitudes(py6s, altitude_km):
    """Target at altitude_km, sensor at satellite level."""
    altitudes = py6s.Altitudes()
    altitudes.set_target_custom_altitude(altitude_km)
    altitudes.set_sensor_satellite_level()
    return altitudes


def _execute_run(task):
    """Run 6S for a (settings, run) task and return the values of interest."""
    settings, run = task
    py6s = _load_py6s()
    six_s = py6s.SixS(path=settings.sixs_path)
    if six_s.sixs_path is None:
        raise RuntimeError(
            "6S executable not found. Install 6SV1.1 and put 'sixsV1.1' on PATH, or set "
            "'sixs_path' to its location."
        )
    six_s.atmos_profile = _atmosphere_profile(py6s, settings)
    six_s.aero_profile = py6s.AeroProfile.PredefinedType(
        _predefined(py6s.AeroProfile, settings.aerosol_profile, "aerosol")
    )
    six_s.aot550 = settings.aot550
    six_s.geometry = _geometry(py6s, run)
    six_s.altitudes = _altitudes(py6s, run.altitude_km)
    six_s.wavelength = py6s.Wavelength(run.wavelength_um)
    six_s.run()
    return _run_results(six_s, run)


def _run_results(six_s, run):
    """Values of interest from the 6S outputs (names from Py6S outputs.py)."""
    outputs = six_s.outputs
    _check_scattering_angle(outputs.values.get("scattering_angle", np.nan), run)
    depth = outputs.rat["optical_depth_total"]
    return {
        "tau_molecular": float(depth.rayleigh),
        "tau_aerosol": float(depth.aerosol),
        "tau_absorption": float(-np.log(outputs.trans["global_gas"].upward)),
        "aerosol_ssa": float(outputs.rat["single_scattering_albedo"].aerosol),
        "phase": float(outputs.rat["phase_function_I"].aerosol),
        "sixs_version": outputs.values.get("version"),
    }


def _check_scattering_angle(reported, run):
    """Check that the 6S scattering angle matches the geometry convention used here."""
    expected = scattering_angle_deg(
        run.solar_zenith_deg, run.view_zenith_deg, run.azimuth_difference_deg
    )
    if np.isfinite(reported) and abs(reported - expected) > SCATTERING_ANGLE_TOLERANCE_DEG:
        raise RuntimeError(
            f"6S reports a scattering angle of {reported:.2f} deg for a geometry that "
            f"gives {expected:.2f} deg with the convention of this backend; the 6S "
            "geometry convention does not match."
        )


def _map_runs(function, tasks, processes):
    """Apply function to the tasks in order; a process pool is used for processes > 1."""
    if processes == 1:
        return [function(task) for task in tasks]
    chunksize = max(1, len(tasks) // (4 * processes))
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        return list(executor.map(function, tasks, chunksize=chunksize))


def _stack(results, key, shape):
    """Values of one key from the run results as an array of the given shape."""
    return np.array([result[key] for result in results], dtype=float).reshape(shape)


def _integrate(values, mu):
    """Trapezoidal integral over mu (from -1 to 1) for a grid in any order."""
    order = np.argsort(mu)
    return float(np.trapezoid(values[order], mu[order]))


def _henyey_greenstein(mu, asymmetry):
    """Henyey-Greenstein function of cos(scattering angle), unnormalised in amplitude."""
    return (1.0 - asymmetry**2) / (1.0 + asymmetry**2 - 2.0 * asymmetry * mu) ** 1.5


def _first_moment_asymmetry(values, mu):
    """Asymmetry of the reachable part of the phase function (first moment ratio)."""
    moment = _integrate(values * mu, mu) / _integrate(values, mu)
    return float(np.clip(moment, -MAX_ASYMMETRY, MAX_ASYMMETRY))


def _complete_phase(mu, reachable, values):
    """
    Aerosol phase function on the full angle grid, normalised to 2 pi int p dmu = 1.

    Angles below the smallest reachable angle are extended with a Henyey-Greenstein
    function. Its amplitude is matched in value at the smallest reachable angle; its
    asymmetry is the first moment of the reachable part. The grid is then renormalised
    numerically, so the 6S normalisation does not matter.
    """
    if not np.all(np.isfinite(values)):
        raise RuntimeError("6S returned a non-finite aerosol phase function")
    mu_reachable = mu[reachable]
    asymmetry = _first_moment_asymmetry(values, mu_reachable)
    phase = np.empty(mu.shape)
    phase[reachable] = values
    below = ~reachable
    phase[below] = (
        values[0]
        * _henyey_greenstein(mu[below], asymmetry)
        / _henyey_greenstein(mu_reachable[0], asymmetry)
    )
    return phase / (2.0 * np.pi * _integrate(phase, mu))


def _phase_tables(values, angles_deg, reachable):
    """Normalised phase functions (n_wl, n_angle) and their asymmetry (n_wl,)."""
    mu = np.cos(np.radians(angles_deg))
    phases = np.array([_complete_phase(mu, reachable, row) for row in values])
    asymmetry = np.array([2.0 * np.pi * _integrate(phase * mu, mu) for phase in phases])
    return phases, asymmetry
