#!/usr/bin/python3

"""Derive obdeect atmospheric transmission and scattering tables with a selected backend."""

from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.application.definition import ApplicationDefinition
from simtools.atmosphere.optical_tables import (
    get_backend,
    write_obdeect_scattering_table,
    write_obdeect_transmission_table,
    write_optical_properties,
)
from simtools.configuration import arguments as cli

_BACKEND_6S = "6s"
_BACKEND_ECSV = "ecsv"
_SIXS_OPTIONS = (
    "atmosphere_profile",
    "aerosol_profile",
    "aot550",
    "ozone",
    "water",
    "sixs_path",
    "processes",
    "phase_angles",
)
_ALTITUDE_RANGE = ("altitude_min", "altitude_max", "altitude_steps")
_TRANSMISSION_FILE = "obdeect_transmission.txt"
_SCATTERING_FILE = "obdeect_scattering.txt"
_SUMMARY_WAVELENGTH = 350.0 * u.nm

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "backend",
        help="Backend that computes the optical properties.",
        choices=[_BACKEND_6S, _BACKEND_ECSV],
        required=True,
    ),
    cli.ArgumentDefinition(
        "ecsv_file",
        help="ECSV file with optical properties (--backend ecsv).",
        type=str,
    ),
    cli.ArgumentDefinition(
        "atmosphere_profile",
        help="6S atmosphere profile name (--backend 6s; default: backend default).",
        type=str,
    ),
    cli.ArgumentDefinition(
        "aerosol_profile",
        help="6S aerosol profile name (--backend 6s; default: backend default).",
        type=str,
    ),
    cli.ArgumentDefinition(
        "aot550",
        help="6S aerosol optical thickness at 550 nm (--backend 6s; default: backend default).",
        type=float,
    ),
    cli.ArgumentDefinition(
        "ozone",
        help="6S ozone column (--backend 6s; default: backend default).",
        type=float,
    ),
    cli.ArgumentDefinition(
        "water",
        help="6S water vapour column (--backend 6s; default: backend default).",
        type=float,
    ),
    cli.ArgumentDefinition(
        "sixs_path",
        help="Path to the 6S executable (--backend 6s; default: search PATH).",
        type=str,
    ),
    cli.ArgumentDefinition(
        "processes",
        help="Number of parallel 6S processes (--backend 6s; default: backend default).",
        type=int,
    ),
    cli.ArgumentDefinition(
        "wavelength_min",
        help="Smallest wavelength in nm.",
        type=float,
        required=True,
    ),
    cli.ArgumentDefinition(
        "wavelength_max",
        help="Largest wavelength in nm (included if on the grid).",
        type=float,
        required=True,
    ),
    cli.ArgumentDefinition(
        "wavelength_step",
        help="Wavelength grid step in nm.",
        type=float,
        required=True,
    ),
    cli.ArgumentDefinition(
        "altitudes",
        help="Altitudes in m a.s.l. (alternative to the altitude range options).",
        type=float,
        nargs="+",
    ),
    cli.ArgumentDefinition(
        "altitude_min",
        help="Smallest altitude in m a.s.l. for log spacing (requires --altitude_max and steps).",
        type=float,
    ),
    cli.ArgumentDefinition(
        "altitude_max",
        help="Largest altitude in m a.s.l. for log spacing.",
        type=float,
    ),
    cli.ArgumentDefinition(
        "altitude_steps",
        help="Number of log-spaced altitudes between --altitude_min and --altitude_max.",
        type=int,
    ),
    cli.ArgumentDefinition(
        "observation_level",
        help="Observation level in m a.s.l. used as the zero-depth reference.",
        type=float,
        required=True,
    ),
    cli.ArgumentDefinition(
        "phase_angles",
        help="Scattering phase angles in degrees, e.g. 0 30 90 150 (--backend 6s; default: 2-180).",
        type=float,
        nargs="+",
    ),
    cli.ArgumentDefinition(
        "depolarization",
        help="Depolarization factor written to the scattering table (0 = none).",
        type=float,
        default=0.0,
    ),
    cli.ArgumentDefinition(
        "transmission_file",
        help="Output transmission table file name (in the output directory).",
        type=str,
        default=_TRANSMISSION_FILE,
    ),
    cli.ArgumentDefinition(
        "scattering_file",
        help="Output scattering table file name (in the output directory).",
        type=str,
        default=_SCATTERING_FILE,
    ),
    cli.ArgumentDefinition(
        "properties_file",
        help="Optional output ECSV file with the optical properties (exchange format).",
        type=str,
    ),
)

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=(),
)


def _backend_kwargs(args):
    """Return the backend keyword arguments and reject options of the other backend."""
    if args["backend"] == _BACKEND_ECSV:
        if not args.get("ecsv_file"):
            raise ValueError("--ecsv_file is required with --backend ecsv")
        used = [name for name in _SIXS_OPTIONS if args.get(name) is not None]
        if used:
            raise ValueError(f"Options only apply to --backend 6s: {', '.join(used)}")
        return {"file_name": args["ecsv_file"]}
    if args.get("ecsv_file") is not None:
        raise ValueError("--ecsv_file only applies to --backend ecsv")
    kwargs = {
        name: args[name]
        for name in _SIXS_OPTIONS
        if args.get(name) is not None and name != "phase_angles"
    }
    phase_angles = args.get("phase_angles")
    if phase_angles is not None:
        kwargs["phase_angle"] = np.asarray(phase_angles, dtype=float) * u.deg
    return kwargs


def _check_altitudes_m(altitudes_m):
    """Altitudes must be positive: the obdeect tables interpolate in log(altitude)."""
    if np.any(altitudes_m <= 0):
        raise ValueError(
            f"Altitudes must be positive m a.s.l., got {altitudes_m}. Give sea level as a small "
            "positive altitude, for example 1 m: obdeect tables interpolate in log(altitude)."
        )
    return altitudes_m


def _altitudes(args):
    """Return altitudes in m a.s.l. from an explicit list or a log-spaced range."""
    explicit = args.get("altitudes")
    range_values = [args.get(name) for name in _ALTITUDE_RANGE]
    if explicit is not None:
        if any(value is not None for value in range_values):
            raise ValueError("Use either --altitudes or the --altitude_min/max/steps range")
        return _check_altitudes_m(np.asarray(explicit, dtype=float)) * u.m
    if any(value is None for value in range_values):
        raise ValueError(
            "Give --altitudes or all of --altitude_min, --altitude_max, --altitude_steps"
        )
    minimum, maximum, steps = range_values
    _check_altitudes_m(np.array([minimum, maximum], dtype=float))
    return np.geomspace(minimum, maximum, steps) * u.m


def _wavelengths(args):
    """Return the wavelength grid in nm, including the upper limit if it lies on the grid."""
    minimum = args["wavelength_min"]
    maximum = args["wavelength_max"]
    step = args["wavelength_step"]
    if step <= 0 or maximum < minimum:
        raise ValueError("Require --wavelength_step > 0 and --wavelength_max >= --wavelength_min")
    return np.arange(minimum, maximum + step / 2, step) * u.nm


def _summary_wavelength(properties):
    """Shortest wavelength for the summary: 350 nm if inside the grid, else the grid minimum."""
    wavelength = properties.wavelength
    if wavelength[0] <= _SUMMARY_WAVELENGTH <= wavelength[-1]:
        return _SUMMARY_WAVELENGTH
    return wavelength[0]


def _log_summary(logger, properties, observation_level):
    """Log the vertical transmission from the top of the grid to the observation level."""
    shortest = _summary_wavelength(properties)
    longest = properties.wavelength[-1]
    for label, wavelength in (("shortest", shortest), ("longest", longest)):
        transmission = properties.interpolate_wavelength([wavelength]).vertical_transmission(
            observation_level
        )[0]
        logger.info(
            f"Vertical transmission at {wavelength.to_value(u.nm):.1f} nm ({label} wavelength) "
            f"from {properties.altitude[-1].to_value(u.km):.3g} km to "
            f"{observation_level.to_value(u.m):.1f} m a.s.l.: {transmission:.4f}"
        )


def main():
    """Compute optical properties with the selected backend and write the tables."""
    app_context = APPLICATION.start()
    args = app_context.args
    output_dir = Path(app_context.paths.get_output_directory())

    backend = get_backend(args["backend"], **_backend_kwargs(args))
    properties = backend.compute(_wavelengths(args), _altitudes(args))
    properties = properties.with_depolarization(args["depolarization"])

    observation_level = args["observation_level"] * u.m
    write_obdeect_transmission_table(
        properties,
        output_dir / args["transmission_file"],
        observation_level=observation_level,
    )
    write_obdeect_scattering_table(properties, output_dir / args["scattering_file"])
    if args.get("properties_file"):
        write_optical_properties(properties, output_dir / args["properties_file"])
    _log_summary(app_context.logger, properties, observation_level)

    app_context.logger.info(f"Wrote obdeect tables for backend {args['backend']} to {output_dir}")


if __name__ == "__main__":
    main()
