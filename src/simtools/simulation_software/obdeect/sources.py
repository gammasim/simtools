"""Build obdeect flasher and illuminator commands from light-source requests."""

import json
import sys
from pathlib import Path

import numpy as np

from simtools.simulation.light_source_profiles import angular_profile, pulse_profile
from simtools.simulation_software.light_sources import request_to_json
from simtools.simulation_software.obdeect.streams import SOFTWARE
from simtools.simulation_software.programs import program_command

ILLUMINATOR_PROGRAM = "obdeect"
FLASHER_MODULE = "simtools.simulation_software.flasher_stream"


def _number(value):
    """Return a number as command text."""
    return format(float(value), ".10g")


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
        return (
            f"gauss:sigma_deg={_number(angular.width_deg)}"
            if shape == "gauss"
            else f"filled_cone:angle_deg={_number(angular.width_deg)}"
        )
    angles, values = angular_profile(angular.shape, angular.width_deg)
    path = Path(work_directory) / f"light_source_angular_distribution{file_token}.dat"
    return f"table:{_write_table(path, (angles, values), 'angle [deg], relative intensity')}"


def illuminator_pulse_token(pulse, work_directory, file_token=""):
    """Return the pulse shape as point-source option value (a table when needed)."""
    shape = pulse.shape.lower()
    if shape in ("gauss", "tophat"):
        if pulse.width_ns == 0:
            return "fixed:time_ns=0"
        return (
            f"gauss:sigma_ns={_number(pulse.width_ns)}"
            if shape == "gauss"
            else f"tophat:width_ns={_number(pulse.width_ns)}"
        )
    if shape == "exponential":
        if pulse.decay_ns <= 0:
            return "fixed:time_ns=0"
        return f"exponential:decay_ns={_number(pulse.decay_ns)},mean_ns={_number(pulse.decay_ns)}"
    profile = pulse_profile(pulse.shape, pulse.width_ns, pulse.decay_ns)
    if profile is None:
        return "fixed:time_ns=0"
    path = Path(work_directory) / f"light_source_pulse_shape{file_token}.dat"
    return f"table:{_write_table(path, profile, 'time [ns], relative rate')}"


def _output_tokens(output):
    """Return the output option (none: the program writes to stdout)."""
    return [] if output is None else ["-o", str(output)]


def flasher_command(
    request,
    work_directory,
    output=None,
    file_token="",
    executable=None,
    simulation_models_path=None,
):
    """
    Return the command tokens that make the photons of a flasher request.

    The command runs ``simtools.simulation_software.flasher_stream``, which traces the light of
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
    simulation_models_path : str or Path, optional
        Simulation models root of the run. The subprocess reads the telescope model from it
        (without it the subprocess has no model source).

    Returns
    -------
    list of str
        Command tokens.
    """
    request_file = request_to_json(
        request, Path(work_directory) / f"light_source_request{file_token}.json"
    )
    models = (
        []
        if simulation_models_path is None
        else ["--simulation-models-path", str(simulation_models_path)]
    )
    return [
        executable or sys.executable,
        "-m",
        FLASHER_MODULE,
        "--request",
        str(request_file),
        "--output-path",
        str(work_directory),
        *models,
        *(["--output", str(output)] if output is not None else []),
    ]


def _profile(token, x_key, y_key):
    """Convert a profile token into a run-description profile."""
    kind, _, arguments = token.partition(":")
    if kind == "table":
        x, y = np.loadtxt(arguments, unpack=True)
        return {"kind": kind, x_key: x.tolist(), y_key: y.tolist()}
    return {"kind": kind} | {
        key: float(value)
        for key, value in (item.split("=", 1) for item in arguments.split(",") if item)
    }


def illuminator_command(
    request,
    work_directory,
    output=None,
    file_token="",
    executable=None,
    simulation_models_path=None,
):
    """
    Return the command tokens that make the photons of an illuminator request.

    The run description and any profile tables are written to ``work_directory``.
    Positions are in metres.

    Parameters
    ----------
    request : IlluminatorRequest
        Illuminator request.
    work_directory : str or Path
        Directory for profile tables.
    output : str or Path, optional
        Output file; without it the program writes the photon stream to stdout.
    file_token : str
        Suffix of the table file names.
    executable : str, optional
        Program (default: the obdeect command of the program catalog).
    simulation_models_path : str or Path, optional
        Not used: the illuminator needs no model files (same signature as the flasher).

    Returns
    -------
    list of str
        Command tokens.
    """
    del simulation_models_path
    angular = illuminator_angular_token(request.angular, work_directory, file_token)
    pulse = illuminator_pulse_token(request.pulse, work_directory, file_token)
    position = np.asarray(request.position_m, dtype=float)
    centre = np.asarray(request.telescope_position_m, dtype=float)
    displacement = centre - position
    distance = float(np.linalg.norm(displacement))
    radius = float(request.telescope_radius_m)
    if distance <= 1.02 * radius:
        target = "none"
    elif angular == "isotropic":
        target = {"kind": "sphere", "centre_m": centre.tolist(), "radius_m": radius}
    else:
        target = {
            "kind": "disc",
            "centre_m": centre.tolist(),
            "normal": (displacement / distance).tolist(),
            "radius_m": radius / np.sqrt(1 - (radius / distance) ** 2),
        }
    emission = {
        "kind": "point",
        "position_m": position.tolist(),
        "axis": list(request.pointing),
        "bunch_size": request.bunch_size,
        "angular": _profile(angular, "angle_deg", "value"),
        "pulse": _profile(pulse, "time_ns", "rate"),
        "spectrum": {"kind": "fixed", "wavelength_nm": request.wavelength_nm},
        "target": target,
    }
    description = {
        "run": {
            "number": request.run_number,
            "seed": request.seed,
            "observation_level_m": request.observation_level_m,
            "frame": "sky",
        },
        "telescopes": [{"id": 1, "position_m": centre.tolist(), "radius_m": radius}],
        "events": [
            {"repeat": request.events, "emissions": [emission | {"photons": n}]}
            for n in request.photons
        ],
    }
    path = Path(work_directory) / f"light_source_run{file_token}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(description), encoding="utf-8")
    return [
        *([executable] if executable else program_command(SOFTWARE, ILLUMINATOR_PROGRAM)),
        "sources",
        "run",
        str(path),
        *_output_tokens(output),
    ]
