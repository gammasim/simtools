"""Build obdeect flasher and illuminator commands from light-source requests."""

import sys
from pathlib import Path

import numpy as np

from simtools.simulation.light_source_profiles import angular_profile, pulse_profile
from simtools.simulation_software.light_sources import request_to_json
from simtools.simulation_software.programs import program_command

SOFTWARE = "obdeect"
ILLUMINATOR_PROGRAM = "obdeect-sources-point-source"
FLASHER_MODULE = "simtools.simulation_software.flasher_stream"


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


def illuminator_command(request, work_directory, output=None, file_token="", executable=None):
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
        Program (default: the point-source program of the program catalog).

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
        *([executable] if executable else program_command(SOFTWARE, ILLUMINATOR_PROGRAM)),
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
