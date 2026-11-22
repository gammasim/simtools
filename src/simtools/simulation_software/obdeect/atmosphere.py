"""Build obdeect atmospheric propagation commands."""

from pathlib import Path

from simtools.simulation_software.efficiency import OBDEECT_QE_FACTOR
from simtools.simulation_software.obdeect.streams import SOFTWARE
from simtools.simulation_software.programs import STDIO, program_command

PROPAGATION_PROGRAM = "obdeect-propagation-run"


def propagation_command(
    request, *, input_file=STDIO, output_file=STDIO, envelope_file=None, program=None
):
    """
    Return the command-line tokens of the obdeect propagation for an atmosphere request.

    Parameters
    ----------
    request : AtmosphereRequest
        Atmosphere request.
    input_file : str or Path
        Input photon stream; ``-`` reads stdin.
    output_file : str or Path
        Output photon stream; ``-`` writes stdout.
    envelope_file : str or Path, optional
        Quantum-efficiency bound (two-column nm, value) thinning the photons (``--envelope qe``).
    program : str, Path or list of str, optional
        Propagation program (command tokens); default: the program of the program catalog.

    Returns
    -------
    list[str]
        Command-line tokens.
    """
    if program is None:
        program = program_command(SOFTWARE, PROPAGATION_PROGRAM)
    elif isinstance(program, (str, Path)):
        program = [str(program)]
    command = [
        *program,
        "--transmission",
        str(request.transmission_file),
        "--site-altitude",
        str(request.observation_level_m),
        "--profile",
        str(request.profile_file),
        "--seed",
        str(request.seed),
        "--mode",
        request.mode,
    ]
    if request.wavelength_range_nm is not None:
        low, high = request.wavelength_range_nm
        command += ["--wavelength-range", str(low), str(high)]
    for telescope_id, pathlength in request.telescope_pathlen_m.items():
        command += ["--telescope-pathlen", f"{telescope_id}={pathlength}"]
    if envelope_file is not None:
        command += ["--envelope", f"{OBDEECT_QE_FACTOR}={envelope_file}"]
    return [*command, "-i", str(input_file), "-o", str(output_file)]
