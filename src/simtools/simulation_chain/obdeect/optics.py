"""Optical model file and command line of the obdeect ray tracing of a photon stream.

``obdeect ray-tracing trace-stream`` traces a photon stream through the optics of one telescope
(see ``simtools.simulation_chain.optics`` for the request). A different software provides its own
module in a sibling subpackage of ``simtools.simulation_chain`` and its programs in the program
catalog.
"""

from pathlib import Path

from simtools.ray_tracing.backends.interface import create_shower_tracer
from simtools.simulation_chain.programs import program_command

SOFTWARE = "obdeect"
COMMAND_PROGRAM = "obdeect"


def write_optical_model(request, telescope_model, directory=None):
    """
    Compile the optical model of a request from its telescope model and write it as a file.

    The model is compiled by the obdeect ray-tracing backend (``optical_model``) for the pointing
    zenith angle and the instrument seed of the request.

    Parameters
    ----------
    request : OpticsRequest
        Optics job.
    telescope_model : TelescopeModel
        Model of the telescope of the request.
    directory : Path or None
        Directory of the file; default: the configuration directory of the telescope model.

    Returns
    -------
    Path
        Optical model file.
    """
    tracer = create_shower_tracer(telescope_model, name="obdeect")
    model = tracer.optical_model(request.zenith_deg, request.instrument_seed)
    path = Path(directory or telescope_model.config_file_directory) / request.model_file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def trace_stream_command(request, model_file, input_file="-", output_file="-"):
    """
    Return the tokens of the obdeect command tracing a photon stream through the optics.

    Parameters
    ----------
    request : OpticsRequest
        Optics job.
    model_file : str or Path
        Optical model file (see :func:`write_optical_model`).
    input_file, output_file : str or Path
        Photon stream files; ``-`` is standard input or output.

    Returns
    -------
    list of str
        Command tokens.
    """
    return [
        *program_command(SOFTWARE, COMMAND_PROGRAM),
        "ray-tracing",
        "trace-stream",
        "--model",
        str(model_file),
        "--seed",
        str(request.tracing_seed),
        "--threads",
        str(request.threads),
        "--telescopes",
        str(request.telescope_id),
        "--pointing-zenith-deg",
        str(request.zenith_deg),
        "--pointing-azimuth-deg",
        str(request.azimuth_deg),
        "--telescope-transmission",
        request.telescope_transmission,
        "--input",
        str(input_file),
        "--output",
        str(output_file),
    ]
