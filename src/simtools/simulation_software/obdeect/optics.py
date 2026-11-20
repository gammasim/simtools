"""Export optical models and build obdeect photon-stream tracing commands."""

from pathlib import Path

from simtools.ray_tracing.backends.interface import create_shower_tracer
from simtools.simulation_software.obdeect.streams import SOFTWARE
from simtools.simulation_software.programs import STDIO, program_command

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


#: Request fields that one tracing process shares between its telescopes.
_SHARED_FIELDS = ("tracing_seed", "threads", "zenith_deg", "azimuth_deg", "telescope_transmission")


def trace_stream_command(requests, model_files, input_file=STDIO, output_file=STDIO):
    """
    Return the tokens of the obdeect command tracing a photon stream through several telescopes.

    One process traces every telescope of the requests with its own optical model and writes
    one camera-frame stream in event order; telescopes without a request are dropped.

    Parameters
    ----------
    requests : sequence of OpticsRequest
        Optics jobs, one per telescope, with the same pointing, tracing seed, telescope
        transmission and threads.
    model_files : sequence of str or Path
        Optical model file of each request (see :func:`write_optical_model`).
    input_file, output_file : str or Path
        Photon stream files; ``-`` is standard input or output.

    Returns
    -------
    list of str
        Command tokens.

    Raises
    ------
    ValueError
        If there is no request, the numbers of requests and model files differ, or the requests
        differ in a shared field.
    """
    requests = list(requests)
    model_files = list(model_files)
    if not requests or len(requests) != len(model_files):
        raise ValueError(
            f"Tracing needs one model file per request; got {len(requests)} requests and "
            f"{len(model_files)} model files."
        )
    first = requests[0]
    for name in _SHARED_FIELDS:
        values = {getattr(request, name) for request in requests}
        if len(values) > 1:
            raise ValueError(f"Optics requests of one tracing process differ in {name}: {values}.")
    models = []
    for request, model_file in zip(requests, model_files):
        models += ["--model", f"{request.telescope_id}={model_file}"]
    return [
        *program_command(SOFTWARE, COMMAND_PROGRAM),
        "ray-tracing",
        "trace-stream",
        *models,
        "--seed",
        str(first.tracing_seed),
        "--threads",
        str(first.threads),
        "--pointing-zenith-deg",
        str(first.zenith_deg),
        "--pointing-azimuth-deg",
        str(first.azimuth_deg),
        "--telescope-transmission",
        first.telescope_transmission,
        "--input",
        str(input_file),
        "--output",
        str(output_file),
    ]
