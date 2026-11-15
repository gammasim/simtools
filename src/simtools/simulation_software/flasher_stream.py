"""Trace flasher light through telescope obscurers and write camera-frame photon streams."""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

from simtools.io.output_paths import OutputPaths
from simtools.model.telescope_model import TelescopeModel
from simtools.model_repository.reader import SimulationModelReader, resolve_model_reader
from simtools.ray_tracing.backends.interface import LightSourceRequest
from simtools.runtime.context import RunContext
from simtools.simulation.light_source_profiles import angular_profile, pulse_profile
from simtools.simulation_software.light_sources import flasher_request_from_json

PRODUCER = "simtools-flasher"
TELESCOPE_ID = 1
MAX_ROWS = 1 << 16
_STREAM_COLUMNS = ("x", "y", "cx", "cy", "cz", "t", "w", "wavelength")


def light_source_request(request):
    """
    Return the ray-tracing light-source request of a ``FlasherRequest``.

    The flasher emits along the optical axis; the telescope points to the zenith.
    """
    return LightSourceRequest(
        position_m=tuple(request.position_m),
        axis=(0.0, 0.0, 1.0),
        angular_profile=angular_profile(request.angular.shape, request.angular.width_deg),
        pulse=pulse_profile(request.pulse.shape, request.pulse.width_ns, request.pulse.decay_ns),
        wavelength_nm=request.wavelength_nm,
        photons_per_pulse=request.photons_per_pulse,
        telescope_zenith=0.0,
        seeds=(request.instrument_seed, request.seed),
        inside_telescope=True,
        camera_radius_m=request.camera_radius_m,
    )


def stream_columns(block):
    """Return the float32 stream columns of a block of photons at the camera plane."""
    position, direction = block["position_m"], block["direction"]
    values = {
        "x": position[:, 0],
        "y": position[:, 1],
        "cx": direction[:, 0],
        "cy": direction[:, 1],
        "cz": direction[:, 2],
        "t": block["time_ns"],
        "w": block["weight"],
        "wavelength": block["wavelength_nm"],
    }
    return {name: np.ascontiguousarray(values[name], dtype=np.float32) for name in _STREAM_COLUMNS}


def _write_pulse(writer, event_number, blocks):
    """Write one pulse as an event with one telescope."""
    writer.write("event", {"event_number": event_number})
    writer.write("telescope", {"telescope_id": TELESCOPE_ID, "frame": 3})
    first_index = 0
    weight = 0.0
    for block in blocks:
        columns = stream_columns(block)
        for start in range(0, len(columns["x"]), MAX_ROWS):
            chunk = {name: values[start : start + MAX_ROWS] for name, values in columns.items()}
            writer.write_photons(TELESCOPE_ID, "camera", first_index + start, chunk)
        first_index += len(columns["x"])
        weight += float(columns["w"].sum(dtype=np.float64))
    writer.write("telescope_end", {"telescope_id": TELESCOPE_ID, "photons": weight})
    writer.write("event_end", {})
    return first_index


def write_flasher_stream(tracer, request, output):
    """
    Trace the light of a flasher and write the camera-frame photon stream.

    Parameters
    ----------
    tracer : ObdeectRayTracer
        Ray tracer of the telescope model.
    request : FlasherRequest
        Flasher request.
    output : str or Path
        Output file, or ``-`` for stdout.

    Returns
    -------
    int
        Number of photons written.
    """
    import obdeect.stream  # pylint: disable=import-outside-toplevel

    pulses = tracer.trace_light_source(light_source_request(request))
    sink = sys.stdout.buffer if str(output) == "-" else str(output)
    total = 0
    with obdeect.stream.Writer(sink, PRODUCER) as writer:
        writer.write(
            "run",
            {
                "run_number": request.run_number,
                "observation_level_m": request.observation_level_m,
                "wavelength_min_nm": request.wavelength_nm,
                "wavelength_max_nm": request.wavelength_nm,
                "n_telescopes": 1,
                "refraction_applied": 0,
            },
        )
        for event_number, blocks in enumerate(pulses, start=1):
            total += _write_pulse(writer, event_number, blocks)
        writer.write("run_end", {})
    return total


def create_telescope_model(request, simulation_models_path=None, context=None):
    """Return the telescope model of a flasher request."""
    if simulation_models_path:
        reader = SimulationModelReader.from_files(simulation_models_path)
    else:
        reader = resolve_model_reader(None, context)
    return TelescopeModel(
        site=request.site,
        telescope_name=request.telescope,
        model_version=request.model_version,
        model_reader=reader,
        context=context,
    )


def parse_args(argv=None):
    """Parse the command line of the flasher stream process."""
    parser = argparse.ArgumentParser(
        description="Write the camera-frame photon stream of a flasher (stdout or --output)."
    )
    parser.add_argument("--request", type=Path, required=True, help="JSON flasher request.")
    parser.add_argument("--output", default="-", help="Photon-stream file, or '-' for stdout.")
    parser.add_argument(
        "--output-path", type=Path, default=None, help="Work directory for exported model files."
    )
    parser.add_argument(
        "--simulation-models-path",
        type=Path,
        default=None,
        help="Simulation models root (default: configured model source).",
    )
    return parser.parse_args(argv)


def main(argv=None, *, telescope_model_factory=create_telescope_model):
    """
    Run the flasher stream process.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments (default: ``sys.argv``).
    telescope_model_factory : callable
        Returns the telescope model of a ``FlasherRequest`` (keyword arguments
        ``simulation_models_path`` and ``context``).

    Returns
    -------
    int
        Exit status.
    """
    from simtools.ray_tracing.backends.obdeect import (  # pylint: disable=import-outside-toplevel
        ObdeectRayTracer,
    )

    args = parse_args(argv)
    work_path = args.output_path or Path(tempfile.gettempdir()) / "simtools-flasher-stream"
    context = RunContext.from_arguments(
        {"output_path": str(work_path)}, paths=OutputPaths(output_path=str(work_path))
    )
    request = flasher_request_from_json(args.request)
    model = telescope_model_factory(
        request, simulation_models_path=args.simulation_models_path, context=context
    )
    total = write_flasher_stream(ObdeectRayTracer(model), request, args.output)
    logging.getLogger(__name__).info(f"Flasher stream: {total} photons")
    return 0


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    sys.exit(main())
