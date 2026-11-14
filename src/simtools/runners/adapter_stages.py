"""Adapter stages of the obdeect chain: CORSIKA photons to photon streams and back to sim_telarray.

``corsika_to_stream`` converts CORSIKA IACT output to a photon stream; ``stream_to_simtel``
converts a camera-frame photon stream to sim_telarray input. The commands are built in
``simtools.simulation_chain.obdeect.streams`` and the programs are located with
``simtools.simulation_chain.programs``; the obdeect package is not imported. In a chain::

    simulation_chain: [corsika, corsika_to_stream, obdeect_atmosphere, obdeect_optics,
                       stream_to_simtel, sim_telarray]
"""

import shlex

from simtools.job_execution.runner_services import RunnerServices
from simtools.runners.stages import StreamTo
from simtools.simulation_chain.obdeect import streams
from simtools.simulation_chain.products import (
    CAMERA_PHOTONS_IACT,
    CHERENKOV_PHOTONS,
    PHOTONS_CAMERA,
    PHOTONS_GROUND,
)

STREAM_SUFFIX = ".pst"
CAMERA_MAP_SUFFIX = ".camera_map.txt"
CAMERA_IACT_SUFFIX = ".camera.iact"


def _corsika_runner_service(inputs, label):
    """Return the file names of the CORSIKA run (the first shower configuration defines it)."""
    if not inputs.shower_configurations:
        raise ValueError("The obdeect chain needs a CORSIKA configuration (shower simulations).")
    config = inputs.shower_configurations[0]
    return RunnerServices(
        config,
        run_type="corsika",
        label=label,
        context=inputs.context or config.context,
    )


def _stream_file(runner_service, run_number, suffix):
    """Return a file next to the CORSIKA output of a run, with a suffix added to its name."""
    corsika_output = runner_service.get_file_name("corsika_output", run_number=run_number)
    return corsika_output.with_name(f"{corsika_output.name}{suffix}")


def _string_list(options, key):
    """Return a list option as a tuple of strings (a single string is one item)."""
    value = options.get(key, ())
    return (value,) if isinstance(value, str) else tuple(value)


def _connect_output(command, output, output_file):
    """
    Return the shell lines of a stream command.

    A streamed output (downstream commands) is piped into each downstream command; the
    downstream commands read the stream from their standard input. Without a streamed output,
    the command writes the stream to ``output_file``.
    """
    if isinstance(output, StreamTo):
        return [f"{command} | {downstream}" for downstream in output.commands]
    return [f"{command} > {shlex.quote(str(output_file))}"]


class CorsikaToStreamStage:
    """
    Convert CORSIKA IACT photon bunches to a photon stream in the ground frame.

    Parameters
    ----------
    runner_service : RunnerServices
        File names of the CORSIKA run.
    """

    name = "corsika_to_stream"
    consumes = CHERENKOV_PHOTONS
    produces = PHOTONS_GROUND
    streams = True

    def __init__(self, runner_service):
        """Initialize CorsikaToStreamStage."""
        self.runner_service = runner_service
        self.file_list = None

    @classmethod
    def instances(cls, inputs, label=None):
        """Return the stage of the CORSIKA run."""
        return [cls(_corsika_runner_service(inputs, label))]

    def prepare(self, run_number, input_file=None, output=None):
        """
        Return the shell lines converting the CORSIKA output.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path
            CORSIKA output file, or ``STDIN`` when reading a stream.
        output : StreamTo or None
            Downstream commands receiving the photon stream; None writes the stream to the
            stage output file.

        Returns
        -------
        list[str]
            Shell lines.
        """
        if input_file is None:
            raise ValueError("corsika_to_stream reads the CORSIKA output: give a file or STDIN.")
        command = shlex.join(streams.from_corsika_command(input_file))
        return _connect_output(command, output, self.output_file(run_number))

    def output_file(self, run_number):
        """Return the photon stream file of the run."""
        return _stream_file(self.runner_service, run_number, STREAM_SUFFIX)

    def environment(self):
        """Return the environment variables of the stage (none)."""
        return {}

    def get_resources(self, runtime=None):
        """Return computing resources used (of the CORSIKA run)."""
        return self.runner_service.get_resources(runtime=runtime)


class StreamToSimtelStage:
    """
    Convert a camera-frame photon stream to sim_telarray input.

    All telescopes of the array are listed in the camera map: their CORSIKA telescope ids, mirror
    class and camera rotation. The photons are given at the camera entrance, so sim_telarray
    needs Bypass_Optics for the telescopes (see ``SimTelArrayStage``).

    Parameters
    ----------
    runner_service : RunnerServices
        File names of the CORSIKA run.
    array_model : ArrayModel
        Telescope models of the array.
    allow_pending : tuple of str
        Efficiency factors pending upstream that sim_telarray does not apply (``--allow-pending``).
    scaled_factors : tuple of str
        Efficiency factors whose sim_telarray table holds f/f_env (``--scaled-factor``).
    """

    name = "stream_to_simtel"
    consumes = PHOTONS_CAMERA
    produces = CAMERA_PHOTONS_IACT
    streams = True

    def __init__(self, runner_service, array_model, allow_pending=(), scaled_factors=()):
        """Initialize StreamToSimtelStage."""
        self.runner_service = runner_service
        self.array_model = array_model
        self.allow_pending = tuple(allow_pending)
        self.scaled_factors = tuple(scaled_factors)
        self.file_list = None

    @classmethod
    def instances(cls, inputs, label=None):
        """Return the stage of the array (the obdeect chain converts one array model per run)."""
        if len(inputs.array_models) != 1:
            raise ValueError(
                "The obdeect chain converts one array model per run; "
                f"got {len(inputs.array_models)} array models."
            )
        options = inputs.options_of(cls.name)
        return [
            cls(
                _corsika_runner_service(inputs, label),
                inputs.array_models[0],
                allow_pending=_string_list(options, "allow_pending"),
                scaled_factors=_string_list(options, "scaled_factors"),
            )
        ]

    def prepare(self, run_number, input_file=None, output=None):
        """
        Write the camera map and return the shell lines converting the photon stream.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path
            Photon stream file, or ``STDIN`` when reading a stream.
        output : StreamTo or None
            Downstream commands (sim_telarray) receiving the converted stream; None writes the
            stream to the stage output file.

        Returns
        -------
        list[str]
            Shell lines.
        """
        if input_file is None:
            raise ValueError("stream_to_simtel reads a photon stream: give a file or STDIN.")
        command = streams.to_simtel_command(
            self.write_camera_map(run_number), input_file, self.allow_pending, self.scaled_factors
        )
        return _connect_output(shlex.join(command), output, self.output_file(run_number))

    def camera_map_file(self, run_number):
        """Return the camera map file of the run."""
        return _stream_file(self.runner_service, run_number, CAMERA_MAP_SUFFIX)

    def write_camera_map(self, run_number):
        """
        Write the camera map of the array and return its file.

        Each line is ``telescope_id mirror_class camera_rotation_deg``, with the CORSIKA
        telescope id of the telescope.
        """
        return streams.write_camera_map(
            self.array_model.telescope_models, self.camera_map_file(run_number)
        )

    def output_file(self, run_number):
        """Return the sim_telarray input file of the run."""
        return _stream_file(self.runner_service, run_number, CAMERA_IACT_SUFFIX)

    def environment(self):
        """Return the environment variables of the stage (none)."""
        return {}

    def get_resources(self, runtime=None):
        """Return computing resources used (of the CORSIKA run)."""
        return self.runner_service.get_resources(runtime=runtime)
