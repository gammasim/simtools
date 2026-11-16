"""Convert CORSIKA photons to streams and camera streams to sim_telarray input."""

import shlex

from simtools.runners.stages import (
    CatalogStage,
    StreamTo,
    efficiency_envelope_enabled,
    files_of_run,
    stage_file,
)
from simtools.simulation_software import efficiency
from simtools.simulation_software.atmosphere import export_atmosphere_files
from simtools.simulation_software.obdeect import streams


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


class CorsikaToStreamStage(CatalogStage):
    """
    Convert CORSIKA IACT photon bunches to a photon stream in the ground frame.

    Parameters
    ----------
    run_files : RunnerServices
        File naming of the run.
    """

    streams = True

    def __init__(self, run_files):
        """Initialize CorsikaToStreamStage."""
        self.run_files = run_files
        self.file_list = None

    @classmethod
    def instances(cls, inputs, label=None):
        """Return the stage of the run."""
        return [cls(files_of_run(inputs, label))]

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
            raise ValueError(f"{self.name} reads the CORSIKA output: give a file or STDIN.")
        command = shlex.join(streams.from_corsika_command(input_file))
        return _connect_output(command, output, self.output_file(run_number))

    def output_file(self, run_number):
        """Return the photon stream file of the run."""
        return stage_file(self, "output", run_number)

    def environment(self):
        """Return the environment variables of the stage (none)."""
        return {}

    def get_resources(self, runtime=None):
        """Return computing resources used (of the CORSIKA run)."""
        return self.run_files.get_resources(runtime=runtime)


class StreamToSimtelStage(CatalogStage):
    """
    Convert a camera-frame photon stream to sim_telarray input.

    All telescopes of the array are listed in the camera map: their CORSIKA telescope ids, mirror
    class and camera rotation. The photons are given at the camera entrance, so sim_telarray
    needs Bypass_Optics for the telescopes (see ``SimTelArrayStage``).

    Parameters
    ----------
    run_files : RunnerServices
        File naming of the run.
    array_model : ArrayModel
        Telescope models of the array.
    allow_pending : tuple of str
        Efficiency factors pending upstream that sim_telarray does not apply (``--allow-pending``).
    scaled_factors : tuple of str
        Efficiency factors whose sim_telarray table holds f/f_env (``--scaled-factor``).
    write_atmprof : bool
        Write the atmospheric profile of the site for the converter (light-source streams).
    """

    streams = True

    def __init__(
        self,
        run_files,
        array_model,
        allow_pending=(),
        scaled_factors=(),
        write_atmprof=False,
    ):
        """Initialize StreamToSimtelStage."""
        self.run_files = run_files
        self.write_atmprof = write_atmprof
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
        options = cls.stage_options(inputs)
        scaled_factors = tuple(options.get("scaled_factors", ()))
        if efficiency_envelope_enabled(inputs):
            # sim_telarray's QE table holds QE / envelope (see simulation_software.efficiency)
            scaled_factors = tuple(dict.fromkeys((*scaled_factors, efficiency.OBDEECT_QE_FACTOR)))
        return [
            cls(
                files_of_run(inputs, label),
                inputs.array_models[0],
                allow_pending=options.get("allow_pending", ()),
                scaled_factors=scaled_factors,
                # light-source streams carry no ATMPROF block: the site profile is written
                write_atmprof=inputs.calibration is not None,
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
            raise ValueError(f"{self.name} reads a photon stream: give a file or STDIN.")
        atmprof = self.write_atmprof_file(run_number) if self.write_atmprof else None
        command = streams.to_simtel_command(
            self.write_camera_map(run_number),
            input_file,
            self.allow_pending,
            self.scaled_factors,
            atmprof=atmprof,
        )
        return _connect_output(shlex.join(command), output, self.output_file(run_number))

    def camera_map_file(self, run_number):
        """Return the camera map file of the run."""
        return stage_file(self, "camera_map", run_number)

    def write_atmprof_file(self, run_number):
        """Write the atmospheric profile of the site (atmprof text) and return its file."""
        profile, _transmission = export_atmosphere_files(
            self.array_model.site_model, self.camera_map_file(run_number).parent
        )
        return profile

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
        return stage_file(self, "output", run_number)

    def environment(self):
        """Return the environment variables of the stage (none)."""
        return {}

    def get_resources(self, runtime=None):
        """Return computing resources used (of the CORSIKA run)."""
        return self.run_files.get_resources(runtime=runtime)
