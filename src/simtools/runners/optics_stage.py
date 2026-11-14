"""Telescope optics as a simulation stage: photons at the telescope traced to the camera.

The stage traces a photon stream (ground or telescope frame) through the optics of one
telescope of an array model and writes camera-frame photons. The photons of the other telescopes
are dropped, so a chain runs one instance per telescope and array model, each with its own random
instrument (mirror errors). The translation of the telescope models into the optics job is in
``simtools.simulation_chain.optics``; the optical model file and the command are made in
``simtools.simulation_chain.obdeect.optics``.

Stage options are given after the stage name in the chain::

    simulation_chain: [corsika, corsika_to_stream, obdeect_atmosphere,
                       "obdeect_optics:telescope_transmission=full,threads=4",
                       stream_to_simtel, sim_telarray]

``telescope_transmission`` is ``full`` (the only mode); ``threads`` is the number of tracing
threads of each instance (default: the usable cores divided among the instances).
"""

import os
import shlex
from dataclasses import replace

from simtools.corsika.corsika_config import corsika_telescope_ids
from simtools.job_execution.process_accounting import build_accounting_command
from simtools.model.model_utils import export_model_files
from simtools.runners.chain_spec import MIRROR_REFLECTIVITY, TELESCOPE_TRANSMISSION
from simtools.runners.stages import StreamTo
from simtools.simulation_chain.obdeect.optics import trace_stream_command, write_optical_model
from simtools.simulation_chain.optics import (
    TELESCOPE_TRANSMISSION_MODES,
    optics_requests,
)
from simtools.simulation_chain.products import PHOTONS_CAMERA, PHOTONS_GROUND, PHOTONS_TELESCOPE

STAGE_NAME = "obdeect_optics"
#: Photon streams number telescopes from 1 (sim_telarray numbering); the IACT ids of the CORSIKA
#: telescope list start at 0.
STREAM_ID_OFFSET = 1


def _telescope_transmission(options):
    """Return the telescope transmission mode of the stage options."""
    mode = options.get("telescope_transmission", "full")
    if mode not in TELESCOPE_TRANSMISSION_MODES:
        raise ValueError(
            f"Invalid telescope transmission {mode!r} of stage {STAGE_NAME!r}; only "
            f"{', '.join(TELESCOPE_TRANSMISSION_MODES)} is supported (with 'angular' the "
            "constant part of the transmission would stay unapplied in the chain)"
        )
    return mode


class ObdeectOpticsStage:
    """
    Optics of one telescope of one array model (obdeect ``trace-stream``).

    Parameters
    ----------
    request : OpticsRequest
        Optics job of the telescope.
    telescope_model : TelescopeModel
        Model of the telescope (the optical model is compiled from it).
    context : RunContext
        Run context (output paths).
    label : str
        Instance label.
    position, count : int
        Position of this instance among the ``count`` instances of the stage. The downstream
        command of a streamed output is selected by the position when there are several.
    """

    name = STAGE_NAME
    consumes = (PHOTONS_GROUND, PHOTONS_TELESCOPE)
    produces = PHOTONS_CAMERA
    streams = True

    def __init__(self, request, telescope_model, context, label=None, position=0, count=1):
        """Initialize ObdeectOpticsStage."""
        self.request = request
        self.telescope_model = telescope_model
        self.context = context
        self.label = label
        self.position = position
        self.count = count
        self.file_list = {}

    @classmethod
    def applied_factors(cls, options):
        """
        Return the efficiency factors the stage applies to the photons.

        obdeect applies the mirror reflectivity and the telescope transmission (the geometry of
        the telescope structure replaces the transmission when it is traced).
        """
        _telescope_transmission(options)
        return {MIRROR_REFLECTIVITY, TELESCOPE_TRANSMISSION}

    @classmethod
    def instances(cls, inputs, label=None):
        """Return one stage per telescope of each array model."""
        if inputs.context is None:
            raise ValueError(f"Stage {STAGE_NAME!r} needs a run context.")
        options = inputs.options_of(STAGE_NAME)
        transmission = _telescope_transmission(options)
        pairs = []
        for parameters, array_model in zip(inputs.simulation_parameters, inputs.array_models):
            ids = corsika_telescope_ids(array_model.telescope_models)
            for request in optics_requests(
                array_model,
                parameters,
                ids,
                stream_id_offset=STREAM_ID_OFFSET,
                telescope_transmission=transmission,
            ):
                pairs.append((request, array_model.telescope_models[request.telescope_name]))
        threads = _threads(options, inputs.args, len(pairs))
        return [
            cls(
                _with_threads(request, threads),
                telescope,
                inputs.context,
                label=label,
                position=position,
                count=len(pairs),
            )
            for position, (request, telescope) in enumerate(pairs)
        ]

    def prepare(self, run_number, input_file=None, output=None):
        """
        Write the optical model and return the shell line that traces the photons.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path
            Photon stream file, or ``STDIN`` when reading a stream.
        output : StreamTo or None
            Downstream commands receiving the camera photons; None writes the output file.

        Returns
        -------
        list[str]
            One shell line.
        """
        if input_file is None:
            raise ValueError("The optics stage needs an input photon stream.")
        export_model_files(self.telescope_model, update_if_necessary=True)
        model_file = write_optical_model(self.request, self.telescope_model)
        streamed = isinstance(output, StreamTo)
        command = trace_stream_command(
            self.request,
            model_file,
            input_file=input_file,
            output_file="-" if streamed else self.output_file(run_number),
        )
        resources_file = self._file("resources.json", run_number)
        log_file = self._file("log", run_number)
        self.file_list = {"obdeect_optics_resources": [resources_file]}
        self.file_list["obdeect_optics_log"] = [log_file]
        text = _guarded(
            shlex.join(
                build_accounting_command(
                    command,
                    resources_file,
                    STAGE_NAME,
                    run_number,
                    model_version=self.request.model_version,
                )
            ),
            log_file,
        )
        if streamed:
            return [f"{text} | {self._downstream_command(output)}"]
        return [text]

    def output_file(self, run_number):
        """Return the camera photon stream file of this telescope."""
        return self._file("camera.pst", run_number)

    def environment(self):
        """Return environment variables needed by the stage (none)."""
        return {}

    def get_resources(self, runtime=None):
        """Return computing resources used (reported by the first stage of the chain only)."""
        del runtime

    def _file(self, suffix, run_number):
        """Return a file of this instance in the output directory."""
        request = self.request
        return self.context.paths.get_output_directory() / (
            f"{request.telescope_name}_{request.model_version}_run{run_number:06d}"
            f"_{STAGE_NAME}.{suffix}"
        )

    def _downstream_command(self, output):
        """Return the downstream command that receives the camera photons of this instance."""
        commands = output.commands
        if len(commands) == 1:
            return commands[0]
        if len(commands) == self.count:
            return commands[self.position]
        raise ValueError(
            f"Stage {STAGE_NAME!r} has {self.count} instances, but {len(commands)} downstream "
            "commands receive its stream."
        )


def _threads(options, args, instances):
    """Return the tracing threads of each instance."""
    if "threads" in options:
        return int(options["threads"])
    if args.get("sequential", False):
        return 1
    return max(1, (os.process_cpu_count() or 1) // max(1, instances))


def _with_threads(request, threads):
    """Return the request with the number of tracing threads."""
    return replace(request, threads=threads)


def _guarded(command_text, log_file):
    """
    Return the command with stderr in a log file, as a group that can start a pipeline.

    A failing command prints its log to the job output and fails the job. In a stream, the
    multipipe_corsika process (``MULTIPIPE_PID``) is also stopped, since it would wait for
    wakeups that no longer come.
    """
    log = shlex.quote(str(log_file))
    on_failure = (
        f"{{ status=$?; cat {log} >&2; "
        '[ -z "${MULTIPIPE_PID:-}" ] || kill -TERM "$MULTIPIPE_PID" 2>/dev/null; '
        "exit $status; }"
    )
    return f"{{ {command_text} 2> {log} || {on_failure}; }}"
