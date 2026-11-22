"""Optics stage of the obdeect chain: mirrors, shadowing and telescope transmission."""

import os
import shlex

from simtools.corsika.corsika_config import corsika_telescope_ids
from simtools.job_execution.process_accounting import build_accounting_command
from simtools.model.model_utils import export_model_files
from simtools.runners.stages import (
    CatalogStage,
    StreamTo,
    files_of_run,
    single_downstream,
    stage_file,
)
from simtools.simulation_software.obdeect.optics import trace_stream_command, write_optical_model
from simtools.simulation_software.obdeect.streams import STREAM_TELESCOPE_ID_OFFSET
from simtools.simulation_software.optics import optics_requests
from simtools.simulation_software.programs import STDIO


class ObdeectOpticsStage(CatalogStage):
    """
    Optics of the telescopes of one array model, traced by one obdeect ``trace-stream`` process.

    Each telescope has its own optical model; the process writes one camera-frame stream with
    the events in input order, so the stage streams to the camera stage.

    Parameters
    ----------
    requests : tuple of OpticsRequest
        Optics job of each telescope, in the order of the array.
    telescope_models : tuple of TelescopeModel
        Model of each telescope (its optical model is compiled from it).
    run_files : RunnerServices
        File naming of the run.
    """

    streams = True

    def __init__(self, requests, telescope_models, run_files):
        """Initialize ObdeectOpticsStage."""
        self.requests = tuple(requests)
        self.telescope_models = tuple(telescope_models)
        self.run_files = run_files
        self.file_list = None

    @classmethod
    def instances(cls, inputs, label=None):
        """Return the optics stage of the array model, tracing all telescopes of the array."""
        if inputs.context is None:
            raise ValueError(f"Stage {cls.name!r} needs a run context.")
        options = inputs.options_of(cls.name)
        array_model = inputs.array_models[0]
        requests = optics_requests(
            array_model,
            inputs.simulation_parameters[0],
            corsika_telescope_ids(array_model.telescope_models),
            stream_id_offset=STREAM_TELESCOPE_ID_OFFSET,
            telescope_transmission=options["telescope_transmission"],
            threads=_threads(options, inputs.args),
        )
        telescopes = [array_model.telescope_models[r.telescope_name] for r in requests]
        return [cls(requests, telescopes, files_of_run(inputs, label))]

    def prepare(self, run_number, input_file=None, output=None):
        """
        Write the optical models and return the shell line that traces the photons.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path
            Photon stream file, or ``STDIN`` when reading a stream.
        output : StreamTo or None
            Downstream command receiving the camera photons; None writes the output file.

        Returns
        -------
        list[str]
            One shell line.
        """
        if input_file is None:
            raise ValueError("The optics stage needs an input photon stream.")
        model_files = []
        for request, telescope_model in zip(self.requests, self.telescope_models):
            export_model_files(telescope_model, update_if_necessary=True)
            model_files.append(write_optical_model(request, telescope_model))
        streamed = isinstance(output, StreamTo)
        command = trace_stream_command(
            self.requests,
            model_files,
            input_file=input_file,
            output_file=STDIO if streamed else self.output_file(run_number),
        )
        resources_file = stage_file(self, "resources", run_number)
        log_file = stage_file(self, "log", run_number)
        self.file_list = {
            f"{self.name}_resources": [resources_file],
            f"{self.name}_log": [log_file],
        }
        text = _guarded(
            shlex.join(
                build_accounting_command(
                    command,
                    resources_file,
                    self.name,
                    run_number,
                    model_version=self.requests[0].model_version,
                )
            ),
            log_file,
        )
        if streamed:
            return [f"{text} | {single_downstream(output, self.name)}"]
        return [text]

    def output_file(self, run_number):
        """Return the camera photon stream file of the array."""
        return stage_file(self, "output", run_number)


def _threads(options, args):
    """Return the tracing threads: the option, one thread if sequential, else all cores."""
    if "threads" in options:
        return int(options["threads"])
    if args.get("sequential", False):
        return 1
    return os.process_cpu_count() or 1


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
