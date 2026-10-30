"""CORSIKA as a simulation stage (air showers and Cherenkov photons)."""

import logging
import shlex
import stat
from pathlib import Path

from simtools.job_execution.process_accounting import build_accounting_command
from simtools.runners.runner_services import RunnerServices
from simtools.runners.stages import CHERENKOV_PHOTONS, StreamTo


class CorsikaStage:
    """
    CORSIKA air-shower simulation stage.

    Writes the CORSIKA input card and the commands that run CORSIKA. A streamed output is
    passed to the downstream commands with ``multipipe_corsika``.

    Parameters
    ----------
    corsika_config : CorsikaConfig
        CORSIKA configuration.
    label : str
        Instance label.
    sequential : bool
        Run multipipe_corsika in sequential mode (no parallel CORSIKA/downstream execution).
    save_output : bool
        Also save the streamed CORSIKA output to a compressed file.
    """

    name = "corsika"
    consumes = None
    produces = CHERENKOV_PHOTONS
    streams = True

    def __init__(
        self, corsika_config, label=None, sequential=False, save_output=False, context=None
    ):
        """Initialize CorsikaStage."""
        self._logger = logging.getLogger(__name__)
        self.corsika_config = corsika_config
        self.label = label
        self.sequential = sequential
        self.save_output = save_output
        self.context = context or corsika_config.context
        self.runner_service = RunnerServices(
            corsika_config, run_type="corsika", label=label, context=self.context
        )
        self.stream_service = RunnerServices(
            corsika_config, run_type="multi_pipe", label=label, context=self.context
        )
        self.file_list = None

    @classmethod
    def instances(cls, inputs, label=None):
        """Return the CORSIKA stage of a run (the first shower configuration defines it)."""
        return [
            cls(
                inputs.shower_configurations[0],
                label=label,
                sequential=inputs.args.get("sequential", False),
                save_output=inputs.args.get("save_corsika_output", False),
                context=inputs.context,
            )
        ]

    def prepare(self, run_number, input_file=None, output=None):
        """
        Write the CORSIKA input card (and stream configuration) and return the run lines.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : None
            CORSIKA reads no input product.
        output : StreamTo or None
            Downstream commands receiving the CORSIKA output stream; None writes the
            CORSIKA output file.

        Returns
        -------
        list[str]
            Shell lines running CORSIKA.
        """
        if input_file is not None:
            raise ValueError("CORSIKA does not read an input product.")
        self.file_list = self.runner_service.load_files(run_number=run_number)
        streamed = isinstance(output, StreamTo)
        if streamed:
            self.file_list.update(self.stream_service.load_files(run_number=run_number))
            corsika_output = self._write_stream(run_number, output)
        else:
            corsika_output = self.runner_service.get_file_name(
                "corsika_output", run_number=run_number
            )
        corsika_input = self.runner_service.get_file_name("corsika_input", run_number=run_number)
        self.corsika_config.generate_corsika_input_file(streamed, corsika_input, corsika_output)

        run_dir = self.runner_service.get_file_name("corsika_output", run_number=run_number).parent
        accounting_command = build_accounting_command(
            [self._corsika_executable()],
            self.runner_service.get_file_name("corsika_resources", run_number=run_number),
            "corsika",
            run_number,
            model_version=self.corsika_config.array_model.model_version,
            log_file=self.runner_service.get_file_name("corsika_log", run_number=run_number),
            input_file=corsika_input,
        )
        return [
            f"export CORSIKA_DATA={run_dir}",
            'mkdir -p "$CORSIKA_DATA"',
            'cd "$CORSIKA_DATA" || exit 2',
            "",
            "# Running corsika",
            shlex.join(accounting_command),
        ]

    def _write_stream(self, run_number, output):
        """Write the multipipe configuration and script; return the script path."""
        multipipe_file = self.stream_service.get_file_name(
            "multi_pipe_config", run_number=run_number
        )
        with open(multipipe_file, "w", encoding="utf-8") as file:
            for command in output.commands:
                file.write(f"{command}\n\n")
            if self.save_output:
                corsika_output = self.runner_service.get_file_name(
                    "corsika_output", run_number=run_number
                )
                file.write(f"zstd > {corsika_output}\n")
        self._logger.info(f"Multipipe script: {multipipe_file}")

        multipipe_script = self.stream_service.get_file_name(
            "multi_pipe_script", run_number=run_number
        )
        multipipe_command = [
            self.context.software.sim_telarray_path / "bin/multipipe_corsika",
            "-c",
            multipipe_file,
        ]
        if self.sequential:
            multipipe_command.append("--sequential")
        accounting_command = build_accounting_command(
            multipipe_command,
            self.stream_service.get_file_name("multi_pipe_resources", run_number=run_number),
            "multipipe",
            run_number,
            model_version=self.corsika_config.array_model.model_version,
        )
        multipipe_script.write_text(shlex.join(accounting_command), encoding="utf-8")
        multipipe_script.chmod(multipipe_script.stat().st_mode | stat.S_IEXEC)
        return multipipe_script

    def _corsika_executable(self):
        """Return the CORSIKA executable for the configured atmosphere geometry."""
        if self.corsika_config.use_curved_atmosphere:
            geometry = "curved"
            executable = Path(self.context.software.corsika_exe_curved)
        else:
            geometry = "flat"
            executable = Path(self.context.software.corsika_exe)

        he_model, le_model = self.context.software.corsika_interaction_models
        transition_energy = self.context.args.get("corsika_hadronic_transition_energy")
        transition_description = (
            "CORSIKA default"
            if transition_energy is None
            else f"{transition_energy.to_value('GeV')} GeV"
        )
        self._logger.info(
            "Resolved CORSIKA configuration: HE=%s LE=%s HILOW=%s geometry=%s executable=%s",
            he_model,
            le_model,
            transition_description,
            geometry,
            executable,
        )
        return executable

    def output_file(self, run_number):
        """Return the CORSIKA output file."""
        return self.runner_service.get_file_name("corsika_output", run_number=run_number)

    def environment(self):
        """Return the CORSIKA interaction-model environment variables."""
        he_model, le_model = self.context.software.corsika_interaction_models
        environment = {
            "SIMTOOLS_CORSIKA_HE_INTERACTION": he_model,
            "SIMTOOLS_CORSIKA_LE_INTERACTION": le_model,
        }
        transition_energy = self.context.args.get("corsika_hadronic_transition_energy")
        if transition_energy is not None:
            environment["SIMTOOLS_CORSIKA_HADRONIC_TRANSITION_ENERGY_GEV"] = str(
                transition_energy.to_value("GeV")
            )
        return environment

    def get_resources(self, runtime=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(runtime=runtime)
