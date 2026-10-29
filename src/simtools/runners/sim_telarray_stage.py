"""sim_telarray as a simulation stage (telescope and camera response)."""

import shlex

from simtools.job_execution.process_accounting import build_accounting_command
from simtools.runners.simtel_runner import SIM_TELARRAY_ENV, sim_telarray_env_as_string
from simtools.runners.stages import CAMERA_EVENTS, CHERENKOV_PHOTONS, STDIN
from simtools.simtel.simulator_array import SimulatorArray


class SimTelArrayStage:
    """
    sim_telarray array simulation stage, one instance per array model.

    Parameters
    ----------
    simulation_parameters : SimulationParameters
        Physical run settings.
    array_model : ArrayModel
        Telescope and site models.
    label : str
        Instance label.
    """

    name = "sim_telarray"
    consumes = CHERENKOV_PHOTONS
    produces = CAMERA_EVENTS
    streams = True

    def __init__(self, simulation_parameters, array_model, label=None, context=None):
        """Initialize SimTelArrayStage."""
        self.simulator_array = SimulatorArray(
            simulation_parameters=simulation_parameters,
            array_model=array_model,
            label=label,
            context=context,
        )
        self.array_model = array_model

    @classmethod
    def instances(cls, inputs, label=None):
        """Return one sim_telarray stage per array model."""
        return [
            cls(
                simulation_parameters=parameters,
                array_model=array_model,
                label=label,
                context=inputs.context,
            )
            for parameters, array_model in zip(inputs.simulation_parameters, inputs.array_models)
        ]

    @property
    def file_list(self):
        """Files of this stage by file type."""
        return self.simulator_array.file_list

    @property
    def runner_service(self):
        """File naming of this stage."""
        return self.simulator_array.runner_service

    def prepare(self, run_number, input_file=None, output=None):
        """
        Write the sim_telarray configuration and return the run lines.

        Parameters
        ----------
        run_number : int
            Run number.
        input_file : str or Path
            CORSIKA output file, or ``STDIN`` when reading a stream.
        output : None
            sim_telarray writes its output file.

        Returns
        -------
        list[str]
            Shell lines running sim_telarray (one per run of a run set; one line when
            reading a stream).
        """
        if output is not None:
            raise ValueError("sim_telarray output can only be written to its file.")
        command = self.simulator_array.make_run_command(
            run_number=run_number, input_file=input_file
        )
        resources_file = self.runner_service.get_file_name(
            file_type="sim_telarray_resources", run_number=run_number
        )
        log_file = self.runner_service.get_file_name(
            file_type="sim_telarray_log", run_number=run_number
        )
        streamed = input_file == STDIN
        runs = 1 if streamed else self.simulator_array.runs_per_set
        resource_files = [
            resources_file.with_name(
                f"{resources_file.stem}.part{run_index + 1:04d}{resources_file.suffix}"
            )
            if runs > 1
            else resources_file
            for run_index in range(runs)
        ]
        lines = [
            f"{sim_telarray_env_as_string()} "
            + shlex.join(
                build_accounting_command(
                    command,
                    process_resources_file,
                    "sim_telarray",
                    run_number,
                    model_version=self.array_model.model_version,
                    log_file=log_file,
                    input_file=None if streamed else input_file,
                )
            )
            for process_resources_file in resource_files
        ]
        self.file_list["sim_telarray_resources"] = (
            resource_files[0] if len(resource_files) == 1 else resource_files
        )
        return lines

    def output_file(self, run_number):
        """Return the sim_telarray output file."""
        return self.runner_service.get_file_name("sim_telarray_output", run_number=run_number)

    def environment(self):
        """Return the sim_telarray environment variables."""
        return dict(SIM_TELARRAY_ENV)

    def get_resources(self, runtime=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(runtime=runtime)
