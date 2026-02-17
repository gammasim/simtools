"""Simulator class for managing simulations of showers and array of telescopes."""

import logging
import shutil
from pathlib import Path

import numpy as np
from astropy import units as u

from simtools import settings
from simtools.corsika import corsika_output_validator
from simtools.corsika.corsika_config import CorsikaConfig
from simtools.io import io_handler, table_handler
from simtools.job_execution import job_manager
from simtools.model.array_model import ArrayModel
from simtools.runners import corsika_runner, corsika_simtel_runner, runner_services, simtel_runner
from simtools.sim_events import file_info, output_validator, writer
from simtools.simtel import simtel_output_validator
from simtools.simtel.simulator_array import SimulatorArray
from simtools.utils import general


class Simulator:
    """
    Simulation of showers and of the array of telescopes.

    Interface with the simulation software packages (e.g., CORSIKA or sim_telarray).
    A single run is simulated per instance, possibly for multiple model versions.

    Parameters
    ----------
    label: str
        Instance label.
    extra_commands: str or list of str
        Extra commands to be added to the run script before the run command.
    """

    def __init__(self, label=None, extra_commands=None):
        """Initialize Simulator class."""
        self.logger = logging.getLogger(__name__)
        self.label = label

        self.site = settings.config.args.get("site", None)
        self.model_version = settings.config.args.get("model_version", None)

        self.simulation_software = settings.config.args.get(
            "simulation_software", "corsika_sim_telarray"
        )
        self.run_mode = settings.config.args.get("run_mode", None)

        self.io_handler = io_handler.IOHandler()

        self._extra_commands = extra_commands
        self.run_number = self._initialize_from_tool_configuration()

        self.array_models, self.corsika_configurations = self._initialize_array_models()
        self._overwrite_flasher_photons_for_direct_injection()
        self._simulation_runner = self._initialize_simulation_runner()
        self.runner_service = runner_services.RunnerServices(
            self._get_first_corsika_config(),
            "sub",
            label,
        )
        self.file_list = self.runner_service.load_files(self.run_number)

    @property
    def simulation_software(self):
        """The attribute simulation_software."""
        return self._simulation_software

    @simulation_software.setter
    def simulation_software(self, simulation_software):
        """
        Set and test simulation_software type.

        Parameters
        ----------
        simulation_software: choices: [sim_telarray, corsika, corsika_sim_telarray]
            implemented are sim_telarray and CORSIKA or corsika_sim_telarray
            (running CORSIKA and piping it directly to sim_telarray)

        Raises
        ------
        ValueError

        """
        if simulation_software not in ["sim_telarray", "corsika", "corsika_sim_telarray"]:
            raise ValueError(f"Invalid simulation software: {simulation_software}")
        self._simulation_software = simulation_software.lower()

    def _initialize_from_tool_configuration(self):
        """Initialize simulator from tool configuration."""
        if settings.config.args.get("corsika_file"):
            run_number = file_info.get_corsika_run_number(settings.config.args["corsika_file"])
        else:
            run_number = settings.config.args.get(
                "run_number_offset", 0
            ) + settings.config.args.get("run_number", 1)
        return runner_services.validate_corsika_run_number(run_number)

    def _initialize_array_models(self):
        """
        Initialize array simulation models and CORSIKA config (one per model version).

        Returns
        -------
        list, list
            List of ArrayModel and CorsikaConfig objects.
        """
        versions = general.ensure_iterable(self.model_version)

        array_model = []
        corsika_configurations = []

        for version in versions:
            model = ArrayModel(
                label=self.label,
                site=self.site,
                layout_name=settings.config.args.get("array_layout_name"),
                model_version=version,
                calibration_device_types=self._get_calibration_device_types(self.run_mode),
                overwrite_model_parameters=settings.config.args.get("overwrite_model_parameters"),
            )
            cfg = CorsikaConfig(array_model=model, label=self.label, run_number=self.run_number)
            model.initialize_seeds(cfg.zenith_angle, cfg.azimuth_angle)

            array_model.append(model)
            corsika_configurations.append(cfg)

        #  'corsika_sim_telarray' allows for multiple model versions (multipipe option)
        corsika_configurations = (
            corsika_configurations
            if self.simulation_software == "corsika_sim_telarray"
            else corsika_configurations[0]
        )

        return array_model, corsika_configurations

    def _initialize_simulation_runner(self):
        """
        Initialize corsika configuration and simulation runners.

        Returns
        -------
        CorsikaRunner or SimulatorArray or CorsikaSimtelRunner
            Simulation runner object.
        """
        runner_class = {
            "corsika": corsika_runner.CorsikaRunner,
            "sim_telarray": SimulatorArray,
            "corsika_sim_telarray": corsika_simtel_runner.CorsikaSimtelRunner,
        }.get(self.simulation_software)

        runner_args = {
            "label": self.label,
            "corsika_config": self.corsika_configurations,
        }

        if runner_class is not SimulatorArray:
            runner_args["curved_atmosphere_min_zenith_angle"] = settings.config.args.get(
                "curved_atmosphere_min_zenith_angle", 65 * u.deg
            )
        if runner_class is corsika_simtel_runner.CorsikaSimtelRunner:
            runner_args["sequential"] = settings.config.args.get("sequential", False)

        return runner_class(**runner_args)

    def simulate(self):
        """
        Prepare and submit a run script as a job.

        Writes submission scripts using the simulation runners and submits the
        run script to the job manager. Collects generated files.
        """
        self._simulation_runner.prepare_run(
            run_number=self.run_number,
            corsika_file=self._get_corsika_file(),
            sub_script=self.runner_service.get_file_name("sub_script", self.run_number),
            extra_commands=self._extra_commands,
        )
        self.update_file_lists()

        job_manager.submit(
            command=self.runner_service.get_file_name("sub_script", self.run_number),
            out_file=self.runner_service.get_file_name("sub_out", self.run_number),
            err_file=self.runner_service.get_file_name("sub_err", self.run_number),
            env=simtel_runner.SIM_TELARRAY_ENV,
        )

    def _get_corsika_file(self):
        """
        Get the CORSIKA input file if applicable (for sim_telarray simulations).

        Returns
        -------
        Path, None
            Path to the CORSIKA input file.
        """
        if self.simulation_software == "sim_telarray":
            return settings.config.args.get("corsika_file", None)
        return None

    def get_files(self, file_type):
        """
        Get file(s) generated by simulations.

        Not all file types are available for all simulation types.
        Returns an empty list for an unknown file type.

        Parameters
        ----------
        file_type : str
            File type to be listed.

        Returns
        -------
        Path or list[Path]:
            File or list with the full path of output files of a certain file type.

        """
        return self.file_list[file_type]

    def _make_resources_report(self):
        """
        Prepare a simple report on computing wall clock time used in the simulations.

        Returns
        -------
        str
           string reporting on computing resources

        """
        runtime = []
        _resources = self._simulation_runner.get_resources(self.get_files(file_type="sub_out"))
        if _resources.get("runtime"):
            runtime.append(_resources["runtime"])

        mean_runtime = np.mean(runtime)

        resource_summary = f"Mean wall time/run [sec]: {mean_runtime}"
        if "n_events" in _resources and _resources["n_events"] > 0:
            resource_summary += f", #events/run: {_resources['n_events']}"

        return resource_summary

    def report(self):
        """
        Report on simulations and computing resources used.

        Includes run time per run only at this point.
        """
        _corsika_config = self._get_first_corsika_config()
        self.logger.info(
            f"Production run complete for primary {_corsika_config.primary_particle} showers "
            f"from {_corsika_config.azimuth_angle} azimuth and "
            f"{_corsika_config.zenith_angle} zenith "
            f"at {self.site} site, using {self.model_version} model."
        )
        self.logger.info(
            f"Computing for {self.simulation_software} Simulations: {self._make_resources_report()}"
        )

    def save_file_lists(self):
        """Save file lists for output and log files."""
        outdir = self.io_handler.get_output_directory()

        for file_type, files in self.file_list.items():
            if not files or any(f is None for f in files):
                self.logger.debug(f"No files to save for {file_type} files.")
                continue

            path = outdir / f"{file_type}_files.txt"
            self.logger.info(f"Saving list of {file_type} files to {path}")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(map(str, files)) + "\n")

    def save_reduced_event_lists(self):
        """
        Save reduced event lists with event data on simulated and triggered events.

        The files are saved with the same name as the sim_telarray output file
        but with a 'hdf5' extension.
        """
        if "sim_telarray" not in self.simulation_software:
            self.logger.warning(
                "Reduced event lists can only be saved for sim_telarray simulations."
            )
            return

        input_files = self.get_files(file_type="sim_telarray_output")
        output_files = self.get_files(file_type="sim_telarray_event_data")
        for input_file, output_file in zip(input_files, output_files):
            generator = writer.EventDataWriter([input_file])
            table_handler.write_tables(
                tables=generator.process_files(),
                output_file=Path(output_file),
                overwrite_existing=True,
            )

    def pack_for_register(self, directory_for_grid_upload=None):
        """
        Pack simulation output files for registering on the grid.

        Creates separate tarballs for each model version's log files.

        Parameters
        ----------
        directory_for_grid_upload: str
            Directory for the tarball with output files.

        """
        self.logger.info(
            f"Packing output files for registering on the grid ({directory_for_grid_upload})"
        )
        output_files = self.get_files(file_type="sim_telarray_output")
        log_files = self.get_files(file_type="sim_telarray_log")
        simtools_log_file = general.get_simtools_log_file()
        histogram_files = self.get_files(file_type="sim_telarray_histogram")
        reduced_event_files = (
            self.get_files(file_type="sim_telarray_event_data")
            if settings.config.args.get("save_reduced_event_lists")
            else []
        )

        directory_for_grid_upload = (
            Path(directory_for_grid_upload)
            if directory_for_grid_upload
            else self.io_handler.get_output_directory().joinpath("directory_for_grid_upload")
        )
        directory_for_grid_upload.mkdir(parents=True, exist_ok=True)

        # Group files by model version
        for model in self.array_models:
            model_version = model.model_version
            model_logs = [f for f in log_files if model_version in str(f)]

            if not model_logs:
                continue

            tar_name = Path(model_logs[0]).name.replace("simtel.log.gz", "log_hist.tar.gz")
            tar_path = directory_for_grid_upload / tar_name

            files_to_tar = (
                model_logs
                + [f for f in histogram_files if model_version in str(f)]
                + [str(self.get_files(file_type="corsika_log"))]
                + list(general.ensure_iterable(model.pack_model_files()))
            )
            # simtools log file duplicated for each model version
            if simtools_log_file and Path(simtools_log_file).exists():
                files_to_tar.append(str(simtools_log_file))
            general.pack_tar_file(tar_path, files_to_tar)

        for file_to_move in output_files + reduced_event_files:
            destination_file = directory_for_grid_upload / Path(file_to_move).name
            if destination_file.exists():
                self.logger.warning(f"Overwriting existing file: {destination_file}")
            shutil.move(file_to_move, destination_file)

        self.logger.info(f"Grid output files grid placed in {directory_for_grid_upload!s}")

    @staticmethod
    def _get_calibration_device_types(run_mode):
        """
        Get the list of calibration device types based on the run mode.

        Parameters
        ----------
        run_mode: str
            Run mode of the simulation.

        Returns
        -------
        list
            List of calibration device types.
        """
        if run_mode == "direct_injection":
            return ["flat_fielding"]
        return []

    def _overwrite_flasher_photons_for_direct_injection(self):
        """Overwrite flasher_photons in calibration models for direct-injection runs."""
        flasher_photons = settings.config.args.get("flasher_photons")
        if self.run_mode != "direct_injection" or flasher_photons is None:
            return

        for array_model in general.ensure_iterable(self.array_models):
            for calibration_models in array_model.calibration_models.values():
                for calibration_model in calibration_models.values():
                    calibration_model.overwrite_model_parameter("flasher_photons", flasher_photons)

    def _get_first_corsika_config(self):
        """
        Return first instance from list of CORSIKA configurations.

        Most values stored in the CORSIKA configurations are identical,
        with the exception of the simulation model version dependent parameters.

        Returns
        -------
        CorsikaConfig
            First CORSIKA configuration instance.
        """
        try:
            return (
                self.corsika_configurations[0]
                if isinstance(self.corsika_configurations, list)
                else self.corsika_configurations
            )
        except (IndexError, TypeError) as exc:
            raise ValueError("CORSIKA configuration not found for verification.") from exc

    def validate_simulations(self):
        """
        Validate simulations data and metadata.

        Validates data, log, and metadata files from CORSIKA, sim_telarray and reduced event lists
        (if saved).
        """
        _corsika_config = self._get_first_corsika_config()
        expected_shower_events = _corsika_config.shower_events
        expected_mc_events = _corsika_config.mc_events
        self.logger.info(
            "Validating simulations "
            f"with {expected_mc_events} MC events and {expected_shower_events} shower events."
        )

        if "sim_telarray" in self.simulation_software:
            simtel_output_validator.validate_sim_telarray(
                data_files=self.get_files(file_type="sim_telarray_output"),
                log_files=self.get_files(file_type="sim_telarray_log"),
                array_models=self.array_models,
                expected_mc_events=expected_mc_events,
                expected_shower_events=expected_shower_events,
                curved_atmo=_corsika_config.use_curved_atmosphere,
                allow_for_changes=["nsb_scaling_factor", "stars"],
            )
        if "corsika" in self.simulation_software:
            corsika_output_validator.validate_corsika_output(
                data_files=self.get_files(file_type="corsika_output")
                if self.simulation_software == "corsika"
                else None,
                log_files=self.get_files(file_type="corsika_log"),
                expected_shower_events=expected_shower_events,
                curved_atmo=_corsika_config.use_curved_atmosphere,
            )

        if settings.config.args.get("save_reduced_event_lists"):
            output_validator.validate_sim_events(
                data_files=self.get_files(file_type="sim_telarray_event_data"),
                expected_mc_events=expected_mc_events,
            )

    def update_file_lists(self):
        """
        Update file lists with all data, log, histogram and submission files.

        Some of these files are generated by the simulation runners.
        """
        if self.file_list is None:
            self.file_list = self._simulation_runner.file_list
        else:
            self.file_list.update(self._simulation_runner.file_list)
