"""Simulator class for managing simulations of showers and array of telescopes."""

import gzip
import logging
import shutil
from pathlib import Path
from types import MappingProxyType

import numpy as np

from simtools.configuration import defaults
from simtools.context import RunContext
from simtools.io import table_handler
from simtools.io.file_type import product_value
from simtools.job_execution import execution
from simtools.job_execution.execution import execute_jobs, options_from_args, submit_jobs
from simtools.job_execution.job import JobSpec
from simtools.model.array_model import ArrayModel
from simtools.runners import runner_services
from simtools.runners.chain_runner import ChainRunner
from simtools.runners.stages import SimulationChain, StageInputs
from simtools.sim_events import file_info, validate_run_outputs, writer
from simtools.sim_events.metadata import build_simulation_metadata, build_standard_metadata
from simtools.simulation.configuration import get_shower_configuration
from simtools.simulation.parameters import SimulationParameters
from simtools.utils import general, value_conversion

_REDUCED_EVENT_DATA_SUFFIX = product_value("sim_telarray_event_data", "default_suffix")
_COMPRESSION_SUFFIXES = {".gz", ".zst", ".bz2", ".xz"}


def _write_reduced_event_list_batch(job_spec, context=None):
    """Write one reduced-event output file from one input-file batch."""
    input_files, output_file = job_spec[:2]
    metadata_args = job_spec[2] if len(job_spec) > 2 else {}
    model_exports = job_spec[3] if len(job_spec) > 3 else []
    generator = writer.EventDataWriter(
        input_files, file_format=metadata_args.get("simulation_file_format", "eventio")
    )
    table_handler.write_table_chunks(
        table_chunks=generator.iter_table_chunks(),
        output_file=Path(output_file),
        overwrite_existing=True,
        metadata_documents=lambda: {
            "METADATA": build_standard_metadata(metadata_args, output_file, context=context),
            "SIMULATION_METADATA": build_simulation_metadata(
                generator.get_simulation_input_metadata(), array_models=model_exports
            ),
        },
        activity_id=metadata_args.get("activity_id") if metadata_args else None,
    )


def _simtools_log_file_path():
    """Return the file path of the first file log handler in the simtools logger hierarchy."""
    logger = logging.getLogger("simtools")
    while logger:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename
        logger = logger.parent
    return None


def _gzip_file(source, destination):
    """Write a gzip-compressed copy of a file."""
    with Path(source).open("rb") as file_in, gzip.open(destination, "wb") as file_out:
        shutil.copyfileobj(file_in, file_out)


def _broadcast_to_runs(values, n_runs, name, hint):
    """Broadcast a single value to all runs or require one value per run."""
    if len(values) == 1:
        return values * n_runs
    if len(values) != n_runs:
        raise ValueError(
            f"Invalid {name} list length for direct_injection. Use one value or {hint}"
        )
    return values


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

    def __init__(self, label=None, extra_commands=None, model_reader=None, context=None):
        """Initialize Simulator class."""
        self.logger = logging.getLogger(__name__)
        self.label = label
        self.context = context or RunContext.from_environment()
        self.model_reader = model_reader or self.context.model_reader
        if model_reader is not None and model_reader is not self.context.model_reader:
            self.context = self.context.replace(model_reader=model_reader)

        self.site = self.context.args.get("site", None)
        self.model_version = self.context.args.get("model_version", None)
        self.simulation_software = self.context.args.get(
            "simulation_software", defaults.SIMULATION_SOFTWARE_DEFAULT
        )
        self.run_mode = self.context.args.get("run_mode", None)
        self.io_handler = self.context.paths

        self._extra_commands = extra_commands
        self._runtime = None
        self.run_number = self._initialize_from_tool_configuration()

        self.array_models, self.corsika_configurations = self._initialize_array_models()
        self._overwrite_flasher_photons_for_direct_injection()
        self._simulation_runner = self._initialize_simulation_runner()
        self.runner_service = runner_services.RunnerServices(
            self.simulation_parameters[0],
            "sub",
            label,
            array_model=self.array_models[0],
            context=self.context,
        )
        self.file_list = self.runner_service.load_files(self.run_number)

    @property
    def simulation_software(self):
        """The attribute simulation_software."""
        return self._simulation_software

    @simulation_software.setter
    def simulation_software(self, simulation_software):
        """
        Set the simulation software.

        Must be one of ``"sim_telarray"``, ``"corsika"`` or ``"corsika_sim_telarray"``
        (CORSIKA piped directly to sim_telarray). Raises ValueError otherwise.
        """
        if simulation_software not in defaults.SIMULATION_SOFTWARE_CHOICES:
            raise ValueError(f"Invalid simulation software: {simulation_software}")
        self._simulation_software = simulation_software.lower()
        self.chain = SimulationChain.from_software(self._simulation_software)

    def _initialize_from_tool_configuration(self):
        """Initialize run number from the tool configuration."""
        if self.context.args.get("corsika_file"):
            run_number = file_info.get_corsika_run_number(
                self.context.args["corsika_file"],
                file_format=self.context.args.get("simulation_file_format", "eventio"),
            )
        else:
            run_number = self.context.args.get("run_number_offset", 0) + self.context.args.get(
                "run_number", 1
            )
        return runner_services.validate_corsika_run_number(run_number)

    def _initialize_array_models(self):
        """Initialize array models and CORSIKA configurations (one per model version)."""
        array_model = []
        corsika_configurations = []
        self.simulation_parameters = []
        model_subdir = f"run{self.run_number:06d}"

        for version in value_conversion.ensure_list(self.model_version):
            model = ArrayModel(
                label=self.label,
                site=self.site,
                layout_name=self.context.args.get("array_layout_name"),
                array_elements=value_conversion.ensure_list(
                    self.context.args.get("telescopes", [])
                ),
                model_version=version,
                calibration_device_types=self._get_calibration_device_types(self.run_mode),
                overwrite_model_parameters=self.context.args.get("overwrite_model_parameters"),
                model_directory_subdir=model_subdir,
                model_reader=self.model_reader,
                context=self.context,
            )
            if self.chain.starts_with_showers:
                cfg = get_shower_configuration(
                    array_model=model,
                    label=self.label,
                    run_number=self.run_number,
                    context=self.context,
                )
                parameters = cfg.simulation_parameters
            else:
                cfg = None
                parameters = SimulationParameters.from_args(
                    self.context.args, self.run_number, model.site_model
                )
            self.simulation_parameters.append(parameters)
            model.initialize_seeds(parameters.zenith_angle, parameters.azimuth_angle)

            array_model.append(model)
            corsika_configurations.append(cfg)

        # chains with several stages run several model versions (one per array model)
        corsika_configurations = (
            corsika_configurations if len(self.chain.stages) > 1 else corsika_configurations[0]
        )
        return array_model, corsika_configurations

    def _initialize_simulation_runner(self):
        """Initialize the ChainRunner writing the run script of all chain stages."""
        models = slice(None) if len(self.chain.stages) > 1 else slice(0, 1)
        inputs = StageInputs(
            array_models=tuple(self.array_models[models]),
            simulation_parameters=tuple(self.simulation_parameters[models]),
            shower_configurations=tuple(value_conversion.ensure_list(self.corsika_configurations)),
            args=dict(self.context.args),
            context=self.context,
        )
        return ChainRunner(self.chain, inputs, label=self.label)

    def simulate(self):
        """Prepare and submit the run script as a job, then collect generated files."""
        self._simulation_runner.prepare_run(
            run_number=self.run_number,
            sub_script=self.runner_service.get_file_name("sub_script", self.run_number),
            input_file=self._get_corsika_file(),
            extra_commands=self._extra_commands,
        )
        self.update_file_lists()
        result = execution.run_command(
            command=self.runner_service.get_file_name("sub_script", self.run_number),
            stdout=self.runner_service.get_file_name("sub_out", self.run_number),
            stderr=self.runner_service.get_file_name("sub_err", self.run_number),
            env=self._simulation_runner.environment(),
        )
        self._runtime = result.runtime

    @classmethod
    def simulate_direct_injection_sequence(cls, label=None, context=None):
        """Run direct-injection simulations for one or several intensity settings.

        This method supports filter-wheel-like sequences by allowing
        ``number_of_events`` and ``flasher_photons`` to be provided as either
        scalars or sequences. For multi-step runs, one value can be broadcast to
        all steps, or one value per step can be provided explicitly.

        Parameters
        ----------
        label : str or None, optional
            Simulation label passed to the simulator instance of each sequence step.
        """
        context = context or RunContext.from_environment()
        base_args = dict(context.args)
        base_run_number = int(base_args.get("run_number", 1))

        events = value_conversion.parse_typed_sequence(base_args.get("number_of_events", 1), int)
        photons = value_conversion.parse_typed_sequence(base_args.get("flasher_photons"), int)
        n_runs = max(len(events), len(photons), 1)
        events = _broadcast_to_runs(
            events or [1], n_runs, "number_of_events", "one value per photon intensity."
        )
        photons = _broadcast_to_runs(
            photons or [None], n_runs, "flasher_photons", "one value per event setting."
        )

        for idx in range(n_runs):
            run_args = base_args.copy()
            run_args["run_number"] = base_run_number + idx
            run_args["number_of_events"] = int(events[idx])
            if photons[idx] is not None:
                run_args["flasher_photons"] = photons[idx]

            simulator = cls(label=label, context=context.replace(args=MappingProxyType(run_args)))
            simulator.simulate()
            simulator.validate_simulations()

    def _get_corsika_file(self):
        """Return the CORSIKA input file if the chain reads one, else None."""
        if self.chain.reads_input_file:
            return self.context.args.get("corsika_file", None)
        return None

    def get_files(self, file_type):
        """
        Get file(s) generated by simulations (empty list for an unknown file type).

        Parameters
        ----------
        file_type : str
            File type to be listed.

        Returns
        -------
        Path or list[Path]:
            Full path(s) of the output files of the given file type.
        """
        return self.file_list.get(file_type, [])

    def _files(self, *file_types):
        """Return the output files of all given file types as one flat list."""
        return [
            file
            for file_type in file_types
            for file in value_conversion.ensure_list(self.get_files(file_type=file_type))
        ]

    def _make_resources_report(self):
        """Return a summary string of the wall time and number of events per run."""
        resources = self._simulation_runner.get_resources(runtime=self._runtime)
        runtimes = [resources["runtime"]] if resources.get("runtime") is not None else []
        mean_runtime = np.mean(runtimes) if runtimes else np.nan

        resource_summary = f"Mean wall time/run [sec]: {mean_runtime}"
        if resources.get("n_events", 0) > 0:
            resource_summary += f", #events/run: {resources['n_events']}"
        return resource_summary

    def report(self):
        """Report on the simulations and the computing resources used."""
        parameters = self.simulation_parameters[0]
        self.logger.info(
            f"Production run complete for primary {parameters.primary_particle} showers "
            f"from {parameters.azimuth_angle} azimuth and "
            f"{parameters.zenith_angle} zenith "
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
            path.write_text("\n".join(map(str, files)) + "\n", encoding="utf-8")

    def save_reduced_event_lists(self):
        """
        Save reduced event lists with event data on simulated and triggered events.

        The files are saved using the configured simulation file list entries.
        """
        if not self.chain.has("sim_telarray"):
            self.logger.warning(
                "Reduced event lists can only be saved for sim_telarray simulations."
            )
            return

        Simulator.write_reduced_event_lists(
            input_files=self._files("sim_telarray_output"),
            output_files=self._files("sim_telarray_event_data"),
            array_models=self.array_models,
            context=self.context,
        )

    @staticmethod
    def write_reduced_event_lists(
        input_files=None,
        output_path=None,
        output_files=None,
        input_file_list=None,
        files_per_reduced_event_file=1,
        max_workers=1,
        backend="local",
        backend_config=None,
        wait_for_completion=True,
        array_models=None,
        input_file_list_pattern=None,
        file_format="eventio",
        context=None,
    ):
        """
        Write reduced event lists using the selected simulation-file reader.

        Input files are passed directly or read from text files (one file per line),
        processed in batches with one output file written per batch.

        Parameters
        ----------
        input_files : list
            Simulation files in the selected input format.
        output_path : str or Path, optional
            Output directory (default: the directory of each input file).
        output_files : list, optional
            Explicit output files, zipped with the input file batches.
        input_file_list : str or Path, optional
            Text file containing one simulation file per line.
        files_per_reduced_event_file : int, optional
            Number of input files combined into each output file (default 1).
        max_workers : int, optional
            Output-file worker processes (None: 60% of CPUs, 0: all CPUs, 1: serial).
        array_models : list, optional
            ``ArrayModel`` instances exported into the simulation metadata.
        input_file_list_pattern : str, optional
            Glob pattern of text files with one simulation file per line; each list
            is processed independently, all jobs share one execution submission.
        backend : str, optional
            Execution backend (default ``"local"``).
        backend_config : dict, optional
            Backend-specific execution settings.
        wait_for_completion : bool, optional
            Wait for all backend jobs; submit-only mode suits asynchronous backends.
        context : RunContext, optional
            Run context; its arguments build the embedded standard metadata.

        Returns
        -------
        SubmissionHandle or None
            Handle for submit-only execution; blocking execution returns ``None``.

        Raises
        ------
        ValueError
            If the input parameters or number of explicit output files are invalid.
        """
        sources = (input_files, input_file_list, input_file_list_pattern)
        if sum(source is not None for source in sources) != 1:
            raise ValueError(
                "Provide exactly one of input_files, input_file_list, or input_file_list_pattern."
            )
        if files_per_reduced_event_file < 1:
            raise ValueError("files_per_reduced_event_file must be greater than zero.")

        input_file_batches = []
        derived_output_files = []
        for (
            resolved_input_files,
            input_file_list_path,
        ) in Simulator._resolve_reduced_event_input_groups(
            input_files=input_files,
            input_file_list=input_file_list,
            input_file_list_pattern=input_file_list_pattern,
        ):
            batches = [
                resolved_input_files[index : index + files_per_reduced_event_file]
                for index in range(0, len(resolved_input_files), files_per_reduced_event_file)
            ]
            input_file_batches.extend(batches)
            derived_output_files.extend(
                Simulator._build_output_files(output_path, batches, input_file_list_path)
            )

        resolved_output_files = output_files if output_files is not None else derived_output_files
        Simulator._validate_reduced_event_output_count(
            input_file_batches=input_file_batches,
            output_files=resolved_output_files,
        )
        for output_file in resolved_output_files:
            runner_services.warn_if_file_name_exceeds_dirac_limit(
                Path(output_file).name, "sim_telarray_event_data"
            )

        return Simulator._run_reduced_event_list_jobs(
            input_file_batches=input_file_batches,
            output_files=resolved_output_files,
            max_workers=max_workers,
            backend=backend,
            backend_config=backend_config,
            wait_for_completion=wait_for_completion,
            metadata_args={
                **(context.args if context is not None else {}),
                "simulation_file_format": file_format,
            },
            array_models=array_models,
            context=context,
        )

    @staticmethod
    def _resolve_reduced_event_input_groups(input_files, input_file_list, input_file_list_pattern):
        """Resolve direct input files or one or more input-file lists into (files, list path)."""
        if input_files is not None:
            return [(input_files, None)]

        if input_file_list_pattern is None:
            list_paths = [Path(input_file_list)]
        else:
            matches = sorted(
                path
                for path in general.resolve_file_patterns(input_file_list_pattern)
                if Path(path).is_file()
            )
            if not matches:
                raise FileNotFoundError(
                    f"No input file lists found for pattern: {input_file_list_pattern}"
                )
            list_paths = [Path(path) for path in matches]

        groups = []
        for list_path in list_paths:
            with open(list_path, encoding="utf-8") as file_list:
                resolved_input_files = [line.strip() for line in file_list if line.strip()]
            groups.append((resolved_input_files, list_path))
        return groups

    @staticmethod
    def _build_output_files(output_path, input_file_batches, input_file_list=None):
        """Build output names: one per input batch, or from the file-list stem and part index."""
        if input_file_list is None:
            return [
                Simulator._build_output_file_for_input_batch(output_path, batch)
                for batch in input_file_batches
            ]
        output_dir = Path(output_path) if output_path else input_file_list.parent
        return [
            output_dir / f"{input_file_list.stem}.part{index:04d}{_REDUCED_EVENT_DATA_SUFFIX}"
            for index in range(1, len(input_file_batches) + 1)
        ]

    @staticmethod
    def _build_output_file_for_input_batch(output_path, input_file_batch):
        """Build output filename for a single input batch."""
        input_file = Path(input_file_batch[0])
        output_dir = Path(output_path) if output_path else input_file.parent
        if input_file.suffix in _COMPRESSION_SUFFIXES:
            input_file = input_file.with_suffix("")
        return output_dir / f"{input_file.stem}{_REDUCED_EVENT_DATA_SUFFIX}"

    @staticmethod
    def _validate_reduced_event_output_count(input_file_batches, output_files):
        """Validate that output-file and batch counts match."""
        if len(output_files) != len(input_file_batches):
            raise ValueError(
                "Length mismatch between input_files and output_files: "
                f"{len(input_file_batches)} batch(es), {len(output_files)} output file(s)."
            )
        if len(set(map(Path, output_files))) != len(output_files):
            raise ValueError("Reduced-event output files must be unique.")

    @staticmethod
    def _run_reduced_event_list_jobs(
        input_file_batches,
        output_files,
        max_workers,
        backend="local",
        backend_config=None,
        wait_for_completion=True,
        metadata_args=None,
        array_models=None,
        context=None,
    ):
        """Run reduced event list jobs serially or in parallel."""
        context = context or RunContext.from_environment()
        model_exports = [model.to_simulation_metadata_dict() for model in (array_models or [])]
        execution_jobs = [
            JobSpec(
                job_id=f"job-{index:06d}",
                index=index,
                function=_write_reduced_event_list_batch,
                item=(input_files, output_file, metadata_args or {}, model_exports),
                context_spec=context.to_spec(),
                output_paths=(Path(output_file),),
            )
            for index, (input_files, output_file) in enumerate(
                zip(input_file_batches, output_files)
            )
        ]
        options = options_from_args(
            {"backend": backend, "backend_config": backend_config},
            max_workers=max_workers,
            work_dir=Path(output_files[0]).parent if output_files else None,
        )
        if not wait_for_completion:
            return submit_jobs(execution_jobs, options)
        execute_jobs(execution_jobs, options)
        return None

    def pack_for_register(self, directory_for_grid_upload=None):
        """
        Prepare simulation output files for registering on the grid.

        Parameters
        ----------
        directory_for_grid_upload : str or pathlib.Path or None, optional
            Output directory; defaults to the ``directory_for_grid_upload`` subdirectory.
        """
        self.logger.info(
            f"Packing output files for registering on the grid ({directory_for_grid_upload})"
        )
        simtools_log_file = _simtools_log_file_path()
        files_to_copy = self._files(
            "sim_telarray_log",
            "sim_telarray_histogram",
            "corsika_log",
            "corsika_resources",
            "sim_telarray_resources",
            "multi_pipe_resources",
            "sim_telarray_err",
            "corsika_err",
        )
        for model in self.array_models:
            files_to_copy += value_conversion.ensure_list(model.pack_model_files())

        files_to_move = self._files("sim_telarray_output")
        if self.context.args.get("reduced_event_lists"):
            files_to_move += self._files("sim_telarray_event_data")
        if self.context.args.get("save_corsika_output"):
            files_to_move += self._files("corsika_output")

        directory_for_grid_upload = (
            Path(directory_for_grid_upload)
            if directory_for_grid_upload
            else self.io_handler.get_output_directory().joinpath("directory_for_grid_upload")
        )
        directory_for_grid_upload.mkdir(parents=True, exist_ok=True)

        for file_to_copy in files_to_copy:
            self._place_file(
                file_to_copy, directory_for_grid_upload / Path(file_to_copy).name, shutil.copy2
            )

        if simtools_log_file and Path(simtools_log_file).exists():
            self._place_file(
                simtools_log_file,
                directory_for_grid_upload / f"{Path(simtools_log_file).name}.gz",
                _gzip_file,
            )

        for file_to_move in files_to_move:
            self._place_file(
                file_to_move, directory_for_grid_upload / Path(file_to_move).name, shutil.move
            )

        self.logger.info(f"Grid output files grid placed in {directory_for_grid_upload!s}")

    def _place_file(self, source, destination, operation):
        """Copy or move a file to its destination, warning about overwrites."""
        if destination.exists():
            self.logger.warning(f"Overwriting existing file: {destination}")
        operation(source, destination)

    @staticmethod
    def _get_calibration_device_types(run_mode):
        """Return the calibration device types for a run mode."""
        return ["flat_fielding"] if run_mode == "direct_injection" else []

    def _overwrite_flasher_photons_for_direct_injection(self):
        """Overwrite direct-injection photons-per-pixel in calibration models."""
        flasher_photons = self.context.args.get("flasher_photons")
        if self.run_mode != "direct_injection" or flasher_photons is None:
            return

        for array_model in value_conversion.ensure_list(self.array_models):
            for calibration_models in array_model.calibration_models.values():
                for calibration_model in calibration_models.values():
                    calibration_model.overwrite_model_parameter(
                        "flasher_photons_at_pixel", flasher_photons
                    )

    def validate_simulations(self):
        """Validate data, log and metadata of CORSIKA, sim_telarray and reduced event lists."""
        parameters = self.simulation_parameters[0]
        expected_shower_events = parameters.shower_events
        expected_mc_events = parameters.mc_events
        self.logger.info(
            "Validating simulations "
            f"with {expected_mc_events} MC events and {expected_shower_events} shower events."
        )

        if self.chain.has("sim_telarray"):
            validate_run_outputs.validate_sim_telarray(
                data_files=self.get_files(file_type="sim_telarray_output"),
                log_files=self.get_files(file_type="sim_telarray_log"),
                array_models=self.array_models,
                expected_mc_events=expected_mc_events,
                expected_shower_events=expected_shower_events,
                curved_atmo=parameters.use_curved_atmosphere,
                allow_for_changes=["nsb_scaling_factor", "stars"],
            )
        if self.chain.has("corsika"):
            validate_run_outputs.validate_corsika_output(
                data_files=self.get_files(file_type="corsika_output")
                if self.chain.stages == ("corsika",) or self.context.args.get("save_corsika_output")
                else None,
                log_files=self.get_files(file_type="corsika_log"),
                expected_shower_events=expected_shower_events,
                curved_atmo=parameters.use_curved_atmosphere,
            )

        if self.context.args.get("reduced_event_lists"):
            validate_run_outputs.validate_sim_events(
                data_files=self.get_files(file_type="sim_telarray_event_data"),
                expected_mc_events=expected_mc_events,
            )

    def update_file_lists(self):
        """Update file lists with the data, log, histogram and submission files."""
        if self.file_list is None:
            self.file_list = self._simulation_runner.file_list
        else:
            self.file_list.update(self._simulation_runner.file_list)
