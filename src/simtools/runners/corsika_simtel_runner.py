"""Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality.

If the ray-tracing backend traces air-shower photons, the photon bunches of CORSIKA are routed
through the backend (``simtools.simtel.shower_route``) before sim_telarray reads them. In
``stream`` mode the route is one process per telescope type in the multipipe, piped into
sim_telarray. In ``disk`` mode the CORSIKA IACT file is kept; the route and sim_telarray run
after the multipipe, on the IACT file and on the camera photon-bunch file of each telescope type.
"""

import json
import logging
import shlex
import stat
import sys
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools import settings
from simtools.corsika.corsika_config import corsika_telescope_ids
from simtools.job_execution.process_accounting import build_accounting_command
from simtools.ray_tracing.backends.interface import create_shower_tracer, get_backend
from simtools.runners import corsika_runner, runner_services, simtel_runner
from simtools.simtel.photon_bunch_file import bypass_optics_level
from simtools.simtel.shower_photons import minimum_input_plane_height_m
from simtools.simtel.shower_route import DEFAULT_SEED
from simtools.simtel.simulator_array import SimulatorArray

SHOWER_ROUTE_MODES = ("stream", "disk")
SHOWER_ROUTE_MODULE = "simtools.simtel.shower_route"
_WITHOUT_MULTIPIPE_SEQUENTIAL = "env -u MULTIPIPE_SEQUENTIAL"


def _absolute_path_argument(value):
    """
    Return a path argument as an absolute path; "-" (stdin or stdout) is kept.

    The shower route runs from the working directory of the multipipe script, so relative
    paths are resolved here, against the working directory of the caller.
    """
    return value if value == "-" else str(Path(value).expanduser().resolve())


_SEED_FILE_PREFIX = "file-by-run:"


def _shower_route_seeds(sim_telarray_seed):
    """
    Return the instrument seed and the tracing seed of the shower route.

    The seeds are read from the seed string ("instrument seed,simulation seed"). For a seed
    file ("file-by-run:..."), the instrument seed is taken from the seed object and the
    simulation seed is the tracing seed.

    Parameters
    ----------
    sim_telarray_seed: SimtelSeeds
        Seeds of the sim_telarray simulations.

    Returns
    -------
    tuple of int
        Instrument seed and tracing seed.
    """
    if sim_telarray_seed.seed_string.startswith(_SEED_FILE_PREFIX):
        if sim_telarray_seed.instrument_seed is None:
            raise ValueError("Instrument seed is not set for the seed file of sim_telarray")
        return int(sim_telarray_seed.instrument_seed), int(sim_telarray_seed.simulation_seed)
    instrument, tracing = sim_telarray_seed.seed_string.split(",")
    return int(instrument), int(tracing)


class CorsikaSimtelRunner:
    """
    Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality.

    Uses CorsikaConfig to manage the CORSIKA configuration and SimulatorArray
    for the sim_telarray configuration.

    Parameters
    ----------
    corsika_config : CorsikaConfig or list of CorsikaConfig
        A list of "CorsikaConfig" instances which
        contain the CORSIKA configuration parameters.
    label : str
        Label.
    sequential : bool
        Run multipipe_corsika in sequential mode (disables parallel CORSIKA/sim_telarray execution).
    save_corsika_output : bool
        Save the CORSIKA output stream to a compressed file.
    shower_route_mode : str
        How the photon bunches are routed when the ray-tracing backend traces air-shower
        photons: ``stream`` (piped, no IACT file on disk) or ``disk`` (IACT file kept, route
        and sim_telarray run on files after the multipipe).
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        sequential=False,
        save_corsika_output=False,
        shower_route_mode="stream",
    ):
        self._logger = logging.getLogger(__name__)
        self.corsika_config = gen.ensure_list(corsika_config)
        # the base corsika config is the one used to define the CORSIKA specific parameters.
        # The others are used for the array configurations.
        self.base_corsika_config = self.corsika_config[0]
        self.label = label
        self.sequential = "--sequential" if sequential else ""
        self.save_corsika_output = save_corsika_output
        if shower_route_mode not in SHOWER_ROUTE_MODES:
            raise ValueError(
                f"Invalid shower route mode: {shower_route_mode}; choose from {SHOWER_ROUTE_MODES}"
            )
        self.shower_route_mode = shower_route_mode
        self._post_multipipe_commands = []

        self.runner_service = runner_services.RunnerServices(
            self.base_corsika_config, run_type="multi_pipe", label=label
        )
        self.file_list = None

        self.corsika_runner = corsika_runner.CorsikaRunner(
            corsika_config=self.base_corsika_config,
            label=label,
            use_multipipe=True,
        )
        # The simulator array should be defined for every CORSIKA configuration
        # because it allows to define multiple sim_telarray instances
        self.simulator_array = []
        for _corsika_config in self.corsika_config:
            self.simulator_array.append(
                SimulatorArray(
                    simulation_parameters=_corsika_config.simulation_parameters,
                    array_model=_corsika_config.array_model,
                    label=label,
                )
            )

    def prepare_run(self, run_number=None, sub_script=None, corsika_file=None, extra_commands=None):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        run_number: int
            Run number.
        corsika_file: str or Path
            Path to the CORSIKA input file.
        extra_commands: str
            Additional commands for running simulations.
        """
        self.file_list = self.runner_service.load_files(run_number=run_number)
        self._export_multipipe_script(run_number)
        self.corsika_runner.prepare_run(
            run_number=run_number,
            sub_script=sub_script,
            corsika_file=self.runner_service.get_file_name(
                file_type="multi_pipe_script", run_number=run_number
            )
            if not corsika_file
            else corsika_file,
            extra_commands=extra_commands,
        )
        self.update_file_list_from_runners()

    def _export_multipipe_script(self, run_number):
        """
        Write the multipipe script used in piping CORSIKA to sim_telarray.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        multipipe_file = self.runner_service.get_file_name(
            "multi_pipe_config", run_number=run_number
        )
        route = self._shower_route_active()
        self._post_multipipe_commands = []
        with open(multipipe_file, "w", encoding="utf-8") as file:
            for simulator_array in self.simulator_array:
                if route:
                    self._export_shower_route(simulator_array, run_number, file)
                    continue
                file.write(self._simtel_command(simulator_array, run_number, input_file="-") + "\n")
                file.write("\n")

            if self.save_corsika_output or (route and self.shower_route_mode == "disk"):
                corsika_output = self.corsika_runner.runner_service.get_file_name(
                    file_type="corsika_output", run_number=run_number
                )
                file.write(f"zstd > {corsika_output}\n")

        self._logger.info(f"Multipipe script: {multipipe_file}")
        self._write_multipipe_script(multipipe_file, run_number)

    def _shower_route_active(self):
        """Return True if the ray-tracing backend traces air-shower photons (shower route)."""
        return bool(getattr(get_backend(), "traces_shower_photons", False))

    def _simtel_command(
        self, simulator_array, run_number, input_file, bypass_level=None, group=None
    ):
        """
        Return the accounting command (shell text) running sim_telarray.

        Parameters
        ----------
        simulator_array: SimulatorArray
            Simulator array of one model version.
        run_number: int
            Run number.
        input_file: str
            Input file of sim_telarray ("-" for stdin).
        bypass_level: int or None
            Value of sim_telarray's Bypass_Optics (None: optics not bypassed).
        group: str or None
            Telescope type; if given, the output, histogram, log and resource files are named
            after it (several sim_telarray processes of one model version).

        Returns
        -------
        str
            Shell text of the sim_telarray command.
        """
        run_command = simulator_array.make_run_command(
            run_number=run_number,
            input_file=input_file,  # instruct sim_telarray to take input from stdout
        )
        if bypass_level is not None:
            # make_run_command ends with "-C show=all <input>"; the bypass goes before it.
            run_command[-3:-3] = ["-C", f"Bypass_Optics={bypass_level}"]
        log_file = simulator_array.runner_service.get_file_name(
            file_type="sim_telarray_log", run_number=run_number
        )
        resources_file = simulator_array.runner_service.get_file_name(
            file_type="sim_telarray_resources", run_number=run_number
        )
        if group is not None:
            run_command = [self._group_output_option(arg, group) for arg in run_command]
            log_file = log_file.with_name(f"{group}_{log_file.name}")
            resources_file = resources_file.with_name(f"{group}_{resources_file.name}")
        accounting_command = build_accounting_command(
            run_command,
            resources_file,
            "sim_telarray",
            run_number,
            model_version=simulator_array.array_model.model_version,
            log_file=log_file,
        )
        return f"{simtel_runner.sim_telarray_env_as_string()} " + shlex.join(accounting_command)

    @staticmethod
    def _group_output_option(argument, group):
        """Return a sim_telarray option with the output or histogram file named after a group."""
        for key in ("output_file=", "histogram_file="):
            if argument.startswith(key):
                path = Path(argument[len(key) :])
                return f"{key}{path.with_name(f'{group}_{path.name}')}"
        return argument

    def _export_shower_route(self, simulator_array, run_number, file):
        """
        Write the route and sim_telarray steps of one model version to the multipipe config.

        Stream mode writes one line per telescope type: the route piped into sim_telarray.
        Disk mode writes the route and sim_telarray commands to the multipipe script, to run
        after the multipipe on the IACT file.

        Parameters
        ----------
        simulator_array: SimulatorArray
            Simulator array of one model version.
        run_number: int
            Run number.
        file: file object
            Multipipe configuration.
        """
        map_file, groups = self._write_telescope_route_map(simulator_array, run_number)
        n_groups = len(groups)
        for group, info in groups.items():
            telescope_ids = ",".join(str(telescope_id) for telescope_id in info["ids"])
            if self.shower_route_mode == "stream":
                route = self._route_command(
                    simulator_array, group, map_file, telescope_ids, "-", "-", run_number
                )
                simtel = self._simtel_command(
                    simulator_array,
                    run_number,
                    input_file="-",
                    bypass_level=info["bypass_level"],
                    group=group if n_groups > 1 else None,
                )
                # sim_telarray reads the route's stream, not the CORSIKA stream: it takes no part
                # in the sequential protocol of the multipipe (the route does), so the variable
                # that switches it on is removed for this process.
                log_file = self._route_log_file(simulator_array, group, run_number)
                file.write(
                    f"{self._route_shell_text(route, log_file)} | "
                    f"{_WITHOUT_MULTIPIPE_SEQUENTIAL} {simtel}\n\n"
                )
                continue
            iact_file = self.corsika_runner.runner_service.get_file_name(
                file_type="corsika_output", run_number=run_number
            )
            camera_file = self._camera_file(simulator_array, run_number, group)
            route = self._route_command(
                simulator_array,
                group,
                map_file,
                telescope_ids,
                str(iact_file),
                str(camera_file),
                run_number,
            )
            simtel = self._simtel_command(
                simulator_array,
                run_number,
                input_file=str(camera_file),
                bypass_level=info["bypass_level"],
                group=group if n_groups > 1 else None,
            )
            log_file = self._route_log_file(simulator_array, group, run_number)
            self._post_multipipe_commands.append(self._route_shell_text(route, log_file))
            self._post_multipipe_commands.append(simtel)

    def _write_telescope_route_map(self, simulator_array, run_number):
        """
        Write the telescope map of the shower route (all telescopes of the array).

        The minimum input plane height is taken from the optical model of the first telescope
        of each telescope type (its height includes the margin, see
        ``minimum_input_plane_height_m``).

        Parameters
        ----------
        simulator_array: SimulatorArray
            Simulator array of one model version.
        run_number: int
            Run number.

        Returns
        -------
        Path, dict
            Telescope map file, and the telescope type groups: telescope ids, Bypass_Optics level.
        """
        array_model = simulator_array.array_model
        theta_deg = float(simulator_array.simulation_parameters.zenith_angle)
        phi_deg = float(simulator_array.simulation_parameters.azimuth_angle)
        ids = corsika_telescope_ids(array_model.telescope_models)
        groups = {}
        for name, telescope in array_model.telescope_models.items():
            group = groups.setdefault(telescope.name, {"ids": [], "telescope": telescope})
            group["ids"].append(ids[name])
        entries = {}
        for group_name, group in groups.items():
            telescope = group.pop("telescope")
            dual_mirror = telescope.get_parameter_value("mirror_class") == 2
            group["bypass_level"] = bypass_optics_level(dual_mirror)
            height_m = self._minimum_input_plane_height_m(telescope, theta_deg)
            radius_m = telescope.get_parameter_value_with_unit("telescope_sphere_radius").to_value(
                u.m
            )
            self._logger.info(
                f"Shower route: {group_name} telescope ids {group['ids']}, "
                f"input plane height {height_m:.2f} m, Bypass_Optics={group['bypass_level']}"
            )
            for telescope_id in group["ids"]:
                entries[str(telescope_id)] = {
                    "model": group_name,
                    "model_version": array_model.model_version,
                    "site": array_model.site,
                    "dual_mirror": dual_mirror,
                    "radius_m": float(radius_m),
                    "input_plane_height_m": height_m,
                    "theta_deg": theta_deg,
                    "phi_deg": phi_deg,
                }
        map_file = simulator_array.runner_service.get_file_name(
            "telescope_route_map", run_number=run_number
        )
        map_file.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        return map_file, groups

    @staticmethod
    def _minimum_input_plane_height_m(telescope, theta_deg):
        """Return the minimum safe input plane height [m] of a telescope at a zenith angle."""
        tracer = create_shower_tracer(telescope)
        model = tracer.optical_model(theta_deg, DEFAULT_SEED)
        return minimum_input_plane_height_m(model, theta_deg)

    def _route_command(
        self, simulator_array, group, map_file, telescope_ids, input_file, output_file, run_number
    ):
        """
        Return the accounting command of the shower route (the route process of one telescope type).

        The route is seeded with the sim_telarray instrument and simulation seeds. Its exported
        model files go to a work directory of the telescope type, next to the telescope map.

        Parameters
        ----------
        simulator_array: SimulatorArray
            Simulator array of one model version.
        group: str
            Telescope type of the route.
        map_file: Path
            Telescope map file of the array.
        telescope_ids: str
            Comma-separated telescope ids traced by the route.
        input_file: str
            Input file of the route ("-" for stdin).
        output_file: str
            Output file of the route ("-" for stdout).
        run_number: int
            Run number.

        Returns
        -------
        list of str
            Command of the route wrapped in process accounting.
        """
        array_model = simulator_array.array_model
        instrument_seed, tracing_seed = _shower_route_seeds(array_model.sim_telarray_seed)
        card_file = self.corsika_runner.runner_service.get_file_name(
            "corsika_input", run_number=run_number
        )
        command = [
            sys.executable,
            "-m",
            SHOWER_ROUTE_MODULE,
            "--telescope-map",
            _absolute_path_argument(str(map_file)),
            "--telescope-ids",
            telescope_ids,
            "--input",
            _absolute_path_argument(input_file),
            "--output",
            _absolute_path_argument(output_file),
            "--corsika-card",
            _absolute_path_argument(str(card_file)),
            "--seed",
            str(instrument_seed),
            "--tracing-seed",
            str(tracing_seed),
            "--output-path",
            _absolute_path_argument(str(map_file.parent / f"shower_route_{group}")),
        ]
        models_path = settings.config.args.get("simulation_models_path")
        if models_path:
            command += ["--simulation-models-path", _absolute_path_argument(str(models_path))]
        resources_file = simulator_array.runner_service.get_file_name(
            file_type="shower_route_resources", run_number=run_number
        )
        return build_accounting_command(
            command,
            resources_file.with_name(f"{group}_{resources_file.name}"),
            "shower_route",
            run_number,
            model_version=array_model.model_version,
        )

    def _route_log_file(self, simulator_array, group, run_number):
        """
        Return the stderr log file of the shower route of one telescope type.

        The log is named like the route resources file (``<group>_..._shower_route.log``). The
        route's stdout is the data stream, so only its stderr is kept, by the shell.
        """
        resources_file = simulator_array.runner_service.get_file_name(
            file_type="shower_route_resources", run_number=run_number
        )
        name = f"{group}_{resources_file.name}".replace(".resources.json", ".log")
        return resources_file.with_name(name)

    @staticmethod
    def _route_shell_text(command, log_file):
        """Return the shell text of the route command with its stderr redirected to a log file."""
        return f"{shlex.join(command)} 2> {shlex.quote(_absolute_path_argument(str(log_file)))}"

    def _camera_file(self, simulator_array, run_number, group):
        """Return the camera photon-bunch file of one telescope type of one model version."""
        path = simulator_array.runner_service.get_file_name(
            "camera_bunch_file", run_number=run_number
        )
        return path.with_name(f"{group}_{path.name}")

    def _write_multipipe_script(self, multipipe_file, run_number):
        """
        Write script used to call the multipipe_corsika command.

        Parameters
        ----------
        multipipe_file: str or Path
            The name of the multipipe file which contains all of the multipipe commands.
        run_number: int
            Run number.
        """
        multipipe_script = self.runner_service.get_file_name(
            "multi_pipe_script", run_number=run_number
        )
        with open(multipipe_script, "w", encoding="utf-8") as file:
            multipipe_command = [
                settings.config.sim_telarray_path / "bin/multipipe_corsika",
                "-c",
                multipipe_file,
            ]
            if self.sequential:
                multipipe_command.append(self.sequential)
            accounting_command = build_accounting_command(
                multipipe_command,
                self.runner_service.get_file_name("multi_pipe_resources", run_number=run_number),
                "multipipe",
                run_number,
                model_version=self.base_corsika_config.array_model.model_version,
            )
            file.write(shlex.join(accounting_command))
            # Disk mode: the route and sim_telarray steps run after the multipipe (the script is
            # the pipe target of CORSIKA, run by the shell).
            for command in self._post_multipipe_commands:
                file.write(f"\n{command}")

        multipipe_script.chmod(multipipe_script.stat().st_mode | stat.S_IEXEC)

    def get_resources(self, runtime=None):
        """Return computing resources used."""
        return self.corsika_runner.get_resources(runtime=runtime)

    def update_file_list_from_runners(self):
        """
        Get list of generated files (independent of type).

        Includes file lists from all runners.

        Parameters
        ----------
        file_type : str
            File type to be listed.

        Returns
        -------
        list
            List with the full path of all output files.

        """
        if self.file_list is None:
            self.file_list = self.corsika_runner.file_list
        else:
            self.file_list.update(self.corsika_runner.file_list)

        for simulator_array in self.simulator_array:
            self._merge_runner_file_list(simulator_array.file_list)

        return self.file_list

    def _merge_runner_file_list(self, runner_file_list):
        """Merge one simulator-array file list into the combined file list."""
        for key, data in runner_file_list.items():
            if not isinstance(data, list):
                data = [data]
            if key in self.file_list:
                self._append_runner_files(key, data)
            else:
                self.file_list[key] = data

    def _append_runner_files(self, key, data):
        """Append runner files, normalizing an existing single path to a list."""
        if not isinstance(self.file_list[key], list):
            self.file_list[key] = [self.file_list[key]]
        self.file_list[key].extend(data)
