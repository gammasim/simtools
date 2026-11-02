"""Prepare shower photon routing commands for sim_telarray stages."""

import json
import logging
import shlex
import sys
from pathlib import Path

import astropy.units as u

from simtools.corsika.corsika_config import corsika_telescope_ids
from simtools.job_execution.process_accounting import build_accounting_command
from simtools.job_execution.runner_services import RunnerServices
from simtools.ray_tracing.backends.interface import create_shower_tracer
from simtools.simtel import simtel_runner
from simtools.simtel.photon_bunch_file import bypass_optics_level
from simtools.simtel.shower_photons import minimum_input_plane_height_m
from simtools.simtel.shower_route import DEFAULT_SEED

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


class ShowerRouting:
    """Build the photon conversion and sim_telarray commands of one array."""

    def __init__(self, corsika_config, context, label=None):
        self.context = context
        self._logger = logging.getLogger(__name__)
        self.corsika_service = RunnerServices(
            corsika_config, run_type="corsika", label=label, context=context
        )

    def commands(self, simulator_array, run_number, input_file):
        """Build one stream or file conversion and sim_telarray command per telescope type."""
        map_file, groups = self._write_telescope_route_map(simulator_array, run_number)
        lines = []
        grouped_files = {}
        for group, info in groups.items():
            telescope_ids = ",".join(str(telescope_id) for telescope_id in info["ids"])
            streamed = input_file == "-"
            camera_file = (
                "-" if streamed else str(self._camera_file(simulator_array, run_number, group))
            )
            simtel = self._simtel_command(
                simulator_array,
                run_number,
                input_file=camera_file,
                bypass_level=info["bypass_level"],
                group=group if len(groups) > 1 else None,
            )
            if len(groups) > 1:
                self._record_group_files(grouped_files, simulator_array.file_list, group)
            route = self._route_command(
                simulator_array,
                group,
                map_file,
                telescope_ids,
                str(input_file),
                camera_file,
                run_number,
            )
            log_file = self._route_log_file(simulator_array, group, run_number)
            route_text = self._route_shell_text(route, log_file)
            if streamed:
                lines.append(f"{route_text} | {_WITHOUT_MULTIPIPE_SEQUENTIAL} {simtel}")
            else:
                lines.extend((route_text, simtel))
        simulator_array.file_list.update(grouped_files)
        return lines

    @staticmethod
    def _record_group_files(grouped_files, file_list, group):
        """Collect the files written by one telescope group."""
        for file_type, path in file_list.items():
            if file_type.startswith("sim_telarray_"):
                grouped_files.setdefault(file_type, []).append(
                    path.with_name(f"{group}_{path.name}")
                )

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

    def _minimum_input_plane_height_m(self, telescope, theta_deg):
        """Return the minimum safe input plane height [m] of a telescope at a zenith angle."""
        tracer = create_shower_tracer(telescope, name=self.context.software.ray_tracing_backend)
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
        card_file = self.corsika_service.get_file_name("corsika_input", run_number=run_number)
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
            "--ray-tracing-backend",
            self.context.software.ray_tracing_backend,
            "--seed",
            str(instrument_seed),
            "--tracing-seed",
            str(tracing_seed),
            "--output-path",
            _absolute_path_argument(str(map_file.parent / f"shower_route_{group}")),
        ]
        source = getattr(self.context.model_reader, "source_config", None) or {}
        models_path = self.context.args.get("simulation_models_path")
        if source.get("type") == "filesystem":
            models_path = source["path"]
        if source.get("type") == "git":
            command = [
                "env",
                "-u",
                "SIMTOOLS_SIMULATION_MODELS_PATH",
                f"SIMTOOLS_SIMULATION_MODELS_GIT_PATH={source['repository']}",
                f"SIMTOOLS_SIMULATION_MODELS_GIT_REVISION={source['commit']}",
                *command,
            ]
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
