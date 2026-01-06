"""Base service methods for simulation runners."""

import logging

from simtools.io import io_handler

_logger = logging.getLogger(__name__)


def validate_corsika_run_number(run_number):
    """
    Validate run number and return it.

    Parameters
    ----------
    run_number: int
        Run number.

    Returns
    -------
    int
        Run number.

    Raises
    ------
    ValueError
        If run_number is not a valid value (< 1 or > 999999).
    """
    if not float(run_number).is_integer() or run_number < 1 or run_number > 999999:
        raise ValueError(
            f"Invalid type of run number ({run_number}) - it must be an uint < 1000000."
        )
    return run_number


class RunnerServices:
    """
    Base service methods for simulation runners.

    Includes file naming and directory management.

    Parameters
    ----------
    corsika_config : CorsikaConfig
        Configuration parameters for CORSIKA.
    label : str
        Label.
    """

    def __init__(self, corsika_config, label=None):
        """Initialize RunnerServices."""
        self._logger = logging.getLogger(__name__)
        self.label = label
        self.corsika_config = corsika_config
        self.directory = {}
        self.file_description = self._define_files_and_paths()

    def _define_files_and_paths(self):
        """
        Define files and paths used by the runners.

        Directories must be consistent with those generated with load_data_directories.
        """
        return {
            # CORSIKA
            "corsika_input": {
                "suffix": ".input",
                "directory": "corsika",
                "sub_dir_type": "run_number",
            },
            "corsika_output": {
                "suffix": ".corsika.zst",
                "directory": "corsika",
                "sub_dir_type": "run_number",
            },
            "corsika_log": {
                "suffix": ".corsika.log.gz",
                "directory": "corsika",
                "sub_dir_type": "run_number",
            },
            # sim_telarray
            "sim_telarray_output": {
                "suffix": ".simtel.zst",
                "directory": "sim_telarray",
                "sub_dir_type": "run_number",
            },
            "sim_telarray_histogram": {
                "suffix": ".hdata.zst",
                "directory": "sim_telarray",
                "sub_dir_type": "run_number",
            },
            "sim_telarray_log": {
                "suffix": ".simtel.log.gz",
                "directory": "sim_telarray",
                "sub_dir_type": "run_number",
            },
            "multi_pipe_config": {
                "suffix": ".multi_pipe.cfg",
                "directory": "multi_pipe",
                "sub_dir_type": "run_number",
            },
            "multi_pipe_script": {
                "suffix": ".multi_pipe.sh",
                "directory": "multi_pipe",
                "sub_dir_type": "run_number",
            },
            # simtools
            "event_data": {
                "suffix": ".reduced_event_data.hdf5",
                "directory": "data",
                "sub_dir_type": "run_number",
            },
            # job submission
            "sub_out": {
                "suffix": ".out",
                "directory": "output",
                "sub_dir_type": "sub",
            },
            "sub_log": {
                "suffix": ".log",
                "directory": "output",
                "sub_dir_type": "sub",
            },
            "sub_script": {
                "suffix": ".sh",
                "directory": "output",
                "sub_dir_type": "sub",
            },
        }

    def _get_info_for_file_name(self, run_number, calibration_run_mode=None):
        """
        Return dictionary for building names for simulation output files.

        Parameters
        ----------
        run_number : int
            Run number.
        calibration_run_mode: str
            Calibration run mode.

        Returns
        -------
        dict
            Dictionary with the keys or building the file names for simulation output files.
        """
        _vc_high = self.corsika_config.get_config_parameter("VIEWCONE")[1]
        if calibration_run_mode is not None and calibration_run_mode != "":
            primary_name = calibration_run_mode
        else:
            primary_name = self.corsika_config.primary_particle.name
            if primary_name == "gamma" and _vc_high > 0:
                primary_name = "gamma_diffuse"
        return {
            "run_number": validate_corsika_run_number(run_number),
            "primary": primary_name,
            "array_name": self.corsika_config.array_model.layout_name,
            "site": self.corsika_config.array_model.site,
            "label": self.label,
            "model_version": self.corsika_config.array_model.model_version,
            "zenith": self.corsika_config.zenith_angle,
            "azimuth": self.corsika_config.azimuth_angle,
            "vc_low": self.corsika_config.get_config_parameter("VIEWCONE")[0],
            "vc_high": _vc_high,
        }

    @staticmethod
    def _get_simulation_software_list(simulation_software):
        """
        Return a list of simulation software based on the input string.

        Parameters
        ----------
        simulation_software: str
            String representing the desired software.

        Returns
        -------
            List of simulation software names.
        """
        software_map = {
            "corsika": ["corsika"],
            "sim_telarray": ["sim_telarray"],
            "corsika_sim_telarray": ["corsika", "sim_telarray", "multi_pipe"],
        }
        return software_map.get(simulation_software.lower(), [])

    def load_data_directories(self, simulation_software):
        """
        Create and return directories for output, data, log and input.

        Parameters
        ----------
        simulation_software : str
            Simulation software to be used.

        Returns
        -------
        dict
            Dictionary containing paths requires for simulation configuration.
        """
        ioh = io_handler.IOHandler()
        self.directory["output"] = ioh.get_output_directory()

        for dir_name in ["sub", *self._get_simulation_software_list(simulation_software)]:
            self.directory[dir_name] = ioh.get_output_directory(dir_name)

        self._logger.debug(f"Data directories for {simulation_software}: {self.directory}")
        return self.directory

    def load_files(self, simulation_software, run_number=None):
        """
        Load files required for the simulation run.

        Parameters
        ----------
        simulation_software : str
            Simulation software to be used.
        run_number: int
            Run number.

        Returns
        -------
        dict
            Dictionary containing paths to files required for the simulation run.
        """
        run_files = {}
        for key in self.file_description:
            # sub files are always included
            if key.startswith("sub_"):
                run_files[key] = self.get_file_name(file_type=key, run_number=run_number)
            # simulation software dependent files
            for name in self._get_simulation_software_list(simulation_software):
                if key.startswith(name.lower()):
                    run_files[key] = self.get_file_name(file_type=key, run_number=run_number)

        for key, file_path in run_files.items():
            self._logger.debug(f"{key}: {file_path}")

        return run_files

    def _get_file_basename(self, run_number, calibration_run_mode, is_multi_pipe=False):
        """
        Get the base name for the simulation files.

        Parameters
        ----------
        run_number: int
            Run number.
        file_type: str
            File type.
        calibration_run_mode: str
            Calibration run mode.

        Returns
        -------
        str
            Base name for the simulation files.
        """
        info_for_file_name = self._get_info_for_file_name(run_number, calibration_run_mode)

        file_label = f"_{info_for_file_name['label']}" if info_for_file_name.get("label") else ""
        zenith = self.corsika_config.get_config_parameter("THETAP")[0]
        azimuth = self.corsika_config.azimuth_angle
        run_number_string = self._get_run_number_string(info_for_file_name["run_number"])
        prefix = (
            f"{info_for_file_name['primary']}_{run_number_string}_"
            if info_for_file_name["primary"]
            else f"{run_number_string}_"
        )
        return (
            prefix
            + f"za{round(zenith):02}deg_azm{azimuth:03}deg_"
            + f"{info_for_file_name['site']}_{info_for_file_name['array_name']}_"
            + (info_for_file_name["model_version"] if not is_multi_pipe else "")
            + file_label
        )

    def _get_sub_directory(self, sub_dir_type, run_number, dir_path):
        """
        Return sub directory depending on data type.

        Parameters
        ----------
        sub_dir_type: str
            Sub directory type.
        run_number: int
            Run number.
        dir_path: Path
            Parent directory path.

        Returns
        -------
        Path
            Child directory path.
        """
        if sub_dir_type is None:
            return dir_path

        name = (
            self._get_run_number_string(run_number)
            if sub_dir_type == "run_number"
            else sub_dir_type
        )

        sub_dir = dir_path / name
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir

    def get_file_name(self, file_type, run_number=None, calibration_run_mode=None):
        """
        Get a file name depending on file type and run number.

        Parameters
        ----------
        file_type : str
            The type of file (determines the file suffix).
        run_number : int
            Run number.
        calibration_run_mode: str
            Calibration run mode.

        Returns
        -------
        str
            File name with full path.

        Raises
        ------
        ValueError
            If file_type is unknown.
        """
        file_name = self._get_file_basename(
            run_number, calibration_run_mode, file_type.startswith("multi_pipe")
        )

        try:
            desc = self.file_description[file_type]
        except KeyError as exc:
            raise ValueError(f"Unknown file type: {file_type}") from exc

        try:
            dir_path = self._get_sub_directory(
                desc["sub_dir_type"],
                run_number,
                self.directory[desc["directory"]],
            )
        except KeyError as exc:
            raise ValueError(
                f"For file type {file_type}, unknown directory type: {desc['directory']}"
                f" or unknown sub directory type: {desc['sub_dir_type']}"
            ) from exc

        return dir_path / f"{file_name}{desc['suffix']}"

    @staticmethod
    def _get_run_number_string(run_number):
        """
        Get run number string as used for the simulation file names (ex. run000014).

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        str
            Run number string.
        """
        if run_number >= 10**6:
            raise ValueError("Run number cannot have more than 6 digits")
        return f"run{run_number:06d}"

    def get_resources(self, run_number=None):
        """
        Read run time of job from last line of submission log file.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        dict
            run time and number of simulated events
        """
        sub_out = self.get_file_name(file_type="sub_out", run_number=run_number)
        _logger.debug(f"Reading resources from {sub_out}")

        runtime = None
        with open(sub_out, encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                if "RUNTIME" in line:
                    runtime = int(line.split()[1])
                    break

        if runtime is None:
            _logger.debug("RUNTIME was not found in run log file")

        return {
            "runtime": runtime,
            "n_events": int(self.corsika_config.get_config_parameter("NSHOW")),
        }
