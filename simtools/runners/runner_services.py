"""Base service methods for simulation runners."""

import logging
from pathlib import Path

from simtools.io_operations import io_handler

_logger = logging.getLogger(__name__)


class RunnerServices:
    """
    Base services for simulation runners.

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
        self._logger.debug("Init RunnerServices")
        self.label = label
        self.corsika_config = corsika_config
        self.directory = {}

    def get_info_for_file_name(self, run_number):
        """
        Return dictionary for building the file names for simulation output files.

        Parameters
        ----------
        run_number : int
            Run number.

        Returns
        -------
        dict
            Dictionary with the keys or building the file names for simulation output files.
        """
        return {
            "run": self.corsika_config.validate_run_number(run_number),
            "primary": self.corsika_config.primary,
            "array_name": self.corsika_config.array_model.layout_name,
            "site": self.corsika_config.array_model.site,
            "label": self.label,
            "zenith": self.corsika_config.zenith_angle,
            "azimuth": self.corsika_config.azimuth_angle,
        }

    @staticmethod
    def _get_simulation_software_list(simulation_software):
        """
        Return a list of simulation software based on the input string.

        Args:
            simulation_software: String representing the desired software.

        Returns
        -------
            List of simulation software names.
        """
        software_map = {
            "corsika": ["corsika"],
            "simtel": ["simtel"],
            "corsika_simtel": ["corsika", "simtel"],
        }
        return software_map.get(simulation_software, [])

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
        self.directory["output"] = io_handler.IOHandler().get_output_directory(self.label)
        _logger.debug(f"Creating output dir {self.directory['output']}")
        for dir_name in ["sub_scripts", "sub_logs"]:
            self.directory[dir_name] = self.directory["output"].joinpath(dir_name)
            self.directory[dir_name].mkdir(parents=True, exist_ok=True)
        for _simulation_software in self._get_simulation_software_list(simulation_software):
            for dir_name in ["data", "inputs", "logs"]:
                self.directory[dir_name] = self.directory["output"].joinpath(
                    _simulation_software, dir_name
                )
                self.directory[dir_name].mkdir(parents=True, exist_ok=True)
        self._logger.debug(f"Data directories for {simulation_software}: {self.directory}")
        return self.directory

    def has_file(self, file_type, run_number=None, mode="out"):
        """
        Check that the file of file_type for the specified run number exists.

        Parameters
        ----------
        file_type: str
            File type to check.
        run_number: int
            Run number.

        """
        info_for_file_name = self.get_info_for_file_name(run_number)
        run_sub_file = self.get_file_name(file_type, **info_for_file_name, mode=mode)
        _logger.debug(f"Checking if {run_sub_file} exists")
        return Path(run_sub_file).is_file()

    def get_file_name(self, file_type, **kwargs):
        """
        Get a CORSIKA/sim_telarray style file name for various log and data file types.

        Parameters
        ----------
        file_type: str
            The type of file (determines the file suffix).
        kwargs: dict
            The dictionary must include the following parameters (unless listed as optional):
                run: int
                    Run number.
                primary: str
                    Primary particle (e.g gamma, proton etc).
                site: str
                    Site name (usually North/South or Paranal/LaPalma).
                array_name: str
                    Array name.
                label: str
                    Instance label (optional).
                mode: str
                    out or err (optional, relevant only for sub_log).

        Returns
        -------
        str
            File name with full path.

        Raises
        ------
        ValueError
            If file_type is unknown.
        """
        file_label = f"_{kwargs.get('label', '')}" if kwargs.get("label") is not None else ""
        zenith = self.corsika_config.get_config_parameter("THETAP")[0]
        azimuth = self.corsika_config.azimuth_angle
        run_dir = self._get_run_number_string(kwargs["run"])
        file_name = (
            f"{run_dir}_{kwargs['primary']}_"
            f"za{round(zenith):02}deg_azm{azimuth:03}deg_"
            f"{kwargs['site']}_{kwargs['array_name']}{file_label}"
        )

        log_suffixes = {
            "log": ".log.gz",
            "histogram": ".hdata.zst",
        }
        if file_type in log_suffixes:
            return self.directory["logs"].joinpath(f"{file_name}{log_suffixes[file_type]}")

        data_suffixes = {
            "output": ".zst",
            "corsika_output": ".zst",
            "corsika_log": ".log",
            "simtel_output": ".simtel.zst",
        }
        if file_type in data_suffixes:
            data_run_dir = self.directory["data"].joinpath(run_dir)
            data_run_dir.mkdir(parents=True, exist_ok=True)
            return data_run_dir.joinpath(f"{file_name}{data_suffixes[file_type]}")

        if file_type in ("sub_log", "sub_script"):
            suffix = ".log" if file_type == "sub_log" else ".sh"
            if kwargs and kwargs.get("mode") != "":
                suffix = f".{kwargs['mode']}"
            sub_log_file_dir = self.directory["output"].joinpath(f"{file_type}s")
            sub_log_file_dir.mkdir(parents=True, exist_ok=True)
            return sub_log_file_dir.joinpath(f"sub_{file_name}{suffix}")

        raise ValueError(f"The requested file type ({file_type}) is unknown")

    @staticmethod
    def _get_run_number_string(run_number):
        """
        Get run number string as used for the simulation file names(ex. run000014).

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        str
            Run number string.
        """
        nn = str(run_number)
        if len(nn) > 6:
            raise ValueError("Run number cannot have more than 6 digits")
        return "run" + nn.zfill(6)

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
        sub_log_file = self.get_file_name(
            file_type="sub_log", **self.get_info_for_file_name(run_number), mode="out"
        )

        _logger.debug(f"Reading resources from {sub_log_file}")

        _resources = {}
        _resources["runtime"] = None
        with open(sub_log_file, encoding="utf-8") as file:
            for line in reversed(list(file)):
                if "RUNTIME" in line:
                    _resources["runtime"] = int(line.split()[1])
                    break

        if _resources["runtime"] is None:
            _logger.debug("RUNTIME was not found in run log file")
        _resources["n_events"] = int(self.corsika_config.get_config_parameter("NSHOW"))
        return _resources
