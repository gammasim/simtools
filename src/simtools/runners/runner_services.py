"""Base service methods for simulation runners."""

import logging
from pathlib import Path

from simtools.io import io_handler

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
            primary_name = self.corsika_config.primary
            if primary_name == "gamma" and _vc_high > 0:
                primary_name = "gamma_diffuse"
        return {
            "run_number": self.corsika_config.validate_run_number(run_number),
            "primary": primary_name,
            "array_name": self.corsika_config.array_model.layout_name,
            "site": self.corsika_config.array_model.site,
            "label": self.label,
            "model_version": self.corsika_config.array_model.model_version,
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
            "sim_telarray": ["sim_telarray"],
            "corsika_sim_telarray": ["corsika", "sim_telarray"],
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
        _logger.debug(f"Creating output dir {self.directory['output']}")
        for dir_name in ["sub_scripts", "sub_logs"]:
            self.directory[dir_name] = ioh.get_output_directory(dir_name)
        for _simulation_software in self._get_simulation_software_list(simulation_software):
            for dir_name in ["data", "inputs", "logs"]:
                self.directory[dir_name] = ioh.get_output_directory(
                    [_simulation_software, dir_name]
                )
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
        run_sub_file = self.get_file_name(file_type, run_number=run_number, mode=mode)
        _logger.debug(f"Checking if {run_sub_file} exists")
        return Path(run_sub_file).is_file()

    def _get_file_basename(self, run_number, calibration_run_mode):
        """
        Get the base name for the simulation files.

        Parameters
        ----------
        run_number: int
            Run number.
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
            + f"{info_for_file_name['model_version']}{file_label}"
        )

    def _get_log_file_path(self, file_type, file_name):
        """
        Return path for log files.

        Parameters
        ----------
        file_type : str
            File type.
        file_name : str
            File name.

        Returns
        -------
        Path
            Path for log files.
        """
        log_suffixes = {
            "log": ".log.gz",
            "histogram": ".hdata.zst",
            "corsika_log": ".corsika.log.gz",
        }
        return self.directory["logs"].joinpath(f"{file_name}{log_suffixes[file_type]}")

    def _get_data_file_path(self, file_type, file_name, run_number):
        """
        Return path for data files.

        Parameters
        ----------
        file_type : str
            File type.
        file_name : str
            File name.
        run_number : int
            Run number.

        Returns
        -------
        Path
            Path for data files.
        """
        data_suffixes = {
            "output": ".zst",
            "corsika_output": ".corsika.zst",
            "simtel_output": ".simtel.zst",
            "event_data": ".reduced_event_data.hdf5",
        }
        run_dir = self._get_run_number_string(run_number)
        data_run_dir = self.directory["data"].joinpath(run_dir)
        data_run_dir.mkdir(parents=True, exist_ok=True)
        return data_run_dir.joinpath(f"{file_name}{data_suffixes[file_type]}")

    def _get_sub_file_path(self, file_type, file_name, mode):
        """
        Return path for submission files.

        Parameters
        ----------
        file_type : str
            File type.
        file_name : str
            File name.
        mode : str
            Mode (out or err).

        Returns
        -------
        Path
            Path for submission files.
        """
        suffix = ".log" if file_type == "sub_log" else ".sh"
        if mode and mode != "":
            suffix = f".{mode}"
        sub_log_file_dir = self.directory["output"].joinpath(f"{file_type}s")
        sub_log_file_dir.mkdir(parents=True, exist_ok=True)
        return sub_log_file_dir.joinpath(f"sub_{file_name}{suffix}")

    def get_file_name(
        self,
        file_type,
        run_number=None,
        mode=None,
        calibration_run_mode=None,
        _model_version_index=0,
    ):  # pylint: disable=unused-argument
        """
        Get a CORSIKA/sim_telarray style file name for various log and data file types.

        Parameters
        ----------
        file_type : str
            The type of file (determines the file suffix).
        run_number : int
            Run number.
        mode: str
            out or err (optional, relevant only for sub_log).
        calibration_run_mode: str
            Calibration run mode.
        model_version_index: int
            Index of the model version.
            This is not used here, but in other implementations of this function is
            used to select the correct simulator_array instance in case
            multiple array models are simulated.

        Returns
        -------
        str
            File name with full path.

        Raises
        ------
        ValueError
            If file_type is unknown.
        """
        file_name = self._get_file_basename(run_number, calibration_run_mode)

        if file_type in ["log", "histogram", "corsika_log"]:
            return self._get_log_file_path(file_type, file_name)

        if file_type in ["output", "corsika_output", "simtel_output", "event_data"]:
            return self._get_data_file_path(file_type, file_name, run_number)

        if file_type in ("sub_log", "sub_script"):
            return self._get_sub_file_path(file_type, file_name, mode)

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
        sub_log_file = self.get_file_name(file_type="sub_log", run_number=run_number, mode="out")
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
