"""Base service methods for simulation runners."""

import logging

from simtools.io import io_handler

_logger = logging.getLogger(__name__)


FILES_AND_PATHS = {
    # CORSIKA
    "corsika_input": {
        "suffix": ".input",
        "sub_dir_type": "run_number",
    },
    "corsika_output": {
        "suffix": ".corsika.zst",
        "sub_dir_type": "run_number",
    },
    "corsika_log": {
        "suffix": ".corsika.log.gz",
        "sub_dir_type": "run_number",
    },
    # sim_telarray
    "sim_telarray_output": {
        "suffix": ".simtel.zst",
        "sub_dir_type": "run_number",
    },
    "sim_telarray_histogram": {
        "suffix": ".hdata.zst",
        "sub_dir_type": "run_number",
    },
    "sim_telarray_log": {
        "suffix": ".simtel.log.gz",
        "sub_dir_type": "run_number",
    },
    "sim_telarray_event_data": {
        "suffix": ".reduced_event_data.hdf5",
        "sub_dir_type": "run_number",
    },
    # multipipe
    "multi_pipe_config": {
        "suffix": ".multi_pipe.cfg",
        "sub_dir_type": "run_number",
    },
    "multi_pipe_script": {
        "suffix": ".multi_pipe.sh",
        "sub_dir_type": "run_number",
    },
    # job submission
    "sub_out": {
        "suffix": ".out",
        "sub_dir_type": "sub",
    },
    "sub_log": {
        "suffix": ".log",
        "sub_dir_type": "sub",
    },
    "sub_script": {
        "suffix": ".sh",
        "sub_dir_type": "sub",
    },
}


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
    corsika_config : CorsikaConfig, list of CorsikaConfig
        Configuration parameters for CORSIKA.
    run_type : str
        Type of simulation runner.
    label : str
        Label.
    """

    def __init__(self, corsika_config, run_type, label=None):
        """Initialize RunnerServices."""
        self._logger = logging.getLogger(__name__)
        self.label = label
        self.corsika_config = corsika_config
        self.run_type = run_type
        self.directory = self.load_data_directory()

    def load_data_directory(self):
        """
        Create and return directory for the given run type.

        Returns
        -------
        Path
            Path to the created directory.
        """
        ioh = io_handler.IOHandler()
        directory = ioh.get_output_directory(self.run_type)
        self._logger.debug(f"Data directories for {self.run_type}: {directory}")
        return directory

    def load_files(self, run_number=None):
        """
        Load files required for the simulation run.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        dict
            Dictionary containing paths to files required for the simulation run.
        """
        run_files = {}
        for key in FILES_AND_PATHS:
            if key.startswith(self.run_type.lower()):
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
        vc_high = self.corsika_config.get_config_parameter("VIEWCONE")[1]
        zenith = self.corsika_config.get_config_parameter("THETAP")[0]

        if calibration_run_mode is not None and calibration_run_mode != "":
            primary_name = calibration_run_mode
        else:
            primary_name = self.corsika_config.primary_particle.name
            if primary_name == "gamma" and vc_high > 0:
                primary_name = "gamma_diffuse"

        file_label = f"_{self.label}" if self.label else ""
        run_number_string = self._get_run_number_string(run_number)

        prefix = f"{primary_name}_{run_number_string}_" if primary_name else f"{run_number_string}_"
        return (
            prefix
            + f"za{round(zenith):02}deg_"
            + f"azm{self.corsika_config.azimuth_angle:03}deg_"
            + f"{self.corsika_config.array_model.site}_"
            + f"{self.corsika_config.array_model.layout_name}_"
            + (self.corsika_config.array_model.model_version if not is_multi_pipe else "")
            + file_label
        )

    def _get_sub_directory(self, run_number, dir_path):
        """
        Return sub directory with / without run number.

        Parameters
        ----------
        run_number: int
            Run number.
        dir_path: Path
            Parent directory path.

        Returns
        -------
        Path
            Child directory path.
        """
        sub_dir = dir_path / self._get_run_number_string(run_number)
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
            desc = FILES_AND_PATHS[file_type]
        except KeyError as exc:
            raise ValueError(f"Unknown file type: {file_type}") from exc

        if desc["sub_dir_type"] == "run_number":
            dir_path = self._get_sub_directory(run_number, self.directory)
        else:
            dir_path = self.directory
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
        run_number = validate_corsika_run_number(run_number)
        if run_number >= 10**6:
            raise ValueError("Run number cannot have more than 6 digits")
        return f"run{run_number:06d}"

    def get_resources(self, sub_out_file):
        """
        Read run time of job from last line of submission log file.

        Parameters
        ----------
        sub_out_file: str or Path
            Path to the submission output file.

        Returns
        -------
        dict
            run time and number of simulated events
        """
        _logger.debug(f"Reading resources from {sub_out_file}")

        runtime = None
        with open(sub_out_file, encoding="utf-8") as f:
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
