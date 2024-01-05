import logging
import os
from copy import copy
from pathlib import Path

from simtools.corsika.corsika_config import (
    CorsikaConfig,
    MissingRequiredInputInCorsikaConfigData,
)
from simtools.io_operations import io_handler
from simtools.utils import names
from simtools.utils.general import collect_data_from_file_or_dict

__all__ = ["CorsikaRunner", "MissingRequiredEntryInCorsikaConfig"]


class MissingRequiredEntryInCorsikaConfig(Exception):
    """Exception for missing required entry in corsika config."""


class CorsikaRunner:
    """
    CorsikaRunner is responsible for running CORSIKA, through the corsika_autoinputs program \
    provided by the sim_telarray package. It provides shell scripts to be run externally or by \
    the module simulator. Same instance can be used to generate scripts for any given run number.

    It uses CorsikaConfig to manage the CORSIKA configuration. User parameters must be given by the\
    corsika_config_data or corsika_config_file arguments. An example of corsika_config_data follows\
    below.

    .. code-block:: python

        corsika_config_data = {
            'data_directory': .
            'primary': 'proton',
            'nshow': 10000,
            'nrun': 1,
            'zenith': 20 * u.deg,
            'viewcone': 5 * u.deg,
            'erange': [10 * u.GeV, 100 * u.TeV],
            'eslope': -2,
            'phi': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }

    The remaining CORSIKA parameters can be set as a yaml file, using the argument \
    corsika_parameters_file. When not given, corsika_parameters will be loaded from \
    data/parameters/corsika_parameters.yml.

    The CORSIKA output directory must be set by the data_directory entry. The following directories\
    will be created to store the logs and input file:
    {data_directory}/corsika/$site/$primary/logs
    {data_directory}/corsika/$site/$primary/scripts

    Parameters
    ----------
    mongo_db_config: dict
        MongoDB configuration.
    site: str
        South or North.
    layout_name: str
        Name of the layout.
    label: str
        Instance label.
    keep_seeds: bool
        Use seeds based on run number and primary particle.  If False, use sim_telarray seeds.
    simtel_source_path: str or Path
        Location of source of the sim_telarray/CORSIKA package.
    corsika_config_data: dict
        Dict with CORSIKA config data.
    corsika_config_file: str or Path
        Path to yaml file containing CORSIKA config data.
    corsika_parameters_file: str or Path
        Path to yaml file containing CORSIKA parameters.
    """

    def __init__(
        self,
        mongo_db_config,
        site,
        layout_name,
        simtel_source_path,
        label=None,
        keep_seeds=False,
        corsika_parameters_file=None,
        corsika_config_data=None,
        corsika_config_file=None,
        use_multipipe=False,
    ):
        """
        CorsikaRunner init.
        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")

        self.label = label
        self.site = names.validate_site_name(site)
        self.layout_name = names.validate_array_layout_name(layout_name)

        self._keep_seeds = keep_seeds

        self._simtel_source_path = Path(simtel_source_path)
        self.io_handler = io_handler.IOHandler()
        _runner_directory = "corsika_simtel" if use_multipipe else "corsika"
        self._output_directory = self.io_handler.get_output_directory(self.label, _runner_directory)
        self._logger.debug(f"Creating output dir {self._output_directory}, if needed,")

        self._corsika_parameters_file = corsika_parameters_file
        corsika_config_data = collect_data_from_file_or_dict(
            corsika_config_file, corsika_config_data
        )
        self._load_corsika_config_data(corsika_config_data)
        self._define_corsika_config(mongo_db_config, use_multipipe)

        self._load_corsika_data_directories()

    def _load_corsika_config_data(self, corsika_config_data):
        """Reads corsika_config_data, creates corsika_config and corsika_input_file."""

        corsika_data_directory_from_config = corsika_config_data.get("data_directory", None)
        if corsika_data_directory_from_config is None:
            # corsika_data_directory not given (or None).
            msg = (
                "data_directory not given in corsika_config "
                "- default output directory will be set."
            )
            self._logger.warning(msg)
            self._corsika_data_directory = self._output_directory
        else:
            # corsika_data_directory given and not None.
            self._corsika_data_directory = Path(corsika_data_directory_from_config)

        self._corsika_data_directory = self._corsika_data_directory.joinpath("corsika-data")

        # Copying corsika_config_data and removing corsika_data_directory
        # (it does not go to CorsikaConfig)
        self._corsika_config_data = copy(corsika_config_data)
        self._corsika_config_data.pop("data_directory", None)

    def _define_corsika_config(self, mongo_db_config, use_multipipe=False):
        """
        Create the CORSIKA config instance.
        This validates the input given in corsika_config_data as well.
        """

        try:
            self.corsika_config = CorsikaConfig(
                mongo_db_config=mongo_db_config,
                site=self.site,
                label=self.label,
                layout_name=self.layout_name,
                corsika_config_data=self._corsika_config_data,
                simtel_source_path=self._simtel_source_path,
                corsika_parameters_file=self._corsika_parameters_file,
            )
            # CORSIKA input file used as template for all runs
            self._corsika_input_file = self.corsika_config.get_input_file(use_multipipe)
        except MissingRequiredInputInCorsikaConfigData:
            msg = "corsika_config_data is missing required entries."
            self._logger.error(msg)
            raise

    def _load_corsika_data_directories(self):
        """Create CORSIKA directories for data, log and input."""
        corsika_base_dir = self._corsika_data_directory.joinpath(self.site)
        corsika_base_dir = corsika_base_dir.joinpath(self.corsika_config.primary)
        corsika_base_dir = corsika_base_dir.absolute()

        self._corsika_data_dir = corsika_base_dir.joinpath("data")
        self._corsika_data_dir.mkdir(parents=True, exist_ok=True)
        self._corsika_input_dir = corsika_base_dir.joinpath("input")
        self._corsika_input_dir.mkdir(parents=True, exist_ok=True)
        self._corsika_log_dir = corsika_base_dir.joinpath("log")
        self._corsika_log_dir.mkdir(parents=True, exist_ok=True)

    def prepare_run_script(self, use_pfp=True, **kwargs):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        use_pfp: bool
            Whether to use the preprocessor in preparing the CORSIKA input file
        kwargs: dict
            The following optional parameters can be provided:
                run_number: int
                    Run number.
                extra_commands: str
                    Additional commands for running simulations.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        kwargs = {
            "run_number": None,
            "extra_commands": None,
            **kwargs,
        }
        run_number = self._validate_run_number(kwargs["run_number"])

        script_file_path = self.get_file_name(
            file_type="script", **self.get_info_for_file_name(run_number)
        )

        # CORSIKA input file for a specific run, created by the preprocessor pfp
        corsika_input_tmp_name = self.corsika_config.get_file_name(
            file_type="config_tmp", run_number=run_number
        )
        corsika_input_tmp_file = self._corsika_input_dir.joinpath(corsika_input_tmp_name)

        if use_pfp:
            pfp_command = self._get_pfp_command(corsika_input_tmp_file)
        autoinputs_command = self._get_autoinputs_command(run_number, corsika_input_tmp_file)

        extra_commands = kwargs["extra_commands"]
        self._logger.debug(f"Extra commands to be added to the run script {extra_commands}")

        with open(script_file_path, "w", encoding="utf-8") as file:
            # shebang
            file.write("#!/usr/bin/env bash\n")

            # Make sure to exit on failed commands and report their error code
            file.write("set -e\n")
            file.write("set -o pipefail\n")

            # Setting SECONDS variable to measure runtime
            file.write("\nSECONDS=0\n")

            if extra_commands is not None:
                file.write("\n# Writing extras\n")
                file.write(f"{extra_commands}\n")
                file.write("# End of extras\n\n")

            file.write(f"export CORSIKA_DATA={self._corsika_data_dir}\n")
            file.write("\n# Creating CORSIKA_DATA\n")
            file.write(f"mkdir -p {self._corsika_data_dir}\n")
            file.write(f"cd {self._corsika_data_dir} || exit 2\n")
            if use_pfp:
                file.write("\n# Running pfp\n")
                file.write(pfp_command)
                file.write("\n# Replacing the XXXXXX placeholder with the run number\n")
                file.write(f"sed -i 's/XXXXXX/{run_number:06}/g' {corsika_input_tmp_file}\n")
            else:
                file.write("\n# Copying CORSIKA input file to run location\n")
                file.write(f"cp {self._corsika_input_file} {corsika_input_tmp_file}")
            file.write("\n# Running corsika_autoinputs\n")
            file.write(autoinputs_command)

            # Printing out runtime
            file.write('\necho "RUNTIME: $SECONDS"\n')

        # Changing permissions
        os.system(f"chmod ug+x {script_file_path}")

        return script_file_path

    def _get_pfp_command(self, input_tmp_file):
        """Get pfp pre-processor command."""
        cmd = self._simtel_source_path.joinpath("sim_telarray/bin/pfp")
        cmd = str(cmd) + f" -V -DWITHOUT_MULTIPIPE - < {self._corsika_input_file}"
        cmd += f" > {input_tmp_file} || exit\n"
        return cmd

    def _get_autoinputs_command(self, run_number, input_tmp_file):
        """Get autoinputs command."""
        corsika_bin_path = self._simtel_source_path.joinpath("corsika-run/corsika")

        log_file = self.get_file_name(
            file_type="corsika_autoinputs_log", **self.get_info_for_file_name(run_number)
        )

        cmd = self._simtel_source_path.joinpath("sim_telarray/bin/corsika_autoinputs")
        cmd = str(cmd) + f" --run {corsika_bin_path}"
        cmd += f" -R {run_number}"
        cmd += f" -p {self._corsika_data_dir}"
        if self._keep_seeds:
            cmd += " --keep-seeds"
        cmd += f" {input_tmp_file} | gzip > {log_file} 2>&1"
        cmd += " || exit 1\n"
        return cmd

    def get_info_for_file_name(self, run_number):
        """
        Get a dictionary with the info necessary for building the CORSIKA runner file names.

        Returns
        -------
        dict
            Dictionary with the keys necessary for building the CORSIKA runner file names.
        """
        run_number = self._validate_run_number(run_number)
        return {
            "run": run_number,
            "primary": self.corsika_config.primary,
            "array_name": self.layout_name,
            "site": self.site,
            "label": self.label,
        }

    def get_file_name(self, file_type, **kwargs):
        """
        Get a CORSIKA style file name for various file types.

        Parameters
        ----------
        file_type: str
            The type of file (determines the file suffix).
            Choices are corsika_autoinputs_log, corsika_log, script, output or sub_log.
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

        file_label = (
            f"_{kwargs['label']}" if "label" in kwargs and kwargs["label"] is not None else ""
        )
        file_name = (
            f"corsika_run{kwargs['run']:06}_{kwargs['primary']}_"
            f"{kwargs['site']}_{kwargs['array_name']}{file_label}"
        )

        if file_type == "corsika_autoinputs_log":
            return self._corsika_log_dir.joinpath(f"log_{file_name}.log.gz")
        if file_type == "corsika_log":
            run_dir = self._get_run_directory(kwargs["run"])
            return self._corsika_data_dir.joinpath(run_dir).joinpath(f"run{kwargs['run']}.log")
        if file_type == "script":
            script_file_dir = self._output_directory.joinpath("scripts")
            script_file_dir.mkdir(parents=True, exist_ok=True)
            return script_file_dir.joinpath(f"{file_name}.sh")
        if file_type == "output":
            zenith = self.corsika_config.get_user_parameter("THETAP")[0]
            azimuth = self.corsika_config.get_user_parameter("AZM")[0]
            file_name = (
                f"corsika_run{kwargs['run']:06}_{kwargs['primary']}_"
                f"za{round(zenith):03}deg_azm{round(azimuth):03}deg_"
                f"{kwargs['site']}_{kwargs['array_name']}{file_label}"
            )
            run_dir = self._get_run_directory(kwargs["run"])
            return self._corsika_data_dir.joinpath(run_dir).joinpath(f"{file_name}.zst")
        if file_type == "sub_log":
            suffix = ".log"
            if "mode" in kwargs and kwargs["mode"] != "":
                suffix = f".{kwargs['mode']}"
            sub_log_file_dir = self._output_directory.joinpath("logs")
            sub_log_file_dir.mkdir(parents=True, exist_ok=True)
            return sub_log_file_dir.joinpath(f"log_sub_{file_name}{suffix}")

        raise ValueError(f"The requested file type ({file_type}) is unknown")

    def has_file(self, file_type, run_number=None, mode="out"):
        """
        Checks that the file of file_type for the specified run number exists.

        Parameters
        ----------
        file_type: str
            File type to check.
            Choices are corsika_autoinputs_log, corsika_log, script, output or sub_log.
        run_number: int
            Run number.

        """

        info_for_file_name = self.get_info_for_file_name(run_number)
        run_sub_file = self.get_file_name(file_type, **info_for_file_name, mode=mode)
        return Path(run_sub_file).is_file()

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

        self._logger.debug(f"Reading resources from {sub_log_file}")

        _resources = {}

        _resources["runtime"] = None
        with open(sub_log_file, "r", encoding="utf-8") as file:
            for line in reversed(list(file)):
                if "RUNTIME" in line:
                    _resources["runtime"] = int(line.split()[1])
                    break

        if _resources["runtime"] is None:
            self._logger.debug("RUNTIME was not found in run log file")

        # Calculating number of events
        _resources["n_events"] = int(self.corsika_config.get_user_parameter("NSHOW"))

        return _resources

    @staticmethod
    def _get_run_directory(run_number):
        """Get run directory created by sim_telarray (ex. run000014)."""
        nn = str(run_number)
        return "run" + nn.zfill(6)

    def _validate_run_number(self, run_number):
        """
        Returns the run number from corsika_config in case run_number is None, Raise ValueError if\
        run_number is not valid (< 1) or returns run_number if it is a valid value.
        """
        if run_number is None:
            return self.corsika_config.get_user_parameter("RUNNR")
        if not float(run_number).is_integer() or run_number < 1:
            msg = f"Invalid type of run number ({run_number}) - it must be an uint."
            self._logger.error(msg)
            raise ValueError(msg)

        return run_number
