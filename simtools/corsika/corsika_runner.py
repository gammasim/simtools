"""Generate run scripts and directories for CORSIKA simulations."""

import logging
import os
from pathlib import Path

from simtools.io_operations import io_handler

__all__ = ["CorsikaRunner", "MissingRequiredEntryInCorsikaConfigError"]


class MissingRequiredEntryInCorsikaConfigError(Exception):
    """Exception for missing required entry in corsika config."""


class CorsikaRunner:
    """
    Generate run scripts and directories for CORSIKA simulations. Run simulations if requested.

    CorsikaRunner is responsible for configuring and running CORSIKA, using corsika_autoinputs
    provided by the sim_telarray package. It provides shell scripts to be run externally or by
    the module simulator. Same instance can be used to generate scripts for any given run number.

    CorsikaRunner is configured through a CorsikaConfig instance.

    The CORSIKA output directory must be set by the data_directory entry. The following directories\
    will be created to store the logs and input file:
    {data_directory}/corsika/$site/$primary/logs
    {data_directory}/corsika/$site/$primary/scripts

    Parameters
    ----------
    corsika_config_data: CorsikaConfig
        CORSIKA configuration.
    simtel_path: str or Path
        Location of source of the sim_telarray/CORSIKA package.
    label: str
        Instance label.
    keep_seeds: bool
        Use seeds based on run number and primary particle. If False, use sim_telarray seeds.
    use_multipipe: bool
        Use multipipe to run CORSIKA and sim_telarray.
    """

    def __init__(
        self,
        corsika_config,
        simtel_path,
        label=None,
        keep_seeds=False,
        use_multipipe=False,
    ):
        """CorsikaRunner init."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")

        self.label = label

        self.corsika_config = corsika_config
        self._keep_seeds = keep_seeds

        self._simtel_path = Path(simtel_path)
        self.io_handler = io_handler.IOHandler()

        self.corsika_config = corsika_config
        self._directory = self._load_corsika_data_directories()
        self._corsika_input_file = self.corsika_config.get_corsika_input_file(use_multipipe)

    def _load_corsika_data_directories(self, use_multipipe=False):
        """
        Create and return CORSIKA directories for output, data, log and input.

        Parameters
        ----------
        use_multipipe: bool
            Use multipipe to run CORSIKA and simtelarray.

        Returns
        -------
        dict
            Dictionary containing paths to output, data, input, and log directories.
        """

        directory = {}
        directory["output"] = self.io_handler.get_output_directory(
            self.label, "corsika_simtel" if use_multipipe else "corsika"
        )
        self._logger.debug(f"Creating output dir {directory['output']}.")
        corsika_base_dir = (
            directory["output"]
            .joinpath(
                "corsika-data", self.corsika_config.array_model.site, self.corsika_config.primary
            )
            .absolute()
        )
        for dir_name in ["data", "input", "log"]:
            directory[dir_name] = corsika_base_dir.joinpath(dir_name)
            directory[dir_name].mkdir(parents=True, exist_ok=True)

        return directory

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
        self.corsika_config.run_number = kwargs["run_number"]

        script_file_path = self.get_file_name(
            file_type="script", **self.get_info_for_file_name(self.corsika_config.run_number)
        )

        # CORSIKA input file for a specific run, created by the preprocessor pfp
        corsika_input_tmp_name = self.corsika_config.get_file_name(
            file_type="config_tmp", run_number=self.corsika_config.run_number
        )
        corsika_input_tmp_file = self._directory["input"].joinpath(corsika_input_tmp_name)

        if use_pfp:
            pfp_command = self._get_pfp_command(corsika_input_tmp_file)
        autoinputs_command = self._get_autoinputs_command(
            self.corsika_config.run_number, corsika_input_tmp_file
        )

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

            file.write(f"export CORSIKA_DATA={self._directory['data']}\n")
            file.write("\n# Creating CORSIKA_DATA\n")
            file.write(f"mkdir -p {self._directory['data']}\n")
            file.write(f"cd {self._directory['data']} || exit 2\n")
            if use_pfp:
                file.write("\n# Running pfp\n")
                file.write(pfp_command)
                file.write("\n# Replacing the XXXXXX placeholder with the run number\n")
                file.write(
                    f"sed -i 's/XXXXXX/{self.corsika_config.run_number:06}/g' "
                    f"{corsika_input_tmp_file}\n"
                )
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
        """
        Get pfp pre-processor command.

        pfp is a pre-processor tool and part of sim_telarray.

        Parameters
        ----------
        input_tmp_file: Path
            Temporary input file.

        Returns
        -------
        str
            pfp command.
        """
        cmd = self._simtel_path.joinpath("sim_telarray/bin/pfp")
        cmd = str(cmd) + f" -V -DWITHOUT_MULTIPIPE - < {self._corsika_input_file}"
        cmd += f" > {input_tmp_file} || exit\n"
        return cmd

    def _get_autoinputs_command(self, run_number, input_tmp_file):
        """
        Get autoinputs command.

        corsika_autoinputs is a tool to generate random and user/host dependent
        parameters for CORSIKA configuration.

        Parameters
        ----------
        run_number: int
            Run number.
        input_tmp_file: Path
            Temporary input file.

        Returns
        -------
        str
            autoinputs command.
        """
        corsika_bin_path = self._simtel_path.joinpath("corsika-run/corsika")

        log_file = self.get_file_name(
            file_type="corsika_autoinputs_log", **self.get_info_for_file_name(run_number)
        )

        cmd = self._simtel_path.joinpath("sim_telarray/bin/corsika_autoinputs")
        cmd = str(cmd) + f" --run {corsika_bin_path}"
        cmd += f" -R {run_number}"
        cmd += f" -p {self._directory['data']}"
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
        return {
            "run": self.corsika_config.validate_run_number(run_number),
            "primary": self.corsika_config.primary,
            "array_name": self.corsika_config.array_model.layout_name,
            "site": self.corsika_config.array_model.site,
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
            return self._directory["log"].joinpath(f"log_{file_name}.log.gz")
        if file_type == "corsika_log":
            run_dir = self._get_run_directory(kwargs["run"])
            return self._directory["data"].joinpath(run_dir).joinpath(f"run{kwargs['run']}.log")
        if file_type == "script":
            script_file_dir = self._directory["output"].joinpath("scripts")
            script_file_dir.mkdir(parents=True, exist_ok=True)
            return script_file_dir.joinpath(f"{file_name}.sh")
        if file_type == "output":
            zenith = self.corsika_config.get_config_parameter("THETAP")[0]
            azimuth = self.corsika_config.get_config_parameter("azimuth_angle")
            file_name = (
                f"corsika_run{kwargs['run']:06}_{kwargs['primary']}_"
                f"za{round(zenith):03}deg_azm{azimuth:03}deg_"
                f"{kwargs['site']}_{kwargs['array_name']}{file_label}"
            )
            run_dir = self._get_run_directory(kwargs["run"])
            return self._directory["data"].joinpath(run_dir).joinpath(f"{file_name}.zst")
        if file_type == "sub_log":
            suffix = ".log"
            if "mode" in kwargs and kwargs["mode"] != "":
                suffix = f".{kwargs['mode']}"
            sub_log_file_dir = self._directory["output"].joinpath("logs")
            sub_log_file_dir.mkdir(parents=True, exist_ok=True)
            return sub_log_file_dir.joinpath(f"log_sub_{file_name}{suffix}")

        raise ValueError(f"The requested file type ({file_type}) is unknown")

    def has_file(self, file_type, run_number=None, mode="out"):
        """
        Check that the file of file_type for the specified run number exists.

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
        self._logger.debug(f"Checking if {run_sub_file} exists")
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
        with open(sub_log_file, encoding="utf-8") as file:
            for line in reversed(list(file)):
                if "RUNTIME" in line:
                    _resources["runtime"] = int(line.split()[1])
                    break

        if _resources["runtime"] is None:
            self._logger.debug("RUNTIME was not found in run log file")

        # Calculating number of events
        _resources["n_events"] = int(self.corsika_config.get_config_parameter("NSHOW"))

        return _resources

    @staticmethod
    def _get_run_directory(run_number):
        """Get run directory created by sim_telarray (ex. run000014)."""
        nn = str(run_number)
        if len(nn) > 6:
            raise ValueError("Run number cannot have more than 6 digits")
        return "run" + nn.zfill(6)
