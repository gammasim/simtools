"""Generate run scripts and directories for CORSIKA simulations."""

import logging
import os
from pathlib import Path

from simtools.io_operations import io_handler
from simtools.runners.runner_services import RunnerServices

__all__ = ["CorsikaRunner", "MissingRequiredEntryInCorsikaConfigError"]


class MissingRequiredEntryInCorsikaConfigError(Exception):
    """Exception for missing required entry in corsika config."""


class CorsikaRunner:
    """
    Generate run scripts and directories for CORSIKA simulations. Run simulations if requested.

    CorsikaRunner is responsible for configuring and running CORSIKA, using corsika_autoinputs
    provided by the sim_telarray package. CorsikaRunner generates shell scripts to be run
    externally or by the simulator module simulator.

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
        """Initialize CorsikaRunner."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")
        self.label = label

        self.corsika_config = corsika_config
        self._keep_seeds = keep_seeds
        self._use_multipipe = use_multipipe

        self._simtel_path = Path(simtel_path)
        self.io_handler = io_handler.IOHandler()

        self.runner_service = RunnerServices(corsika_config, label)
        self._directory = self.runner_service.load_data_directories(
            "corsika_simtel" if use_multipipe else "corsika"
        )

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

        script_file_path = self.runner_service.get_file_name(
            file_type="sub_script",
            **self.runner_service.get_info_for_file_name(self.corsika_config.run_number),
        )
        corsika_input_file = self.corsika_config.generate_corsika_input_file(self._use_multipipe)

        # CORSIKA input file for a specific run, created by the preprocessor pfp
        corsika_input_tmp_name = self.corsika_config.get_corsika_config_file_name(
            file_type="config_tmp", run_number=self.corsika_config.run_number
        )
        corsika_input_tmp_file = self._directory["inputs"].joinpath(corsika_input_tmp_name)

        if use_pfp:
            pfp_command = self._get_pfp_command(corsika_input_tmp_file, corsika_input_file)
        autoinputs_command = self._get_autoinputs_command(
            self.corsika_config.run_number, corsika_input_tmp_file
        )

        extra_commands = kwargs["extra_commands"]
        self._logger.debug(f"Extra commands to be added to the run script {extra_commands}")

        with open(script_file_path, "w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n")
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
            file.write('mkdir -p "$CORSIKA_DATA"\n')
            file.write('cd "$CORSIKA_DATA" || exit 2\n')
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
                file.write(f"cp {corsika_input_file} {corsika_input_tmp_file}")
            file.write("\n# Running corsika_autoinputs\n")
            file.write(autoinputs_command)

            file.write('\necho "RUNTIME: $SECONDS"\n')

        # Changing permissions
        os.system(f"chmod ug+x {script_file_path}")

        return script_file_path

    def get_resources(self, run_number=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(run_number)

    def _get_pfp_command(self, input_tmp_file, corsika_input_file):
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
        cmd = str(cmd) + f" -V -DWITHOUT_MULTIPIPE - < {corsika_input_file}"
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

        log_file = self.runner_service.get_file_name(
            file_type="log", **self.runner_service.get_info_for_file_name(run_number)
        )

        cmd = self._simtel_path.joinpath("sim_telarray/bin/corsika_autoinputs")
        cmd = str(cmd) + f" --run {corsika_bin_path}"
        cmd += f" -R {run_number}"
        cmd += ' -p "$CORSIKA_DATA"'
        if self._keep_seeds:
            cmd += " --keep-seeds"
        cmd += f" {input_tmp_file} | gzip > {log_file} 2>&1"
        cmd += " || exit 1\n"
        return cmd
