"""Generate run scripts and directories for CORSIKA simulations."""

import logging
import stat

from simtools import settings
from simtools.io import io_handler
from simtools.runners.runner_services import RunnerServices


class MissingRequiredEntryInCorsikaConfigError(Exception):
    """Exception for missing required entry in corsika config."""


class CorsikaRunner:
    """
    Generate run scripts and directories for CORSIKA simulations. Run simulations if requested.

    CorsikaRunner is responsible for configuring and running CORSIKA, using corsika_autoinputs
    provided by the sim_telarray package. CorsikaRunner generates shell scripts to be run
    externally or by the simulator module simulator.

    CorsikaRunner is configured through a CorsikaConfig instance.

    Parameters
    ----------
    corsika_config_data: CorsikaConfig
        CORSIKA configuration.
    label: str
        Instance label.
    keep_seeds: bool
        Use seeds based on run number and primary particle. If False, use sim_telarray seeds.
    use_multipipe: bool
        Use multipipe to run CORSIKA and sim_telarray.
    curved_atmosphere_min_zenith_angle: Quantity
        Minimum zenith angle for which to use the curved-atmosphere CORSIKA binary.
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        keep_seeds=False,
        use_multipipe=False,
        curved_atmosphere_min_zenith_angle=None,
    ):
        """Initialize CorsikaRunner."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")
        self.label = label

        self.corsika_config = corsika_config
        self._keep_seeds = keep_seeds
        self._use_multipipe = use_multipipe
        self.curved_atmosphere_min_zenith_angle = curved_atmosphere_min_zenith_angle

        self.io_handler = io_handler.IOHandler()

        self.runner_service = RunnerServices(corsika_config, label)
        self._directory = self.runner_service.load_data_directories("corsika")

    def prepare_run_script(
        self, run_number=None, extra_commands=None, input_file=None, use_pfp=True
    ):
        """
        Prepare and write CORSIKA run script.

        Parameters
        ----------
        use_pfp: bool
            Whether to use the preprocessor in preparing the CORSIKA input file
        run_number: int
            Run number.
        extra_commands: str
            Additional commands for running simulations.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        if input_file is not None:
            self._logger.warning(
                "input_file parameter is not used in CorsikaRunner.prepare_run_script"
            )
        self.corsika_config.run_number = run_number

        script_file_path = self.get_file_name(
            file_type="sub_script", run_number=self.corsika_config.run_number
        )
        corsika_input_file = self.corsika_config.generate_corsika_input_file(
            use_multipipe=self._use_multipipe, use_test_seeds=self._keep_seeds
        )

        # CORSIKA input file for a specific run, created by the preprocessor pfp
        corsika_input_tmp_name = self.corsika_config.get_corsika_config_file_name(
            file_type="config_tmp", run_number=self.corsika_config.run_number
        )
        corsika_input_tmp_file = self._directory["inputs"].joinpath(corsika_input_tmp_name)
        # CORSIKA log file naming (temporary and final)
        corsika_log_tmp_file = (
            self._directory["data"]
            .joinpath(f"run{self.corsika_config.run_number:06}")
            .joinpath(f"run{self.corsika_config.run_number}.log")
        )
        corsika_log_file = self.get_file_name(
            file_type="corsika_log", run_number=self.corsika_config.run_number
        )

        if use_pfp:
            pfp_command = self._get_pfp_command(corsika_input_tmp_file, corsika_input_file)
        autoinputs_command = self._get_autoinputs_command(
            self.corsika_config.run_number, corsika_input_tmp_file
        )

        self._logger.debug(f"Extra commands to be added to the run script: {extra_commands}")
        self._logger.debug(f"CORSIKA data will be set to {self._directory['data']}")

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
            file.write("\n# Moving log files to the corsika log directory\n")
            file.write(f"gzip {corsika_log_tmp_file}\n")
            file.write(f"mv -v {corsika_log_tmp_file}.gz {corsika_log_file}\n")

            file.write('\necho "RUNTIME: $SECONDS"\n')

        script_file_path.chmod(script_file_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
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
        cmd = settings.config.sim_telarray_path / "bin/pfp"
        cmd = str(cmd) + f" -V -DWITHOUT_MULTIPIPE - < {corsika_input_file}"
        cmd += f" > {input_tmp_file} || exit\n"
        return cmd

    def _get_autoinputs_command(self, run_number, input_tmp_file):
        """
        Get autoinputs command.

        corsika_autoinputs is a tool to generate random seeds and user/host dependent
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
        if self.corsika_config.use_curved_atmosphere:
            corsika_bin_path = settings.config.corsika_exe_curved
            self._logger.debug("Using curved-atmosphere CORSIKA binary.")
        else:
            corsika_bin_path = settings.config.corsika_exe
            self._logger.debug("Using flat-atmosphere CORSIKA binary.")

        log_file = self.get_file_name(file_type="log", run_number=run_number)
        if self._use_multipipe:
            log_file = log_file.with_name(f"multipipe_{log_file.name}")

        cmd = settings.config.sim_telarray_path.joinpath("bin/corsika_autoinputs")
        cmd = str(cmd) + f" --run {corsika_bin_path}"
        cmd += f" -R {run_number}"
        cmd += ' -p "$CORSIKA_DATA"'
        if self._keep_seeds:
            logging.warning(
                "Using --keep-seeds option in corsika_autoinputs is not recommended. "
                "It should only be used for testing purposes."
            )
            cmd += " --keep-seeds"
        cmd += f" {input_tmp_file} | gzip > {log_file} 2>&1"
        cmd += " || exit 1\n"
        return cmd

    def get_file_name(
        self,
        simulation_software="corsika",
        file_type=None,
        run_number=None,
        mode="",
        model_version_index=0,
    ):
        """
        Get the full path of a file for a given run number.

        Parameters
        ----------
        simulation_software: str
            Simulation software.
        file_type: str
            File type.
        run_number: int
            Run number.
        model_version_index: int
            Index of the model version.
            This is used to select the correct simulator_array instance in case
            multiple array models are simulated.

        Returns
        -------
        str
            File name with full path.
        """
        if simulation_software.lower() != "corsika":
            raise ValueError(
                f"simulation_software ({simulation_software}) is not supported in CorsikaRunner"
            )
        return self.runner_service.get_file_name(
            file_type=file_type,
            run_number=run_number,
            mode=mode,
            _model_version_index=model_version_index,
        )
