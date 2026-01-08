"""Generate run scripts and directories for CORSIKA simulations."""

import logging
import stat
from pathlib import Path

from simtools import settings
from simtools.runners.runner_services import RunnerServices


class CorsikaRunner:
    """
    Prepare and run CORSIKA simulations.

    Generate run scripts and directories for CORSIKA simulations. Run simulations if requested.

    Parameters
    ----------
    corsika_config_data: CorsikaConfig
        CORSIKA configuration.
    label: str
        Instance label.
    corsika_seeds: list
        List of fixed seeds used for CORSIKA random number generators.
    use_multipipe: bool
        Use multipipe to run CORSIKA and sim_telarray.
    curved_atmosphere_min_zenith_angle: Quantity
        Minimum zenith angle for which to use the curved-atmosphere CORSIKA binary.
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        corsika_seeds=None,
        use_multipipe=False,
        curved_atmosphere_min_zenith_angle=None,
    ):
        """Initialize CorsikaRunner."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")
        self.label = label

        self.corsika_config = corsika_config
        self._corsika_seeds = corsika_seeds
        self._use_multipipe = use_multipipe
        self.curved_atmosphere_min_zenith_angle = curved_atmosphere_min_zenith_angle

        self.runner_service = RunnerServices(corsika_config, "corsika", label)
        self.file_list = None

    def prepare_run(self, run_number, sub_script, extra_commands=None, corsika_file=None):
        """
        Prepare CORSIKA run script and run directory.

        The CORSIKA run directory includes all input files needed for the simulation.

        Parameters
        ----------
        run_number: int
            Run number.
        sub_script: str or Path
            Path to the CORSIKA run script to be created.
        corsika_file: str or Path
            Path to the multipipe script (used only if use_multipipe is True).
        extra_commands: str
            Additional commands for running simulations.
        """
        self.file_list = self.runner_service.load_files(run_number=run_number)

        self.corsika_config.generate_corsika_input_file(
            self._use_multipipe,
            self._corsika_seeds,
            self.file_list["corsika_input"],
            self.file_list["corsika_output"] if not self._use_multipipe else corsika_file,
            corsika_path=self._corsika_executable().parent.resolve(),
        )

        self._logger.debug(f"Extra commands to be added to the run script: {extra_commands}")

        corsika_run_dir = self.file_list["corsika_output"].parent

        self._export_run_script(sub_script, corsika_run_dir, extra_commands)

    def _corsika_executable(self):
        """Get the CORSIKA executable path."""
        if self.corsika_config.use_curved_atmosphere:
            self._logger.debug("Using curved-atmosphere CORSIKA binary.")
            return Path(settings.config.corsika_exe_curved)
        self._logger.debug("Using flat-atmosphere CORSIKA binary.")
        return Path(settings.config.corsika_exe)

    def _export_run_script(self, sub_script, corsika_run_dir, extra_commands):
        """Export CORSIKA run script."""
        corsika_log_file = self.file_list["corsika_log"].with_suffix("")
        sub_script = Path(sub_script)
        with open(sub_script, "w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n")
            file.write("set -e\n")
            file.write("set -o pipefail\n")

            # Setting SECONDS variable to measure runtime
            file.write("\nSECONDS=0\n")

            if extra_commands is not None:
                file.write("\n# Writing extras\n")
                file.write(f"{extra_commands}\n")
                file.write("# End of extras\n\n")

            file.write(f"export CORSIKA_DATA={corsika_run_dir}\n")
            file.write('mkdir -p "$CORSIKA_DATA"\n')
            file.write('cd "$CORSIKA_DATA" || exit 2\n')

            file.write("\n# Running corsika\n")
            file.write(
                f"{self._corsika_executable()} < {self.file_list['corsika_input']} "
                f"> {corsika_log_file} 2>&1\n"
            )
            file.write("\n# Cleanup\n")
            file.write(f"gzip {corsika_log_file}\n")

            file.write('\necho "RUNTIME: $SECONDS"\n')

        sub_script.chmod(sub_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    def get_resources(self, sub_out_file):
        """Return computing resources used."""
        return self.runner_service.get_resources(sub_out_file)
