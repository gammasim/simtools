"""Simulation runner for array simulations."""

import logging

from simtools import settings
from simtools.io import io_handler
from simtools.runners.simtel_runner import InvalidOutputFileError, SimtelRunner
from simtools.utils.general import clear_default_sim_telarray_cfg_directories


class SimulatorArray(SimtelRunner):
    """
    SimulatorArray is the interface with sim_telarray to perform array simulations.

    Parameters
    ----------
    corsika_config_data: CorsikaConfig
        CORSIKA configuration.
    label: str
        Instance label.
    use_multipipe: bool
        Use multipipe to run CORSIKA and sim_telarray.
    sim_telarray_seeds: dict
        Dictionary with configuration for sim_telarray random instrument setup.
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        use_multipipe=False,
        sim_telarray_seeds=None,
        calibration_config=None,
    ):
        """Initialize SimulatorArray."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimulatorArray")
        super().__init__(
            label=label,
            corsika_config=corsika_config,
            use_multipipe=use_multipipe,
            calibration_run_mode=calibration_config.get("run_mode") if calibration_config else None,
        )

        self.sim_telarray_seeds = sim_telarray_seeds
        self.corsika_config = corsika_config
        self.calibration_config = calibration_config
        self.io_handler = io_handler.IOHandler()
        self._log_file = None

    def prepare_run_script(self, test=False, input_file=None, run_number=None, extra_commands=None):
        """
        Build and return the full path of the bash run script containing the sim_telarray command.

        Parameters
        ----------
        test: bool
            Test flag for faster execution.
        input_file: str or Path
            Full path of the input CORSIKA file.
        run_number: int
            Run number.
        extra_commands: list[str]
            Additional commands for running simulations given in config.yml.

        Returns
        -------
        Path
            Full path of the run script.
        """
        script_file_path = self.get_file_name(file_type="sub_script", run_number=run_number)
        self._logger.debug(f"Run bash script - {script_file_path}")
        self._logger.debug(f"Extra commands to be added to the run script {extra_commands}")

        command = self.make_run_command(run_number=run_number, input_file=input_file)
        with script_file_path.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")
            file.write("set -e\n")
            file.write("set -o pipefail\n")
            file.write("\nSECONDS=0\n")

            if extra_commands:
                file.write("# Writing extras\n")
                for line in extra_commands:
                    file.write(f"{line}\n")
                file.write("# End of extras\n\n")

            n = 1 if test else self.runs_per_set
            for _ in range(n):
                file.write(f"{command}\n\n")

            file.write('\necho "RUNTIME: $SECONDS"\n')

        return script_file_path

    def make_run_command(self, run_number=None, input_file=None, weak_pointing=None):
        """
        Build and return the command to run sim_telarray.

        Parameters
        ----------
        input_file: str
            Full path of the input CORSIKA file
        run_number: int (optional)
            run number
        weak_pointing: bool (optional)
            Specify weak pointing option for sim_telarray.

        Returns
        -------
        str
            Command to run sim_telarray.
        """
        command = self._common_run_command(run_number, weak_pointing)

        if self.calibration_config:
            command += self._make_run_command_for_calibration_simulations()
        else:
            command += self._make_run_command_for_shower_simulations()

        # "-C show=all" should be the last option
        command += super().get_config_option("show", "all")
        command += f" {input_file} | gzip > {self._log_file} 2>&1 || exit"

        return clear_default_sim_telarray_cfg_directories(command)

    def _make_run_command_for_shower_simulations(self):
        """
        Build and return the command to run sim_telarray shower simulations.

        Returns
        -------
        str
            Command to run sim_telarray.
        """
        return super().get_config_option(
            "power_law",
            SimulatorArray.get_power_law_for_sim_telarray_histograms(
                self.corsika_config.primary_particle
            ),
        )

    def _make_run_command_for_calibration_simulations(self):
        """Build sim_telarray command for calibration simulations."""
        cfg = self.calibration_config
        altitude = self.corsika_config.array_model.site_model.get_parameter_value_with_unit(
            "reference_point_altitude"
        ).to_value("m")

        command = super().get_config_option("Altitude", altitude)

        for key in ("nsb_scaling_factor", "stars"):
            if cfg.get(key):
                command += super().get_config_option(key, cfg[key])

        run_mode = cfg.get("run_mode")
        if run_mode in ("pedestals", "pedestals_nsb_only"):
            n_events = cfg.get("number_of_pedestal_events", cfg["number_of_events"])
            command += super().get_config_option("pedestal_events", n_events)
        if run_mode == "pedestals_nsb_only":
            command += self._pedestals_nsb_only_command()
        if run_mode == "pedestals_dark":
            n_events = cfg.get("number_of_dark_events", cfg["number_of_events"])
            command += super().get_config_option("dark_events", n_events)
        if run_mode == "direct_injection":
            n_events = cfg.get("number_of_flasher_events", cfg["number_of_events"])
            command += super().get_config_option("laser_events", n_events)

        return command

    def _common_run_command(self, run_number, weak_pointing=None):
        """Build generic run command for sim_telarray."""
        config_dir = self.corsika_config.array_model.get_config_directory()
        self._log_file = self.get_file_name(file_type="log", run_number=run_number)
        histogram_file = self.get_file_name(file_type="histogram", run_number=run_number)
        output_file = self.get_file_name(file_type="simtel_output", run_number=run_number)
        self.corsika_config.array_model.export_all_simtel_config_files()

        command = str(settings.config.sim_telarray_exe)
        command += f" -c {self.corsika_config.array_model.config_file_path}"
        command += f" -I{config_dir}"
        command += super().get_config_option(
            "telescope_theta", self.corsika_config.zenith_angle, weak_pointing
        )
        command += super().get_config_option(
            "telescope_phi", self.corsika_config.azimuth_angle, weak_pointing
        )
        command += super().get_config_option("histogram_file", histogram_file)
        command += super().get_config_option("random_state", "none")
        if self.sim_telarray_seeds and self.sim_telarray_seeds.get("random_instrument_instances"):
            command += super().get_config_option(
                "random_seed",
                f"file-by-run:{config_dir}/{self.sim_telarray_seeds['seed_file_name']},auto",
            )
        elif self.sim_telarray_seeds and self.sim_telarray_seeds.get("seed"):
            command += super().get_config_option("random_seed", self.sim_telarray_seeds["seed"])
        command += super().get_config_option("output_file", output_file)

        return command

    def _pedestals_nsb_only_command(self):
        """
        Generate the command to run sim_telarray for nsb-only pedestal simulations.

        Returns
        -------
        str
            Command to run sim_telarray.
        """
        null_values = [
            "fadc_noise",
            "fadc_lg_noise",
            "qe_variation",
            "gain_variation",
            "fadc_var_pedestal",
            "fadc_err_pedestal",
            "fadc_sysvar_pedestal",
            "fadc_dev_pedestal",
        ]
        null_command_parts = [super().get_config_option(param, 0.0) for param in null_values]
        command = " ".join(null_command_parts)

        one_values = [
            "fadc_lg_var_pedestal",
            "fadc_lg_err_pedestal",
            "fadc_lg_dev_pedestal",
            "fadc_lg_sysvar_pedestal",
        ]
        one_command_parts = [super().get_config_option(param, -1.0) for param in one_values]
        command += " " + " ".join(one_command_parts)
        return command

    def _check_run_result(self, run_number=None):
        """
        Check if sim_telarray output file exists.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        bool
            True if sim_telarray output file exists.

        Raises
        ------
        InvalidOutputFileError
            If sim_telarray output file does not exist.
        """
        output_file = self.get_file_name(file_type="simtel_output", run_number=run_number)
        if not output_file.exists():
            raise InvalidOutputFileError(f"sim_telarray output file {output_file} does not exist.")
        self._logger.debug(f"sim_telarray output file {output_file} exists.")
        return True

    @staticmethod
    def get_power_law_for_sim_telarray_histograms(primary):
        """
        Get the power law index for sim_telarray.

        Events will be filled in histograms in sim_telarray with a weight according to
        the difference between this exponent and the one used for the shower simulations.

        Parameters
        ----------
        primary: str
            Primary particle.

        Returns
        -------
        float
            Power law index.
        """
        power_laws = {
            "gamma": 2.5,
            "electron": 3.3,
        }
        if primary.name in power_laws:
            return power_laws[primary.name]

        return 2.68
