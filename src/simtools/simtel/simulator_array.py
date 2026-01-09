"""Simulation runner for array simulations."""

from pathlib import Path

from simtools import settings
from simtools.io import io_handler
from simtools.runners.simtel_runner import InvalidOutputFileError, SimtelRunner


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
    is_calibration_run: bool
        Flag to indicate if this is a calibration run.
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        sim_telarray_seeds=None,
        is_calibration_run=False,
    ):
        """Initialize SimulatorArray."""
        super().__init__(
            label=label,
            corsika_config=corsika_config,
            is_calibration_run=is_calibration_run,
        )

        self.sim_telarray_seeds = sim_telarray_seeds
        self.corsika_config = corsika_config
        self.is_calibration_run = is_calibration_run
        self.io_handler = io_handler.IOHandler()
        self._log_file = None

    def prepare_run(self, run_number=None, sub_script=None, corsika_file=None, extra_commands=None):
        """
        Build and return the full path of the bash run script containing the sim_telarray command.

        Parameters
        ----------
        run_number: int
            Run number.
        corsika_file: str or Path
            Full path of the input CORSIKA file.
        extra_commands: list[str]
            Additional commands for running simulations given in config.yml.
        """
        command = self.make_run_command(run_number=run_number, corsika_input_file=corsika_file)
        sub_script = Path(sub_script)
        self._logger.debug(f"Run bash script - {sub_script}")
        self._logger.debug(f"Extra commands to be added to the run script {extra_commands}")
        with sub_script.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")
            file.write("set -e\n")
            file.write("set -o pipefail\n")
            file.write("\nSECONDS=0\n")

            if extra_commands:
                file.write("# Writing extras\n")
                for line in extra_commands:
                    file.write(f"{line}\n")
                file.write("# End of extras\n\n")

            for _ in range(self.runs_per_set):
                file.write("SIM_TELARRAY_CONFIG_PATH='' " + " ".join(command) + "\n")

            file.write('\necho "RUNTIME: $SECONDS"\n')

    def make_run_command(self, run_number=None, corsika_input_file=None, weak_pointing=None):
        """
        Build and return the command to run sim_telarray.

        Parameters
        ----------
        corsika_input_file: str
            Full path of the input CORSIKA file
        run_number: int (optional)
            run number

        Returns
        -------
        str
            Command to run sim_telarray.
        """
        self.file_list = self.runner_service.load_files(run_number=run_number)
        command = self._common_run_command(run_number, weak_pointing)

        if self.is_calibration_run:
            command += self._make_run_command_for_calibration_simulations()
        else:
            command += self._make_run_command_for_shower_simulations()

        # "-C show=all" should be the last option
        return [*command, "-C", "show=all", corsika_input_file]

    def _make_run_command_for_shower_simulations(self):
        """
        Build and return the command to run sim_telarray shower simulations.

        Returns
        -------
        str
            Command to run sim_telarray.
        """
        return [
            "-C",
            "power_law="
            f"{
                SimulatorArray.get_power_law_for_sim_telarray_histograms(
                    self.corsika_config.primary_particle
                )
            }",
        ]

    def _make_run_command_for_calibration_simulations(self):
        """Build sim_telarray command for calibration simulations."""
        cfg = settings.config.args
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
        self._log_file = self.runner_service.get_file_name(
            file_type="sim_telarray_log", run_number=run_number
        )
        histogram_file = self.runner_service.get_file_name(
            file_type="sim_telarray_histogram", run_number=run_number
        )
        output_file = self.runner_service.get_file_name(
            file_type="sim_telarray_output", run_number=run_number
        )
        self.corsika_config.array_model.export_all_simtel_config_files()

        cmd = [
            str(settings.config.sim_telarray_exe),
            "-c",
            str(self.corsika_config.array_model.config_file_path),
            f"-I{config_dir}",
        ]
        weak_options = {
            "telescope_theta": self.corsika_config.zenith_angle,
            "telescope_phi": self.corsika_config.azimuth_angle,
        }
        options = {
            "histogram_file": histogram_file,
            "random_state": "none",
            "output_file": output_file,
        }

        if self.sim_telarray_seeds:
            if self.sim_telarray_seeds.get("random_instrument_instances"):
                options["random_seed"] = (
                    f"file-by-run:{config_dir}/{self.sim_telarray_seeds['seed_file_name']},auto"
                )
            elif self.sim_telarray_seeds.get("seed"):
                options["random_seed"] = self.sim_telarray_seeds["seed"]

        for key, value in options.items():
            cmd.extend(["-C", f"{key}={value}"])
        for key, value in weak_options.items():
            cmd.extend([("-W" if weak_pointing else "-C"), f"{key}={value}"])
        return cmd

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
        output_file = self.runner_service.get_file_name(
            file_type="simtel_output", run_number=run_number
        )
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
