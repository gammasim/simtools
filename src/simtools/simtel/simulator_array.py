"""Simulation runner for array simulations."""

from simtools.simtel.simtel_runner import SimtelRunner
from simtools.simulation.configuration import get_model_writer


class SimulatorArray(SimtelRunner):
    """
    SimulatorArray is the interface with sim_telarray to perform array simulations.

    Parameters
    ----------
    simulation_parameters : SimulationParameters
        Physical run settings.
    array_model : ArrayModel
        Telescope and site models.
    label: str
        Instance label.
    """

    def __init__(
        self,
        simulation_parameters,
        array_model,
        label=None,
        context=None,
    ):
        """Initialize SimulatorArray."""
        super().__init__(
            label=label, config=simulation_parameters, array_model=array_model, context=context
        )

        self.simulation_parameters = simulation_parameters
        self.array_model = array_model
        self.io_handler = self.context.paths
        self._log_file = None
        self._nsb_zenith_angle = simulation_parameters.zenith_angle
        # Directories searched before the model directory (e.g. scaled quantum-efficiency tables)
        self.include_directories = ()
        # min_photons override (e.g. 0 when an upstream stage thins photons); None: sim_telarray's
        # default. A telescope event with fewer photons than min_photons is skipped.
        self.min_photons = None

    def make_run_command(
        self, run_number=None, input_file=None, bypass_optics=None, atmospheric_transmission=None
    ):
        """
        Build and return the command to run sim_telarray.

        Parameters
        ----------
        input_file: str
            Full path of the input CORSIKA file
        run_number: int (optional)
            run number
        bypass_optics: int (optional)
            Value of sim_telarray's Bypass_Optics for photons given at the camera entrance;
            None (default) leaves the option out of the command.
        atmospheric_transmission: str or Path (optional)
            Transmission table that sim_telarray reads instead of the model's table (e.g. a
            table with optical depth 0 when an upstream stage applies the extinction).

        Returns
        -------
        list
            Command to run sim_telarray.
        """
        self.file_list = self.runner_service.load_files(run_number=run_number)
        if not self.context.args.get("write_hdata"):
            self.file_list.pop("sim_telarray_histogram", None)
        self._nsb_zenith_angle = 0.0 if bypass_optics else self.simulation_parameters.zenith_angle
        command = self._common_run_command(run_number)

        if self.simulation_parameters.is_calibration_run():
            command += self._make_run_command_for_calibration_simulations()
        else:
            command += self._make_run_command_for_shower_simulations()

        bypass = [] if bypass_optics is None else ["-C", f"Bypass_Optics={bypass_optics}"]
        if self.min_photons is not None:
            bypass += ["-C", f"min_photons={self.min_photons}"]
        transmission = (
            []
            if atmospheric_transmission is None
            else ["-C", f"atmospheric_transmission={atmospheric_transmission}"]
        )
        # "-C show=all" should be the last option
        return [*command, *bypass, *transmission, "-C", "show=all", input_file]

    def _make_run_command_for_shower_simulations(self):
        """
        Build and return the command to run sim_telarray shower simulations.

        Returns
        -------
        str
            Command to run sim_telarray.
        """
        power_law = SimulatorArray.get_power_law_for_sim_telarray_histograms(
            self.simulation_parameters.primary_particle
        )
        return [
            "-C",
            f"power_law={power_law}",
        ]

    def _make_run_command_for_calibration_simulations(self):
        """Build sim_telarray command for calibration simulations."""
        cfg = self.context.args
        altitude = self.array_model.site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        ).to_value("m")

        options = {"Altitude": altitude}
        for key in ("stars",):
            if key in cfg:
                options[key] = cfg[key]
                # stars cannot be 'None' (only 'none')
                if key == "stars" and options[key] is None:
                    options[key] = "none"

        run_mode = cfg.get("run_mode")
        if run_mode in ("pedestals", "pedestals_nsb_only"):
            n_events = cfg.get("number_of_pedestal_events", cfg["number_of_events"])
            options["pedestal_events"] = n_events
        if run_mode == "pedestals_nsb_only":
            options.update(self._pedestals_nsb_only_options())
        if run_mode == "pedestals_dark":
            n_events = cfg.get("number_of_dark_events", cfg["number_of_events"])
            options["dark_events"] = n_events
        if run_mode == "direct_injection":
            n_events = cfg.get("number_of_flasher_events", cfg["number_of_events"])
            options["laser_events"] = n_events

        cmd = []
        for key, value in options.items():
            cmd.extend(["-C", f"{key}={value}"])
        return cmd

    def _common_run_command(self, run_number):
        """Build generic run command for sim_telarray."""
        weak_pointing = self._determine_pointing_option()
        config_dir = self.array_model.get_config_directory()
        self._log_file = self.runner_service.get_file_name(
            file_type="sim_telarray_log", run_number=run_number
        )
        output_file = self.runner_service.get_file_name(
            file_type="sim_telarray_output", run_number=run_number
        )
        writer = get_model_writer(self.array_model, "sim_telarray")
        nsb_settings = {
            "zenith": self._nsb_zenith_angle,
            "manual": self.context.args.get("nsb_scaling_factor"),
            "autoscale": not self.context.args.get("disable_nsb_autoscaling", False),
        }
        if writer.array_model_file_exported is True and writer.nsb_settings != nsb_settings:
            raise ValueError(
                "NSB settings changed after configuration export; use a new array model."
            )
        writer.nsb_settings = nsb_settings
        writer.export_config_files()

        cmd = [
            str(self.context.software.sim_telarray_exe),
            "-c",
            str(self.array_model.config_file_path),
            *[f"-I{directory}" for directory in self.include_directories],
            f"-I{config_dir}",
        ]
        weak_options = {
            "telescope_theta": self.simulation_parameters.zenith_angle,
            "telescope_phi": self.simulation_parameters.azimuth_angle,
        }
        histogram_file = "/dev/null"
        if self.context.args.get("write_hdata"):
            histogram_file = self.runner_service.get_file_name(
                file_type="sim_telarray_histogram", run_number=run_number
            )
        options = {
            "histogram_file": histogram_file,
            "random_state": "none",
            "output_file": output_file,
        }

        try:
            options["random_seed"] = self.array_model.sim_telarray_seed.seed_string
        except AttributeError as exc:
            raise AttributeError("Error setting sim_telarray seed string") from exc

        for key, value in options.items():
            cmd.extend(["-C", f"{key}={value}"])
        for key, value in weak_options.items():
            cmd.extend([("-W" if weak_pointing else "-C"), f"{key}={value}"])
        return cmd

    def _determine_pointing_option(self):
        """
        Determine the pointing option for sim_telarray.

        Parameters
        ----------
        label: str
            Label of the simulation.

        Returns
        -------
        bool:
            True if weak pointing is to be used.
        """
        if isinstance(self.label, str):
            return any(pointing in self.label for pointing in ["divergent", "convergent"])
        return False

    def _pedestals_nsb_only_options(self):
        """
        Generate options to run sim_telarray for nsb-only pedestal simulations.

        Returns
        -------
        dict
            Command to run sim_telarray.
        """
        options = {}

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
        for param in null_values:
            options[param] = 0.0

        one_values = [
            "fadc_lg_var_pedestal",
            "fadc_lg_err_pedestal",
            "fadc_lg_dev_pedestal",
            "fadc_lg_sysvar_pedestal",
        ]
        for param in one_values:
            options[param] = -1.0

        return options

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
        FileNotFoundError
            If sim_telarray output file does not exist.
        """
        output_file = self.runner_service.get_file_name(
            file_type="sim_telarray_output", run_number=run_number
        )
        if not output_file.exists():
            raise FileNotFoundError(f"sim_telarray output file {output_file} does not exist.")
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
