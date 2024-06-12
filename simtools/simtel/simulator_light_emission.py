"""Simulation using the light emission package for calibration."""

import logging
import os

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
from simtools.io_operations import io_handler
from simtools.simtel.simtel_runner import SimtelRunner

__all__ = ["SimulatorLightEmission"]


class SimulatorLightEmission(SimtelRunner):
    """
    Interface with sim_telarray to perform light emission package simulations.

    The light emission package is used to simulate a artificial light source, used for calibration.

    The angle and pointing vector calculations use the convention north (x) towards
    east (-y).

    Parameters
    ----------
    telescope_model:
        Instance of the TelescopeModel class.
    calibration_model:
        CalibrationModel instance to define calibration device.
    site_model:
        SiteModel instance to define the site specific parameters.
    default_le_config: dict
        defines parameters for running the sim_telarray light emission application.
    le_application: str
        Name of the application. Default sim_telarray application running
        the sim_telarray LightEmission package is xyzls.
    simtel_source_path: str or Path
        Location of sim_telarray installation.
    light_source_type: str
        The light source type.
    label: str
        Label for output directory.
    config_data: dict
        Dict containing the configurable parameters.
    config_file: str or Path
        Path of the yaml file containing the configurable parameters.
    test: bool
        Test flag.
    """

    def __init__(
        self,
        telescope_model,
        calibration_model,
        site_model,
        default_le_config,
        le_application,
        simtel_source_path,
        light_source_type,
        label=None,
        config_data=None,
        config_file=None,
        test=False,
    ):
        """Initialize SimtelRunner."""
        super().__init__(label=label, simtel_source_path=simtel_source_path)

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerLightEmission")
        self._simtel_source_path = simtel_source_path

        self._telescope_model = telescope_model

        self.label = label if label is not None else self._telescope_model.label

        self._calibration_model = calibration_model
        self._site_model = site_model
        self.io_handler = io_handler.IOHandler()
        self.output_directory = self.io_handler.get_output_directory(self.label)
        try:
            self.config = gen.validate_config_data(
                gen.collect_data_from_file_or_dict(config_file, config_data),
                self.light_emission_default_configuration(),
            )
        except TypeError:
            self.config = gen.validate_config_data(
                {},
                self.light_emission_default_configuration(),
            )

        # LightEmission - default parameters
        self._rep_number = 0
        self.runs = 1
        self.photons_per_run = 1e10 if not test else 1e7

        self.le_application = le_application
        self.default_le_config = default_le_config
        self.distance = self.telescope_calibration_device_distance()
        self.light_source_type = light_source_type
        self._telescope_model.export_config_file()

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Build a LightEmission object from kwargs only.

        The configurable parameters can be given as kwargs, instead of using the
        config_data or config_file arguments.

        Parameters
        ----------
        kwargs
            Containing the arguments and the configurable parameters.

        Returns
        -------
        Instance of this class.
        """
        args, config_data = gen.separate_args_and_config_data(
            expected_args=[
                "telescope_model",
                "calibration_model",
                "site_model",
                "default_le_config",
                "le_application",
                "simtel_source_path",
                "label",
                "light_source_type",
            ],
            **kwargs,
        )

        return cls(**args, config_data=config_data)

    @staticmethod
    def light_emission_default_configuration():
        """
        Get default light emission configuration.

        Returns
        -------
        dict
            Default configuration light emission.

        """
        return {
            "zenith_angle": {
                "len": 1,
                "unit": u.Unit("deg"),
                "default": 0.0 * u.deg,
                "names": ["zenith", "theta"],
            },
            "azimuth_angle": {
                "len": 1,
                "unit": u.Unit("deg"),
                "default": 0.0 * u.deg,
                "names": ["azimuth", "phi"],
            },
            "source_distance": {
                "len": 1,
                "unit": u.Unit("m"),
                "default": 1000 * u.m,
                "names": ["sourcedist", "srcdist"],
            },
            "off_axis_angle": {
                "len": 1,
                "unit": u.Unit("deg"),
                "default": 0 * u.deg,
                "names": ["off_axis"],
            },
            "fadc_bins": {
                "len": 1,
                "unit": u.dimensionless_unscaled,
                "default": 128,
                "names": ["fadc_bins"],
            },
        }

    def calibration_pointing_direction(self):
        """
        Calculate the pointing of the calibration device towards the telescope.

        Returns
        -------
        list
            The pointing vector from the calibration device to the telescope.
        """
        # x_cal = self._calibration_model.get_parameter_value("x_pos")
        # y_cal = self._calibration_model.get_parameter_value("y_pos")
        # z_cal = self._calibration_model.get_parameter_value("z_pos")
        x_cal = self.default_le_config["x_pos_ILLN-01"]["default"].to(u.m).value
        y_cal = self.default_le_config["y_pos_ILLN-01"]["default"].to(u.m).value
        z_cal = self.default_le_config["z_pos_ILLN-01"]["default"].to(u.m).value

        cal_vect = np.array([x_cal, y_cal, z_cal])
        # x_tel = self._telescope_model.get_parameter_value("x_pos")
        # y_tel = self._telescope_model.get_parameter_value("y_pos")
        # z_tel = self._telescope_model.get_parameter_value("z_pos")
        x_tel = self.default_le_config["x_pos"]["real"].to(u.m).value
        y_tel = self.default_le_config["y_pos"]["real"].to(u.m).value
        z_tel = self.default_le_config["z_pos"]["real"].to(u.m).value
        tel_vect = np.array([x_tel, y_tel, z_tel])

        direction_vector = tel_vect - cal_vect
        # pointing vector from calibration device to telescope

        pointing_vector = np.round(direction_vector / np.linalg.norm(direction_vector), 6)

        # Calculate telescope theta and phi angles
        tel_theta = 180 - np.round(
            np.rad2deg(np.arccos(direction_vector[2] / np.linalg.norm(direction_vector))), 6
        )
        tel_phi = 180 - np.round(
            np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0])), 6
        )

        # Calculate laser beam theta and phi angles
        direction_vector_inv = direction_vector * -1
        laser_theta = np.round(
            np.rad2deg(np.arccos(direction_vector_inv[2] / np.linalg.norm(direction_vector_inv))), 6
        )
        laser_phi = np.round(
            np.rad2deg(np.arctan2(direction_vector_inv[1], direction_vector_inv[0])), 6
        )
        return pointing_vector.tolist(), [tel_theta, tel_phi, laser_theta, laser_phi]

    def telescope_calibration_device_distance(self):
        """
        Calculate the distance between the telescope and the calibration device.

        Returns
        -------
        astropy Quantity
            The distance between the telescope and the calibration device.
        """
        if "real" in self.default_le_config["x_pos"]:
            x_cal = self.default_le_config["x_pos_ILLN-01"]["default"].to(u.m).value
            y_cal = self.default_le_config["y_pos_ILLN-01"]["default"].to(u.m).value
            z_cal = self.default_le_config["z_pos_ILLN-01"]["default"].to(u.m).value

            x_tel = self.default_le_config["x_pos"]["real"].to(u.m).value
            y_tel = self.default_le_config["y_pos"]["real"].to(u.m).value
            z_tel = self.default_le_config["z_pos"]["real"].to(u.m).value

        else:
            x_tel = self.default_le_config["x_pos"]["default"].to(u.m).value
            y_tel = self.default_le_config["y_pos"]["default"].to(u.m).value
            z_tel = self.default_le_config["z_pos"]["default"].to(u.m).value

            x_cal, y_cal, z_cal = 0, 0, 0

        tel_vect = np.array([x_tel, y_tel, z_tel])
        cal_vect = np.array([x_cal, y_cal, z_cal])
        distance = np.linalg.norm(cal_vect - tel_vect)

        return distance * u.m

    def _make_light_emission_script(self):
        """
        Create the light emission script to run the light emission package.

        Require the specified pre-compiled light emission package application
        in the sim_telarray/LightEmission/ path.

        Returns
        -------
        str
            The commands to run the Light Emission package
        """
        command = f" rm {self.output_directory}/"
        command += f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n"
        command += str(self._simtel_source_path.joinpath("sim_telarray/LightEmission/"))
        command += f"/{self.le_application[0]}"

        if self.light_source_type == "led":
            if self.le_application[1] == "variable":
                command += f" -x {self.default_le_config['x_pos']['default'].to(u.cm).value}"
                command += f" -y {self.default_le_config['y_pos']['default'].to(u.cm).value}"
                command += f" -z {self.default_le_config['z_pos']['default'].to(u.cm).value}"
                command += (
                    f" -d {','.join(map(str, self.default_le_config['direction']['default']))}"
                )
                command += f" -n {self._calibration_model.get_parameter_value('photons_per_run')}"

            elif self.le_application[1] == "layout":
                x_origin = (
                    self.default_le_config["x_pos_ILLN-01"]["default"]
                    - self.default_le_config["x_pos"]["real"]
                )
                y_origin = (
                    self.default_le_config["y_pos_ILLN-01"]["default"]
                    - self.default_le_config["y_pos"]["real"]
                )
                z_origin = (
                    self.default_le_config["z_pos_ILLN-01"]["default"]
                    - self.default_le_config["z_pos"]["real"]
                )
                # light_source coordinates relative to telescope
                command += f" -x {x_origin.to(u.cm).value}"
                command += f" -y {y_origin.to(u.cm).value}"
                command += f" -z {z_origin.to(u.cm).value}"
                pointing_vector = self.calibration_pointing_direction()[0]
                command += f" -d {','.join(map(str, pointing_vector))}"

                command += f" -n {self._calibration_model.get_parameter_value('photons_per_run')}"

                # same wavelength as for laser
                command += f" -s {self._calibration_model.get_parameter_value('laser_wavelength')}"

                # pulse
                command += (
                    f" -p Gauss:{self._calibration_model.get_parameter_value('led_pulse_sigtime')}"
                )
                command += " -a isotropic"  # angular distribution
                # {self._calibration_model.get_parameter_value('led_pulse_offset')}"
                # TODO further parameters require modification of application

                # command += f" -s {self._calibration_model.get_parameter_value('led_var_photons')}"
                # command += f" -s {self._calibration_model.get_parameter_value('pedestal_events')}"
            command += f" -A {self.output_directory}/model/"
            command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"
        elif self.light_source_type == "laser":
            # TODO: this option requires the atmospheric profiles in the include directory,
            # or adjusting the application to use a path
            command += " --events 1"
            command += " --bunches 2500000"
            command += " --step 0.1"
            command += " --bunchsize 1"
            command += (
                f" --spectrum {self._calibration_model.get_parameter_value('laser_wavelength')}"
            )
            command += " --lightpulse Gauss:"
            command += f"{self._calibration_model.get_parameter_value('laser_pulse_sigtime')}"
            # command += " --angular-distribution Gauss:0.1" # specify laser angular distribution

            x_origin = (
                self.default_le_config["x_pos_ILLN-01"]["default"]
                - self.default_le_config["x_pos"]["real"]
            )
            y_origin = (
                self.default_le_config["y_pos_ILLN-01"]["default"]
                - self.default_le_config["y_pos"]["real"]
            )
            z_origin = (
                self.default_le_config["z_pos_ILLN-01"]["default"]
                - self.default_le_config["z_pos"]["real"]
            )
            _, angles = self.calibration_pointing_direction()
            angle_theta = angles[0]
            angle_phi = angles[1]
            command += f" --laser-position '{x_origin.value},{y_origin.value},{z_origin.value}'"

            command += f" --telescope-theta {angle_theta}"
            command += f" --telescope-phi {angle_phi}"
            command += f" --laser-theta {90-angles[2]}"
            command += f" --laser-phi {angles[3]}"  # convention north (x) towards east (-y)

            # further optional properties not used here:
            # 'laser_external_trigger', 'laser_pulse_exptime',
            # 'laser_pulse_offset' 'laser_pulse_twidth'

            command += f" --atmosphere {self.output_directory}/model/"
            command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"
        command += f" -o {self.output_directory}/{self.le_application[0]}.iact.gz"
        command += "\n"

        return command

    def _make_simtel_script(self):
        """
        Return the command to run simtel_array using the output from the previous step.

        Returns
        -------
        str
            The command to run simtel_array
        """
        # LightEmission
        command = f"{self._simtel_source_path.joinpath('sim_telarray/bin/sim_telarray/')}"
        command += f" -c {self._telescope_model.get_config_file()}"

        def remove_line_from_config(file_path, line_prefix):
            with open(file_path, encoding="utf-8") as file:
                lines = file.readlines()

            with open(file_path, "w", encoding="utf-8") as file:
                for line in lines:
                    if not line.startswith(line_prefix):
                        file.write(line)

        remove_line_from_config(self._telescope_model.get_config_file(), "array_triggers")

        command += " -DNUM_TELESCOPES=1"
        command += " -I../cfg/CTA"
        command += "iobuf_maximum=1000000000"
        command += super()._config_option(
            "altitude", self._site_model.get_parameter_value("corsika_observation_level")
        )
        command += super()._config_option(
            "atmospheric_transmission",
            self._telescope_model.get_parameter_value("atmospheric_transmission"),
        )
        # command += super()._config_option("show", "all") # for debugging
        command += super()._config_option("TRIGGER_CURRENT_LIMIT", "20")
        command += super()._config_option("TRIGGER_TELESCOPES", "1")
        command += super()._config_option("TELTRIG_MIN_SIGSUM", "7.8")
        command += super()._config_option("PULSE_ANALYSIS", "-30")

        if "real" in self.default_le_config["x_pos"]:
            _, angles = self.calibration_pointing_direction()
            command += super()._config_option("telescope_theta", f"{angles[0]}")
            command += super()._config_option("telescope_phi", f"{angles[1]}")
        else:
            command += super()._config_option("telescope_theta", 0)
            command += super()._config_option("telescope_phi", 0)

        command += super()._config_option("power_law", "2.68")
        command += super()._config_option(
            "input_file", f"{self.output_directory}/{self.le_application[0]}.iact.gz"
        )
        command += super()._config_option(
            "output_file",
            f"{self.output_directory}/"
            f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n",
        )

        return command

    def _create_postscript(self, **kwargs):
        """
        Write out post-script file using read_cta in hessioxxx/bin/read_cta.

        parts from the documentation
        -r level        (Use 10/5 tail-cut image cleaning and redo reconstruction.)
                level >= 1: show parameters from sim_hessarray.
                level >= 2: redo shower reconstruction
                level >= 3: redo image cleaning (and shower reconstruction
                            with new image parameters)
                level >= 4: redo amplitude summation
                level >= 5: PostScript file includes original and
                            new shower reconstruction.
        --integration-window w,o[,ps] *(Set integration window width and offset.)
            For some integration schemes there is a pulse shaping option.


        Returns
        -------
        str
            Command to create the postscript file
        """
        postscript_dir = self.output_directory.joinpath("postscripts")
        postscript_dir.mkdir(parents=True, exist_ok=True)

        command = str(self._simtel_source_path.joinpath("hessioxxx/bin/read_cta"))
        command += " --min-tel 1 --min-trg-tel 1"
        command += " -q --integration-scheme 4"
        command += " --integration-window "
        command += f"{kwargs['integration_window'][0]},{kwargs['integration_window'][1]}"
        command += f" -r {kwargs['level']}"
        command += " --plot-with-sum-only"
        command += " --plot-with-pixel-amp --plot-with-pixel-id"
        command += (
            f" -p {postscript_dir}/"
            f"{self.le_application[0]}_{self.le_application[1]}_"
            f"d_{int(self.distance.to(u.m).value)}.ps"
        )
        command += (
            f" {self.output_directory}/"
            f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n"
        )
        # ps2pdf required, now only store postscripts
        # command += f"ps2pdf {self.output_directory}/{self.le_application}.ps
        #  {self.output_directory}/{self.le_application}.pdf"
        return command

    def prepare_script(self, generate_postscript=False, **kwargs):
        """
        Build and return bash run script containing the light-emission command.

        Parameters
        ----------
        plot: bool
            If output should be plotted.

        generate_postscript: bool
            If postscript should be generated with read_cta.

        Returns
        -------
        Path
            Full path of the run script.
        """
        self._logger.debug("Creating run bash script")

        self._script_dir = self.output_directory.joinpath("scripts")
        self._script_dir.mkdir(parents=True, exist_ok=True)
        self._script_file = self._script_dir.joinpath(f"{self.le_application[0]}-lightemission.sh")
        self._logger.debug(f"Run bash script - {self._script_file}")

        command_le = self._make_light_emission_script()
        command_simtel = self._make_simtel_script()

        with self._script_file.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")

            file.write(f"{command_le}\n\n")
            file.write(f"{command_simtel}\n\n")

            if generate_postscript:
                self._logger.debug("Write out postscript file")
                command_plot = self._create_postscript(**kwargs)
                file.write("# Generate postscript\n\n")
                file.write(f"{command_plot}\n\n")
                file.write("# End\n\n")

        os.system(f"chmod ug+x {self._script_file}")
        return self._script_file
