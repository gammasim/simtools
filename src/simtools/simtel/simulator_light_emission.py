"""Simulation using the light emission package for calibration."""

import logging
import stat
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.io_operations import io_handler
from simtools.runners.simtel_runner import SimtelRunner
from simtools.utils.general import clear_default_sim_telarray_cfg_directories

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
    simtel_path: str or Path
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
        simtel_path,
        light_source_type,
        label=None,
        test=False,
    ):
        """Initialize SimtelRunner."""
        super().__init__(label=label, simtel_path=simtel_path)

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerLightEmission")
        self._simtel_path = simtel_path

        self._telescope_model = telescope_model

        self.label = label if label is not None else self._telescope_model.label

        self._calibration_model = calibration_model
        self._site_model = site_model
        self.io_handler = io_handler.IOHandler()
        self.output_directory = self.io_handler.get_output_directory(self.label)

        # LightEmission - default parameters
        self._rep_number = 0
        self.runs = 1
        self.photons_per_run = (
            self._calibration_model.get_parameter_value("photons_per_run") if not test else 1e7
        )

        self.le_application = le_application
        self.default_le_config = default_le_config
        self.distance = self.telescope_calibration_device_distance()
        self.light_source_type = light_source_type
        self._telescope_model.export_config_file()
        self.test = test

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
        # use DB coordinates later
        x_cal, y_cal, z_cal = self._calibration_model.get_parameter_value(
            "array_element_position_ground"
        )
        cal_vect = np.array([x_cal, y_cal, z_cal])
        x_tel, y_tel, z_tel = self._telescope_model.get_parameter_value(
            "array_element_position_ground"
        )

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
        if not self.default_le_config:
            x_cal, y_cal, z_cal = self._calibration_model.get_parameter_value(
                "array_element_position_ground"
            )
            x_tel, y_tel, z_tel = self._telescope_model.get_parameter_value(
                "array_element_position_ground"
            )

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
        x_cal, y_cal, z_cal = (
            self._calibration_model.get_parameter_value("array_element_position_ground") * u.m
        )
        x_tel, y_tel, z_tel = (
            self._telescope_model.get_parameter_value("array_element_position_ground") * u.m
        )
        _model_directory = self.io_handler.get_output_directory(self.label, "model")
        command = f" rm {self.output_directory}/"
        command += f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n"
        command += str(self._simtel_path.joinpath("sim_telarray/LightEmission/"))
        command += f"/{self.le_application[0]}"

        if self.light_source_type == "led":
            if self.le_application[1] == "variable":
                command += f" -x {self.default_le_config['x_pos']['default'].to(u.cm).value}"
                command += f" -y {self.default_le_config['y_pos']['default'].to(u.cm).value}"
                command += f" -z {self.default_le_config['z_pos']['default'].to(u.cm).value}"
                command += (
                    f" -d {','.join(map(str, self.default_le_config['direction']['default']))}"
                )
                command += f" -n {self.photons_per_run}"

            elif self.le_application[1] == "layout":
                x_origin = x_cal - x_tel
                y_origin = y_cal - y_tel
                z_origin = z_cal - z_tel
                # light_source coordinates relative to telescope
                command += f" -x {x_origin.to(u.cm).value}"
                command += f" -y {y_origin.to(u.cm).value}"
                command += f" -z {z_origin.to(u.cm).value}"
                pointing_vector = self.calibration_pointing_direction()[0]
                command += f" -d {','.join(map(str, pointing_vector))}"

                command += f" -n {self.photons_per_run}"

                # same wavelength as for laser
                command += f" -s {self._calibration_model.get_parameter_value('laser_wavelength')}"

                # pulse
                command += (
                    f" -p Gauss:{self._calibration_model.get_parameter_value('led_pulse_sigtime')}"
                )
                command += " -a isotropic"  # angular distribution

            command += f" -A {_model_directory}/"
            command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"

        elif self.light_source_type == "laser":
            command += " --events 1"
            command += " --bunches 2500000"
            command += " --step 0.1"
            command += " --bunchsize 1"
            command += (
                f" --spectrum {self._calibration_model.get_parameter_value('laser_wavelength')}"
            )
            command += " --lightpulse Gauss:"
            command += f"{self._calibration_model.get_parameter_value('laser_pulse_sigtime')}"
            x_origin = x_cal - x_tel
            y_origin = y_cal - y_tel
            z_origin = z_cal - z_tel
            _, angles = self.calibration_pointing_direction()
            angle_theta = angles[0]
            angle_phi = angles[1]
            positions = x_origin.to(u.cm).value, y_origin.to(u.cm).value, z_origin.to(u.cm).value
            command += f" --laser-position '{positions[0]},{positions[1]},{positions[2]}'"

            command += f" --telescope-theta {angle_theta}"
            command += f" --telescope-phi {angle_phi}"
            command += f" --laser-theta {90 - angles[2]}"
            command += f" --laser-phi {angles[3]}"  # convention north (x) towards east (-y)
            command += f" --atmosphere {_model_directory}/"
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
        _, angles = self.calibration_pointing_direction()

        command = f"{self._simtel_path.joinpath('sim_telarray/bin/sim_telarray/')}"
        command += " -I"
        command += f" -I{self._telescope_model.config_file_directory}"
        command += f" -c {self._telescope_model.get_config_file(no_export=True)}"
        if not self.test:
            self._remove_line_from_config(
                self._telescope_model.get_config_file(no_export=True), "array_triggers"
            )
            self._remove_line_from_config(
                self._telescope_model.get_config_file(no_export=True), "axes_offsets"
            )

        command += " -DNUM_TELESCOPES=1"

        command += super().get_config_option(
            "altitude", self._site_model.get_parameter_value("corsika_observation_level")
        )
        command += super().get_config_option(
            "atmospheric_transmission",
            self._telescope_model.get_parameter_value("atmospheric_transmission"),
        )
        command += super().get_config_option("TRIGGER_TELESCOPES", "1")

        command += super().get_config_option("TELTRIG_MIN_SIGSUM", "2")
        command += super().get_config_option("PULSE_ANALYSIS", "-30")
        command += super().get_config_option("MAXIMUM_TELESCOPES", 1)

        if self.le_application[1] == "variable":
            command += super().get_config_option("telescope_theta", 0)
            command += super().get_config_option("telescope_phi", 0)
        else:
            command += super().get_config_option("telescope_theta", f"{angles[0]}")
            command += super().get_config_option("telescope_phi", f"{angles[1]}")

        command += super().get_config_option("power_law", "2.68")
        command += super().get_config_option(
            "input_file", f"{self.output_directory}/{self.le_application[0]}.iact.gz"
        )
        command += super().get_config_option(
            "output_file",
            f"{self.output_directory}/{self.le_application[0]}_{self.le_application[1]}.simtel.gz",
        )
        command += super().get_config_option(
            "histogram_file",
            f"{self.output_directory}/"
            f"{self.le_application[0]}_{self.le_application[1]}.ctsim.hdata\n",
        )

        # Remove the default sim_telarray configuration directories
        return clear_default_sim_telarray_cfg_directories(command)

    def _remove_line_from_config(self, file_path, line_prefix):
        """
        Remove lines starting with a specific prefix from the config.

        Parameters
        ----------
        file_path : Path
            The path to the configuration file.
        line_prefix : str
            The prefix of lines to be removed.
        """
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as file:
            lines = file.readlines()

        with file_path.open("w", encoding="utf-8") as file:
            for line in lines:
                if not line.startswith(line_prefix):
                    file.write(line)

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

        command = str(self._simtel_path.joinpath("hessioxxx/bin/read_cta"))
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

        _script_dir = self.output_directory.joinpath("scripts")
        _script_dir.mkdir(parents=True, exist_ok=True)
        _script_file = _script_dir.joinpath(f"{self.le_application[0]}-lightemission.sh")
        self._logger.debug(f"Run bash script - {_script_file}")

        command_le = self._make_light_emission_script()
        command_simtel = self._make_simtel_script()

        with _script_file.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")

            file.write(f"{command_le}\n\n")
            file.write(f"{command_simtel}\n\n")

            if generate_postscript:
                self._logger.debug("Write out postscript file")
                command_plot = self._create_postscript(**kwargs)
                file.write("# Generate postscript\n\n")
                file.write(f"{command_plot}\n\n")
                file.write("# End\n\n")

        _script_file.chmod(_script_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        return _script_file
