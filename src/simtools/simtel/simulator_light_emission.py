"""Simulation using the light emission package for calibration and flasher devices."""

import logging
import stat
import subprocess
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.corsika.corsika_histograms_visualize import save_figs_to_pdf
from simtools.io import io_handler
from simtools.runners.simtel_runner import SimtelRunner
from simtools.utils.general import clear_default_sim_telarray_cfg_directories
from simtools.visualization.visualize import plot_simtel_ctapipe

__all__ = ["SimulatorLightEmission"]


class SimulatorLightEmission(SimtelRunner):
    """
    Interface with sim_telarray to perform light emission package simulations.

    The light emission package is used to simulate an artificial light source, used for calibration.
    """

    def __init__(
        self,
        telescope_model,
        calibration_model=None,
        flasher_model=None,
        site_model=None,
        light_emission_config=None,
        le_application=None,
        simtel_path=None,
        light_source_type=None,
        label=None,
        test=False,
    ):
        """Initialize SimtelRunner.

        Parameters
        ----------
        telescope_model : TelescopeModel
            Model of the telescope to be simulated
        calibration_model : CalibrationModel, optional
            Model of the calibration device to be simulated
        flasher_model : FlasherModel, optional
            Model of the flasher device to be simulated
        site_model : SiteModel, optional
            Model of the site
        light_emission_config : dict, optional
            Configuration for the light emission
        le_application : tuple, optional
            Light emission application to be used
        simtel_path : Path, optional
            Path to the sim_telarray installation
        light_source_type : str, optional
            Type of light source: 'led', 'laser', or 'flasher'
        label : str, optional
            Label for the simulation
        test : bool, optional
            Whether this is a test run
        """
        super().__init__(label=label, simtel_path=simtel_path)

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerLightEmission")
        self._simtel_path = simtel_path

        self._telescope_model = telescope_model

        self.label = label if label is not None else self._telescope_model.label
        self.test = test

        self._calibration_model = calibration_model
        self._flasher_model = flasher_model
        self._site_model = site_model
        self.io_handler = io_handler.IOHandler()
        self.output_directory = self.io_handler.get_output_directory(self.label)

        # LightEmission - default parameters
        self._rep_number = 0
        self.runs = 1

        # Set photons per run based on the model (calibration or flasher)
        if self._calibration_model is not None:
            self.photons_per_run = (
                (self._calibration_model.get_parameter_value("photons_per_run"))
                if not self.test
                else 1e8
            )
        elif self._flasher_model is not None:
            self.photons_per_run = (
                (self._flasher_model.get_parameter_value("photons_per_flasher"))
                if not self.test
                else 1e8
            )
        else:
            self.photons_per_run = 1e8

        self.le_application = le_application
        self.light_emission_config = light_emission_config
        self.distance = None
        self.light_source_type = light_source_type
        self._telescope_model.write_sim_telarray_config_file(additional_model=site_model)

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

    @staticmethod
    def flasher_default_configuration():
        """
        Get default flasher configuration.

        Returns
        -------
        dict
            Default configuration for flasher devices.
        """
        return {
            "events": {
                "len": 1,
                "unit": None,
                "default": 1,
                "names": ["events"],
            },
            "photons_per_flasher": {
                "len": 1,
                "unit": None,
                "default": 2.5e6,
                "names": ["photons"],
            },
            "bunch_size": {
                "len": 1,
                "unit": None,
                "default": 1.0,
                "names": ["bunchsize"],
            },
            "flasher_position": {
                "len": 2,
                "unit": u.Unit("cm"),
                "default": [0.0, 0.0] * u.cm,
                "names": ["xy", "position"],
            },
            "flasher_depth": {
                "len": 1,
                "unit": u.Unit("cm"),
                "default": 2.46 * u.cm,
                "names": ["depth", "distance"],
            },
            "flasher_inclination": {
                "len": 1,
                "unit": u.Unit("deg"),
                "default": 0.0 * u.deg,
                "names": ["inclination"],
            },
            "spectrum": {
                "len": 1,
                "unit": u.Unit("nm"),
                "default": 400 * u.nm,
                "names": ["wavelength"],
            },
            "lightpulse": {
                "len": 1,
                "unit": None,
                "default": "Simple:0",
                "names": ["pulse"],
            },
            "angular_distribution": {
                "len": 1,
                "unit": None,
                "default": "isotropic",
                "names": ["angular"],
            },
            "flasher_pattern": {
                "len": 1,
                "unit": None,
                "default": "all",
                "names": ["fire", "pattern"],
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
        x_cal, y_cal, z_cal = self._calibration_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )
        x_cal, y_cal, z_cal = [coord.to(u.m).value for coord in (x_cal, y_cal, z_cal)]
        cal_vect = np.array([x_cal, y_cal, z_cal])
        x_tel, y_tel, z_tel = self._telescope_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )
        x_tel, y_tel, z_tel = [coord.to(u.m).value for coord in (x_tel, y_tel, z_tel)]

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

    def _write_telpos_file(self):
        """
        Write the telescope positions to a telpos file.

        The file will contain lines in the format: x y z r in cm

        Returns
        -------
        Path
            The path to the generated telpos file.
        """
        telpos_file = self.output_directory.joinpath("telpos.dat")
        x_tel, y_tel, z_tel = self._telescope_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )
        x_tel, y_tel, z_tel = [coord.to(u.cm).value for coord in (x_tel, y_tel, z_tel)]

        radius = self._telescope_model.get_parameter_value_with_unit("telescope_sphere_radius")
        radius = radius.to(u.cm).value  # Convert radius to cm
        with telpos_file.open("w", encoding="utf-8") as file:
            file.write(f"{x_tel} {y_tel} {z_tel} {radius}\n")

        return telpos_file

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
        x_tel, y_tel, z_tel = self._telescope_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )

        config_directory = self.io_handler.get_output_directory(
            label=self.label, sub_dir=f"model/{self._site_model.model_version}"
        )

        telpos_file = self._write_telpos_file()

        command = f"rm {self.output_directory}/"
        command += f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n"
        command += str(self._simtel_path.joinpath("sim_telarray/LightEmission/"))
        command += f"/{self.le_application[0]}"
        corsika_observation_level = self._site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        )
        command += f" -h  {corsika_observation_level.to(u.m).value}"
        command += f" --telpos-file {telpos_file}"

        if self.light_source_type == "flasher":
            command = self._add_flasher_command_options(command)
        elif self.light_source_type == "led":
            command = self._add_led_command_options(command)
        elif self.light_source_type == "laser":
            command = self._add_laser_command_options(
                command, x_tel, y_tel, z_tel, config_directory
            )

        # Add common options for the specific light source type
        if self.light_source_type in ["flasher", "led"]:
            command += f" -A {config_directory}/"
            command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"

        command += f" -o {self.output_directory}/{self.le_application[0]}.iact.gz"
        command += "\n"

        return command

    def _add_flasher_command_options(self, command):
        """
        Add flasher-specific command options to the light emission script.

        Parameters
        ----------
        command : str
            The command string to add options to

        Returns
        -------
        str
            The updated command string
        """
        telescope_name = self._telescope_model.name
        if "SST" in telescope_name:
            command = self._add_sst_flasher_options(command)
        else:
            command = self._add_mst_lst_flasher_options(command)
        return command

    def _add_sst_flasher_options(self, command):
        """
        Add SST-specific flasher options to the command.

        Parameters
        ----------
        command : str
            The command string to add options to

        Returns
        -------
        str
            The updated command string
        """
        # For SST we use ff-gct style flashers (dual-mirror design)
        flasher_xy = self._flasher_model.get_parameter_value("flasher_position")
        flasher_depth = self._flasher_model.get_parameter_value("flasher_depth")
        flasher_inclination = self._flasher_model.get_parameter_value("flasher_inclination")
        mirror_camera_distance = self._flasher_model.get_parameter_value("mirror_camera_distance")
        spectrum = self._flasher_model.get_parameter_value("spectrum")
        pulse = self._flasher_model.get_parameter_value("lightpulse")
        angular = self._flasher_model.get_parameter_value("angular_distribution")
        fire_pattern = self._flasher_model.get_parameter_value("flasher_pattern")
        bunch_size = self._flasher_model.get_parameter_value("bunch_size")

        command += f" --events {self.runs}"
        command += f" --photons {self.photons_per_run}"
        command += f" --bunchsize {bunch_size}"
        command += f" --flasher-xy {flasher_xy[0]}"
        command += f" --flasher-depth {flasher_depth}"
        command += f" --flasher-inclination {flasher_inclination}"
        command += f" --mirror-camera-distance {mirror_camera_distance}"
        command += f" --spectrum {spectrum}"
        command += f" --lightpulse {pulse}"
        command += f" --angular-distribution {angular}"
        command += f" --fire {fire_pattern}"
        return command

    def _add_mst_lst_flasher_options(self, command):
        """
        Add MST/LST-specific flasher options to the command.

        Parameters
        ----------
        command : str
            The command string to add options to

        Returns
        -------
        str
            The updated command string
        """
        # For MST/LST we use ff-1m style flashers (single-mirror design)
        flasher_xy = self._flasher_model.get_parameter_value("flasher_position")
        flasher_distance = self._flasher_model.get_parameter_value("flasher_depth")
        camera_radius = (
            self._telescope_model.get_parameter_value_with_unit("camera_radius").to(u.cm).value
        )
        spectrum = self._flasher_model.get_parameter_value("spectrum")
        pulse = self._flasher_model.get_parameter_value("lightpulse")
        angular = self._flasher_model.get_parameter_value("angular_distribution")
        bunch_size = self._flasher_model.get_parameter_value("bunch_size")

        command += f" --events {self.runs}"
        command += f" --photons {self.photons_per_run}"
        command += f" --bunchsize {bunch_size}"
        command += f" --xy {flasher_xy[0]},{flasher_xy[1]}"
        command += f" --distance {flasher_distance}"
        command += f" --camera-radius {camera_radius}"
        command += f" --spectrum {spectrum}"
        command += f" --lightpulse {pulse}"
        command += f" --angular-distribution {angular}"
        return command

    def _add_led_command_options(self, command):
        """
        Add LED-specific command options to the light emission script.

        Parameters
        ----------
        command : str
            The command string to add options to

        Returns
        -------
        str
            The updated command string
        """
        if self.le_application[1] == "variable":
            command += f" -x {self.light_emission_config['x_pos']['default'].to(u.cm).value}"
            command += f" -y {self.light_emission_config['y_pos']['default'].to(u.cm).value}"
            command += f" -z {self.light_emission_config['z_pos']['default'].to(u.cm).value}"
            command += (
                f" -d {','.join(map(str, self.light_emission_config['direction']['default']))}"
            )
            command += f" -n {self.photons_per_run}"

        elif self.le_application[1] == "layout":
            x_cal, y_cal, z_cal = self._calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
            command += f" -x {x_cal.to(u.cm).value}"
            command += f" -y {y_cal.to(u.cm).value}"
            command += f" -z {z_cal.to(u.cm).value}"
            pointing_vector = self.calibration_pointing_direction()[0]
            command += f" -d {','.join(map(str, pointing_vector))}"

            command += f" -n {self.photons_per_run}"
            self._logger.info(f"Photons per run: {self.photons_per_run} ")

            laser_wavelength = self._calibration_model.get_parameter_value_with_unit(
                "laser_wavelength"
            )
            command += f" -s {int(laser_wavelength.to(u.nm).value)}"

            led_pulse_sigtime = self._calibration_model.get_parameter_value_with_unit(
                "led_pulse_sigtime"
            )
            command += f" -p Gauss:{led_pulse_sigtime.to(u.ns).value}"
            command += " -a isotropic"

        return command

    def _add_laser_command_options(self, command, x_tel, y_tel, z_tel, config_directory):
        """
        Add laser-specific command options to the light emission script.

        Parameters
        ----------
        command : str
            The command string to add options to
        x_tel : astropy.units.Quantity
            X position of the telescope
        y_tel : astropy.units.Quantity
            Y position of the telescope
        z_tel : astropy.units.Quantity
            Z position of the telescope
        config_directory : Path
            Path to the configuration directory

        Returns
        -------
        str
            The updated command string
        """
        x_cal, y_cal, z_cal = self._calibration_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )
        command += " --events 1"
        command += " --bunches 2500000"
        command += " --step 0.1"
        command += " --bunchsize 1"
        spectrum = self._calibration_model.get_parameter_value_with_unit("laser_wavelength")
        command += f" --spectrum {int(spectrum.to(u.nm).value)}"
        command += " --lightpulse Gauss:"
        pulse_sigtime = self._calibration_model.get_parameter_value_with_unit("laser_pulse_sigtime")
        command += f"{pulse_sigtime.to(u.ns).value}"
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
        command += f" --laser-phi {angles[3]}"
        command += f" --atmosphere {config_directory}/"
        command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"

        return command

    def _make_simtel_script(self):
        """
        Return the command to run sim_telarray using the output from the previous step.

        Returns
        -------
        str
            The command to run sim_telarray
        """
        # LightEmission
        _, angles = self.calibration_pointing_direction()

        command = f"{self._simtel_path.joinpath('sim_telarray/bin/sim_telarray/')}"
        command += " -I"
        command += f" -I{self._telescope_model.config_file_directory}"
        command += f" -c {self._telescope_model.config_file_path}"
        self._remove_line_from_config(self._telescope_model.config_file_path, "array_triggers")
        self._remove_line_from_config(self._telescope_model.config_file_path, "axes_offsets")

        command += " -DNUM_TELESCOPES=1"

        command += super().get_config_option(
            "altitude",
            self._site_model.get_parameter_value_with_unit("corsika_observation_level")
            .to(u.m)
            .value,
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

    def process_simulation_output(self, args_dict, figures):
        """Process the simulation output, including plotting and saving figures."""
        try:
            filename = self._get_simulation_output_filename()
            distance = self._get_distance_for_plotting()

            fig = self._plot_simulation_output(
                filename,
                args_dict["boundary_thresh"],
                args_dict["picture_thresh"],
                args_dict["min_neighbors"],
                distance,
                args_dict["return_cleaned"],
            )
            figures.append(fig)

        except AttributeError:
            msg = (
                f"Telescope not triggered at distance of "
                f"{self.light_emission_config['z_pos']['default']}"
            )
            self._logger.warning(msg)

    def _get_simulation_output_filename(self):
        """Get the filename of the simulation output."""
        return (
            f"{self.output_directory}/{self.le_application[0]}_{self.le_application[1]}.simtel.gz"
        )

    def _get_distance_for_plotting(self):
        """Get the distance to be used for plotting."""
        try:
            return self.light_emission_config["z_pos"]["default"]
        except KeyError:
            return round(self.distance, 2)

    def _plot_simulation_output(
        self, filename, boundary_thresh, picture_thresh, min_neighbors, distance, return_cleaned
    ):
        """Plot the simulation output."""
        return plot_simtel_ctapipe(
            filename,
            cleaning_args=[boundary_thresh, picture_thresh, min_neighbors],
            distance=distance,
            return_cleaned=return_cleaned,
        )

    def save_figures_to_pdf(self, figures, telescope):
        """Save the generated figures to a PDF file."""
        save_figs_to_pdf(
            figures,
            f"{self.output_directory}/"
            f"{telescope}_{self.le_application[0]}_{self.le_application[1]}.pdf",
        )

    def run_simulation(self, args_dict, figures):
        """Run the light emission simulation."""
        run_script = self.prepare_script(generate_postscript=True, **args_dict)
        log_file = Path(self.output_directory) / "logfile.log"
        with open(log_file, "w", encoding="utf-8") as log_file:
            subprocess.run(
                run_script,
                shell=False,
                check=False,
                text=True,
                stdout=log_file,
                stderr=log_file,
            )
        self.process_simulation_output(args_dict, figures)

    def distance_list(self, arg):
        """
        Convert distance list to astropy quantities.

        Parameters
        ----------
        arg: list
            List of distances.

        Returns
        -------
        values: list
            List of distances as astropy quantities.
        """
        try:
            return [float(x) * u.m for x in arg]
        except ValueError as exc:
            raise ValueError("Distances must be numeric values") from exc

    def update_light_emission_config(self, key: str, value):
        """
        Update the light emission configuration.

        Parameters
        ----------
        key : str
            The key in the configuration to update.
        value : Any
            The new value to set for the key.
        """
        if key in self.light_emission_config:
            self.light_emission_config[key]["default"] = value
        else:
            raise KeyError(f"Key '{key}' not found in light emission configuration.")

    def calculate_distance_telescope_calibration_device(self):
        """
        Calculate the distance(s) between the telescope and the calibration device.

        Returns
        -------
        list of astropy.Quantity
            A list of distances for variable positions or a single distance for layout positions.
        """
        if not self.light_emission_config:
            # Layout positions: Use DB coordinates
            x_cal, y_cal, z_cal = self._calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
            x_cal, y_cal, z_cal = [coord.to(u.m).value for coord in (x_cal, y_cal, z_cal)]
            x_tel, y_tel, z_tel = self._telescope_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
            x_tel, y_tel, z_tel = [coord.to(u.m).value for coord in (x_tel, y_tel, z_tel)]
            tel_vect = np.array([x_tel, y_tel, z_tel])
            cal_vect = np.array([x_cal, y_cal, z_cal])
            distance = np.linalg.norm(cal_vect - tel_vect)
            print("Distance between telescope and calibration device:", distance * u.m)
            return [distance * u.m]

        # Variable positions: Calculate distances for all positions
        x_tel = self.light_emission_config["x_pos"]["default"].to(u.m).value
        y_tel = self.light_emission_config["y_pos"]["default"].to(u.m).value
        z_positions = self.light_emission_config["z_pos"]["default"]

        distances = []
        for z in z_positions:
            tel_vect = np.array([x_tel, y_tel, z.to(u.m).value])
            cal_vect = np.array([0, 0, 0])  # Calibration device at origin
            distances.append(np.linalg.norm(cal_vect - tel_vect) * u.m)
        return distances

    def simulate_variable_distances(self, args_dict):
        """Simulate light emission for variable distances."""
        if args_dict["distances_ls"] is not None:
            self.update_light_emission_config(
                "z_pos", self.distance_list(args_dict["distances_ls"])
            )
        self._logger.info(
            f"Simulating for distances: {self.light_emission_config['z_pos']['default']}"
        )

        figures = []
        distances = self.calculate_distance_telescope_calibration_device()

        for current_distance, z_pos in zip(
            distances, self.light_emission_config["z_pos"]["default"]
        ):
            self.update_light_emission_config("z_pos", z_pos)
            self.distance = current_distance
            self.run_simulation(args_dict, figures)

        self.save_figures_to_pdf(figures, args_dict["telescope"])

    def simulate_layout_positions(self, args_dict):
        """Simulate light emission for layout positions."""
        figures = []
        self.distance = self.calculate_distance_telescope_calibration_device()[
            0
        ]  # Single distance for layout
        self.run_simulation(args_dict, figures)
        self.save_figures_to_pdf(figures, args_dict["telescope"])
