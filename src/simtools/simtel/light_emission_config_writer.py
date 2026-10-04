"""Translate calibration-source settings into LightEmission configuration."""

import logging
import shutil

import astropy.units as u

from simtools import settings
from simtools.simtel import simtel_file_writer
from simtools.utils.geometry import fiducial_radius_from_shape


class LightEmissionConfigWriter:
    """Write settings and tables for ff-1m and xyzls.

    Parameters
    ----------
    simulation : SimulatorLightEmission
        Physical source configuration, model parameters, and geometry helpers.
    """

    def __init__(self, simulation):
        self.simulation = simulation
        self._logger = logging.getLogger(__name__)

    def get_light_emission_application_name(self):
        """
        Return the LightEmission application and mode from type.

        Returns
        -------
        str
            app_name
        """
        if self.simulation.light_emission_config["light_source_type"] == "flat_fielding":
            return "ff-1m"
        # default to illuminator xyzls, mode from setup
        return "xyzls"

    def prepare_flasher_atmosphere_files(self, config_directory, model_id=1):
        """
        Prepare canonical atmosphere aliases for ff-1m and return model id.

        The ff-1m tool requires atmosphere files atmprof1.dat or atm_profile_model_1.dat and
        as configuration parameter the atmosphere id ('--atmosphere id').

        """
        src_path = config_directory / self.simulation.site_model.get_parameter_value(
            "atmospheric_profile"
        )
        self._logger.debug(f"Using atmosphere profile: {src_path}")

        for name in (f"atmprof{model_id}.dat", f"atm_profile_model_{model_id}.dat"):
            dst = config_directory / name
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                try:
                    dst.symlink_to(src_path)
                except OSError:
                    shutil.copy2(src_path, dst)
            except OSError as copy_err:
                self._logger.warning(f"Failed to create atmosphere alias {dst.name}: {copy_err}")
        return model_id

    def make_command(self, iact_output):
        """
        Create the light emission command to run the light emission package.

        Require the specified pre-compiled light emission package application
        in the sim_telarray/LightEmission/ path.

        Parameters
        ----------
        iact_output: str or Path
            The output iact file path.

        Returns
        -------
        str
            The commands to run the Light Emission package
        """
        config_directory = self.simulation.telescope_model.config_file_directory
        obs_level = self.simulation.site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        )

        app = self.get_light_emission_application_name()
        cmd = [
            str(settings.config.sim_telarray_path / "LightEmission" / app),
            *self.get_site_command(app, config_directory, obs_level),
            *self.get_light_source_command(),
        ]

        if self.simulation.light_emission_config["light_source_type"] == "illuminator":
            cmd += [
                "-A",
                (
                    f"{config_directory}/"
                    f"{self.simulation.site_model.get_parameter_value('atmospheric_profile')}"
                ),
            ]

        cmd += ["-o", str(iact_output)]
        log_file = self.simulation.runner_service.get_file_name(file_type="light_emission_log")
        return " ".join(cmd) + f" 2>&1 | gzip > {log_file}\n"

    def get_site_command(self, app_name, config_directory, corsika_observation_level):
        """Return site command with altitude, atmosphere and telescope_position handling."""
        if app_name in ("ff-1m",):
            atmo_id = self.prepare_flasher_atmosphere_files(config_directory)
            return [
                "-I.",
                f"-I{settings.config.sim_telarray_path / 'cfg'}",
                f"-I{config_directory}",
                f"--altitude {corsika_observation_level.to(u.m).value}",
                f"--atmosphere {atmo_id}",
            ]
        # default path (not used for flasher now, but kept for completeness)
        cmd = [f"-h  {corsika_observation_level.to(u.m).value} "]

        if self.simulation.light_emission_config.get("light_source_type") == "illuminator":
            illuminator_position = self.simulation.get_illuminator_position()
            pointing_vector = self.simulation.get_illuminator_pointing_vector(illuminator_position)
            if self.simulation.uses_telescope_position_file(pointing_vector):
                self._logger.info(
                    "Using telescope position file for illuminator setup "
                    f"(pointing={pointing_vector})."
                )
                position_file = self.simulation.write_telescope_position_file(illuminator_position)
                cmd.append(f"--telpos-file {position_file}")
        return cmd

    def get_light_source_command(self):
        """Return light-source specific command options."""
        if self.simulation.light_emission_config["light_source_type"] == "flat_fielding":
            return self.add_flasher_command_options()
        if self.simulation.light_emission_config["light_source_type"] == "illuminator":
            return self.add_illuminator_command_options()
        source_type = self.simulation.light_emission_config["light_source_type"]
        raise ValueError(f"Unknown light_source_type '{source_type}'")

    def add_flasher_command_options(self):
        """Add flasher options for all telescope types (ff-1m style)."""
        events, photons = self.simulation.build_flasher_event_and_photon_sequences()
        flasher_xyz = self.simulation.calibration_model.get_parameter_value_with_unit(
            "flasher_position"
        )
        camera_diam_cm = (
            self.simulation.telescope_model.get_parameter_value_with_unit("camera_body_diameter")
            .to(u.cm)
            .value
        )
        camera_shape = self.simulation.telescope_model.get_parameter_value("camera_body_shape")
        camera_radius = fiducial_radius_from_shape(camera_diam_cm, camera_shape)
        flasher_wavelength = self.simulation.calibration_model.get_parameter_value_with_unit(
            "flasher_wavelength"
        )
        distance = self.simulation.calculate_distance_focal_plane_calibration_device()
        dist_cm = distance.to(u.cm).value
        angular_distribution = self.get_angular_distribution_string_for_sim_telarray()

        pulse_arg = self.get_pulse_shape_argument_for_sim_telarray()

        bunch_size = self.simulation.calibration_model.get_parameter_value("flasher_bunch_size")
        return [
            f"--events {','.join(str(event) for event in events)}",
            f"--photons {','.join(str(photon) for photon in photons)}",
            f"--bunchsize {bunch_size}",
            f"--xy {flasher_xyz[0].to(u.cm).value},{flasher_xyz[1].to(u.cm).value}",
            f"--distance {dist_cm}",
            f"--camera-radius {camera_radius}",
            f"--spectrum {int(flasher_wavelength.to(u.nm).value)}",
            f"--lightpulse {pulse_arg}",
            f"--angular-distribution {angular_distribution}",
        ]

    def add_illuminator_command_options(self):
        """Get illuminator-specific command options for light emission script."""
        pos = self.simulation.get_illuminator_position()
        x_cal, y_cal, z_cal = pos
        pointing_vector = self.simulation.get_illuminator_pointing_vector(pos)
        flasher_wavelength = self.simulation.calibration_model.get_parameter_value_with_unit(
            "flasher_wavelength"
        )
        angular_distribution = self.get_angular_distribution_string_for_sim_telarray()

        pulse_arg = self.get_pulse_shape_argument_for_sim_telarray()

        return [
            f"-x {x_cal.to(u.cm).value}",
            f"-y {y_cal.to(u.cm).value}",
            f"-z {z_cal.to(u.cm).value}",
            f"-d {','.join(map(str, pointing_vector))}",
            f"-n {self.simulation.light_emission_config['flasher_photons']}",
            f"-s {int(flasher_wavelength.to(u.nm).value)}",
            f"-p {pulse_arg}",
            f"-a {angular_distribution}",
        ]

    def generate_lambertian_angular_distribution_table(self):
        """Generate Lambertian angular distribution table and return path.

        Uses a pure cosine profile normalized to 1 at 0 deg and spans 0..max_angle_deg.
        """
        tel = self.simulation.get_file_name_token(
            self.simulation.light_emission_config.get("telescope") or "telescope"
        )
        cal = self.simulation.get_file_name_token(
            self.simulation.light_emission_config.get("light_source") or "calibration"
        )
        fname = f"flasher_angular_distribution_{tel}_{cal}.dat"
        max_angle_deg = (
            self.simulation.calibration_model.get_parameter_value_with_unit(
                "flasher_angular_distribution_width"
            )
            .to(u.deg)
            .value
        )
        path = simtel_file_writer.write_angular_distribution_table_lambertian(
            file_path=self.simulation.io_handler.get_output_directory("light_emission") / fname,
            max_angle_deg=max_angle_deg,
            n_samples=100,
        )
        return str(path)

    def get_angular_distribution_string_for_sim_telarray(self):
        """
        Get the angular distribution string for sim_telarray.

        Returns
        -------
        str
            The angular distribution string.
        """
        opt = self.simulation.calibration_model.get_parameter_value("flasher_angular_distribution")
        option_string = str(opt).lower() if opt is not None else ""
        if option_string == "lambertian":
            try:
                return self.generate_lambertian_angular_distribution_table()
            except (OSError, ValueError) as err:
                self._logger.warning(
                    f"Failed to write Lambertian angular distribution table: {err};"
                    f" using token instead."
                )
                return option_string

        if option_string == "isotropic":
            return option_string

        width = self.simulation.calibration_model.get_parameter_value_with_unit(
            "flasher_angular_distribution_width"
        )
        return f"{option_string}:{width.to(u.deg).value}" if width is not None else option_string

    def get_pulse_shape_argument_for_sim_telarray(self):
        """
        Get the pulse shape argument for sim_telarray.

        For Gauss-Exponential shapes, writes a DAT file and returns the file path.
        For other shapes, returns a string token representation.

        Returns
        -------
        str
            The pulse shape argument (either a file path or a token string).
        """
        pulse_shape_value = self.simulation.calibration_model.get_parameter_value(
            "flasher_pulse_shape"
        )
        shape_name = pulse_shape_value[0]
        width_ns = pulse_shape_value[1]
        exp_ns = pulse_shape_value[2]

        # Handle Gauss-Exponential by writing a DAT file
        if shape_name == "Gauss-Exponential":
            if width_ns <= 0 or exp_ns <= 0:
                raise ValueError(
                    "Gauss-Exponential pulse shape requires positive width"
                    " and exponential decay values"
                )
            try:
                tel = self.simulation.light_emission_config.get("telescope") or "telescope"
                cal = self.simulation.light_emission_config.get("light_source") or "calibration"
                telescope_token = self.simulation.get_file_name_token(tel)
                source_token = self.simulation.get_file_name_token(cal)
                fname = f"flasher_pulse_shape_{telescope_token}_{source_token}.dat"
                table_path = (
                    self.simulation.io_handler.get_output_directory("light_emission") / fname
                )
                fadc_bins = self.simulation.telescope_model.get_parameter_value("fadc_sum_bins")

                simtel_file_writer.write_light_pulse_table_gauss_exp_conv(
                    file_path=table_path,
                    width_ns=width_ns,
                    exp_decay_ns=exp_ns,
                    fadc_sum_bins=fadc_bins,
                    time_margin_ns=5.0,
                )
                return str(table_path)
            except (ValueError, OSError) as err:
                self._logger.warning(f"Failed to write pulse shape table, using token: {err}")
                return self.get_pulse_shape_string_token(shape_name, width_ns, exp_ns)

        # For other shapes, return token string
        return self.get_pulse_shape_string_token(shape_name, width_ns, exp_ns)

    def get_pulse_shape_string_token(self, shape_name, width_ns, exp_ns):
        """
        Get the pulse shape string token for sim_telarray.

        Parameters
        ----------
        shape_name : str
            Name of the pulse shape.
        width_ns : float
            Width parameter in nanoseconds.
        exp_ns : float
            Exponential decay parameter in nanoseconds.

        Returns
        -------
        str
            The pulse shape token string.
        """
        shape = shape_name.lower()
        # Map internal shapes to sim_telarray expected tokens
        shape_token_map = {
            "tophat": "simple",
        }
        shape_out = shape_token_map.get(shape, shape)

        if shape_out in ("gauss", "simple") and width_ns is not None:
            return f"{shape_out}:{float(width_ns)}"
        if shape_out == "exponential" and exp_ns is not None:
            return f"{shape_out}:{float(exp_ns)}"
        return shape_out
