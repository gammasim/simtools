"""Light emission simulation (e.g. illuminators or flashers)."""

import logging
import shutil
import stat
import subprocess
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.io import io_handler
from simtools.model.model_utils import initialize_simulation_models
from simtools.runners.simtel_runner import SimtelRunner
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils.general import clear_default_sim_telarray_cfg_directories
from simtools.utils.geometry import fiducial_radius_from_shape


class SimulatorLightEmission(SimtelRunner):
    """
    Light emission simulation (e.g. illuminators or flashers).

    Uses the sim_telarray LightEmission package to simulate the light emission.

    Parameters
    ----------
    light_emission_config : dict, optional
        Configuration for the light emission (e.g. number of events, model names)
    label : str, optional
        Label for the simulation
    """

    def __init__(self, light_emission_config, db_config=None, label=None):
        """Initialize SimulatorLightEmission."""
        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()

        super().__init__(
            simtel_path=light_emission_config.get("simtel_path"), label=label, corsika_config=None
        )

        self.output_directory = self.io_handler.get_output_directory()

        self.telescope_model, self.site_model, self.calibration_model = (
            initialize_simulation_models(
                label=label,
                db_config=db_config,
                site=light_emission_config.get("site"),
                telescope_name=light_emission_config.get("telescope"),
                calibration_device_name=light_emission_config.get("light_source"),
                model_version=light_emission_config.get("model_version"),
            )
        )
        self.telescope_model.write_sim_telarray_config_file(additional_models=self.site_model)

        self.light_emission_config = self._initialize_light_emission_configuration(
            light_emission_config
        )

    def _initialize_light_emission_configuration(self, config):
        """Initialize light emission configuration."""
        if self.calibration_model.get_parameter_value("flasher_type"):
            config["light_source_type"] = self.calibration_model.get_parameter_value(
                "flasher_type"
            ).lower()

        config["flasher_photons"] = (
            self.calibration_model.get_parameter_value("flasher_photons")
            if not config.get("test", False)
            else 1e8
        )

        if config.get("light_source_position") is not None:
            config["light_source_position"] = (
                np.array(config["light_source_position"], dtype=float) * u.m
            )

        return config

    def simulate(self):
        """
        Simulate light emission.

        Returns
        -------
        Path
            The output simtel file path.
        """
        run_script = self.prepare_script()
        log_path = Path(self.output_directory) / "logfile.log"
        with open(log_path, "w", encoding="utf-8") as fh:
            subprocess.run(
                run_script,
                shell=False,
                check=False,
                text=True,
                stdout=fh,
                stderr=fh,
            )
        out = Path(self._get_simulation_output_filename())
        if not out.exists():
            self._logger.warning(f"Expected sim_telarray output not found: {out}")
        return out

    def prepare_script(self):
        """
        Build and return bash run script containing the light-emission command.

        Returns
        -------
        Path
            Full path of the run script.
        """
        script_dir = self.output_directory.joinpath("scripts")
        script_dir.mkdir(parents=True, exist_ok=True)

        app_name = self._get_light_emission_application_name()
        script_file = script_dir / f"{app_name}-light_emission.sh"
        self._logger.debug(f"Run bash script - {script_file}")

        target_out = Path(self._get_simulation_output_filename())
        if target_out.exists():
            raise FileExistsError(
                f"sim_telarray output file exists, cancelling simulation: {target_out}"
            )

        lines = [
            "#!/usr/bin/env bash\n",
            f"{self._make_light_emission_script()}\n\n",
            (
                f"[ -s '{self.output_directory}/{app_name}.iact.gz' ] || "
                f"{{ echo 'LightEmission did not produce IACT file' >&2; exit 1; }}\n\n"
            ),
            f"{self._make_simtel_script()}\n\n",
            f"rm -f '{self.output_directory}/{app_name}.iact.gz'\n\n",
        ]

        script_file.write_text("".join(lines), encoding="utf-8")
        script_file.chmod(script_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        return script_file

    def _get_prefix(self):
        prefix = self.light_emission_config.get("output_prefix", "")
        if prefix is not None:
            return f"{prefix}_"
        return ""

    def _get_light_emission_application_name(self):
        """
        Return the LightEmission application and mode from type.

        Returns
        -------
        str
            app_name
        """
        if self.light_emission_config["light_source_type"] == "flat_fielding":
            return "ff-1m"
        # default to illuminator xyzls, mode from setup
        return "xyzls"

    def _get_telescope_pointing(self):
        """
        Return telescope pointing based on light source type.

        For flat_fielding sims, avoid calibration pointing entirely; default angles to (0,0).

        Returns
        -------
        tuple
            The telescope pointing angles (theta, phi).

        """
        if self.light_emission_config["light_source_type"] == "flat_fielding":
            return 0.0, 0.0
        if self.light_emission_config.get("light_source_position") is not None:
            self._logger.info("Using fixed (vertical up) telescope pointing.")
            return 0.0, 0.0
        _, angles = self._calibration_pointing_direction()
        return angles[0], angles[1]

    def _calibration_pointing_direction(self, x_cal=None, y_cal=None, z_cal=None):
        """
        Calculate the pointing of the calibration device towards the telescope.

        This is for calibration devices not installed on telescopes (e.g. illuminators).

        Returns
        -------
        list
            The pointing vector from the calibration device to the telescope.
        """
        if x_cal is None or y_cal is None or z_cal is None:
            x_cal, y_cal, z_cal = self.calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
        x_cal, y_cal, z_cal = [coord.to(u.m).value for coord in (x_cal, y_cal, z_cal)]
        cal_vect = np.array([x_cal, y_cal, z_cal])
        x_tel, y_tel, z_tel = self.telescope_model.get_parameter_value_with_unit(
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
        # Calculate source beam theta and phi angles
        direction_vector_inv = direction_vector * -1
        source_theta = np.round(
            np.rad2deg(np.arccos(direction_vector_inv[2] / np.linalg.norm(direction_vector_inv))),
            6,
        )
        source_phi = np.round(
            np.rad2deg(np.arctan2(direction_vector_inv[1], direction_vector_inv[0])), 6
        )
        return pointing_vector.tolist(), [tel_theta, tel_phi, source_theta, source_phi]

    def _write_telescope_position_file(self):
        """
        Write the telescope positions to a telescope_position file.

        The file will contain lines in the format: x y z r in cm

        Returns
        -------
        Path
            The path to the generated telescope_position file.
        """
        x_tel, y_tel, z_tel = self.telescope_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )
        x_tel, y_tel, z_tel = [coord.to(u.cm).value for coord in (x_tel, y_tel, z_tel)]

        radius = self.telescope_model.get_parameter_value_with_unit("telescope_sphere_radius")
        radius = radius.to(u.cm).value  # Convert radius to cm

        telescope_position_file = self.output_directory.joinpath("telescope_position.dat")
        telescope_position_file.write_text(f"{x_tel} {y_tel} {z_tel} {radius}\n", encoding="utf-8")
        return telescope_position_file

    def _prepare_flasher_atmosphere_files(self, config_directory, model_id=1):
        """
        Prepare canonical atmosphere aliases for ff-1m and return model id.

        The ff-1m tool requires atmosphere files atmprof1.dat or atm_profile_model_1.dat and
        as configuration parameter the atmosphere id ('--atmosphere id').

        """
        src_path = config_directory / self.site_model.get_parameter_value("atmospheric_profile")
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
        config_directory = self.io_handler.get_model_configuration_directory(
            model_version=self.site_model.model_version
        )
        app_name = self._get_light_emission_application_name()
        corsika_observation_level = self.site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        )

        parts = [str(self._simtel_path / "sim_telarray/LightEmission") + f"/{app_name}"]
        parts.extend(self._get_site_command(app_name, config_directory, corsika_observation_level))
        parts.extend(self._get_light_source_command())
        if self.light_emission_config["light_source_type"] == "illuminator":
            parts += [
                "-A",
                (
                    f"{config_directory}/"
                    f"{self.telescope_model.get_parameter_value('atmospheric_profile')}"
                ),
            ]
        parts += [f"-o {self.output_directory}/{app_name}.iact.gz", "\n"]
        return " ".join(parts)

    def _get_site_command(self, app_name, config_directory, corsika_observation_level):
        """Return site command with altitude, atmosphere and telescope_position handling."""
        if app_name in ("ff-1m",):
            atmo_id = self._prepare_flasher_atmosphere_files(config_directory)
            return [
                "-I.",
                f"-I{self._simtel_path / 'sim_telarray/cfg'}",
                f"-I{config_directory}",
                f"--altitude {corsika_observation_level.to(u.m).value}",
                f"--atmosphere {atmo_id}",
            ]
        # default path (not used for flasher now, but kept for completeness)
        return [
            f"-h  {corsika_observation_level.to(u.m).value} ",
            f"--telpos-file {self._write_telescope_position_file()}",
        ]

    def _get_light_source_command(self):
        """Return light-source specific command options."""
        if self.light_emission_config["light_source_type"] == "flat_fielding":
            return self._add_flasher_command_options()
        if self.light_emission_config["light_source_type"] == "illuminator":
            return self._add_illuminator_command_options()
        raise ValueError(
            f"Unknown light_source_type '{self.light_emission_config['light_source_type']}'"
        )

    def _add_flasher_command_options(self):
        """Add flasher options for all telescope types (ff-1m style)."""
        flasher_xyz = self.calibration_model.get_parameter_value_with_unit("flasher_position")
        camera_diam_cm = (
            self.telescope_model.get_parameter_value_with_unit("camera_body_diameter")
            .to(u.cm)
            .value
        )
        camera_shape = self.telescope_model.get_parameter_value("camera_body_shape")
        camera_radius = fiducial_radius_from_shape(camera_diam_cm, camera_shape)
        flasher_wavelength = self.calibration_model.get_parameter_value_with_unit(
            "flasher_wavelength"
        )
        dist_cm = self.calculate_distance_focal_plane_calibration_device().to(u.cm).value
        angular_distribution = self._get_angular_distribution_string_for_sim_telarray()

        # Build pulse table for ff-1m using unified list parameter [shape, width, exp]
        pulse_shape_value = self.calibration_model.get_parameter_value("flasher_pulse_shape")
        shape_name = pulse_shape_value[0]
        width_ns = pulse_shape_value[1]
        exp_ns = pulse_shape_value[2]
        pulse_arg = self._get_pulse_shape_string_for_sim_telarray()

        if shape_name == "Gauss-Exponential" and width_ns > 0 and exp_ns > 0:
            try:
                base_dir = self.io_handler.get_output_directory("pulse_shapes")

                def _sanitize_name(value):
                    return "".join(
                        ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(value)
                    )

                tel = self.light_emission_config.get("telescope") or "telescope"
                cal = self.light_emission_config.get("light_source") or "calibration"
                fname = f"flasher_pulse_shape_{_sanitize_name(tel)}_{_sanitize_name(cal)}.dat"
                table_path = base_dir / fname
                fadc_bins = self.telescope_model.get_parameter_value("fadc_sum_bins")

                SimtelConfigWriter.write_light_pulse_table_gauss_exp_conv(
                    file_path=table_path,
                    width_ns=width_ns,
                    exp_decay_ns=exp_ns,
                    fadc_sum_bins=fadc_bins,
                    time_margin_ns=5.0,
                )
                pulse_arg = str(table_path)
            except (ValueError, OSError) as err:
                self._logger.warning(f"Failed to write pulse shape table, using token: {err}")

        return [
            f"--events {self.light_emission_config['number_of_events']}",
            f"--photons {self.light_emission_config['flasher_photons']}",
            f"--bunchsize {self.calibration_model.get_parameter_value('flasher_bunch_size')}",
            f"--xy {flasher_xyz[0].to(u.cm).value},{flasher_xyz[1].to(u.cm).value}",
            f"--distance {dist_cm}",
            f"--camera-radius {camera_radius}",
            f"--spectrum {int(flasher_wavelength.to(u.nm).value)}",
            f"--lightpulse {pulse_arg}",
            f"--angular-distribution {angular_distribution}",
        ]

    def _add_illuminator_command_options(self):
        """Get illuminator-specific command options for light emission script."""
        pos = self.light_emission_config.get("light_source_position")
        if pos is None:
            pos = self.calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
        x_cal, y_cal, z_cal = pos
        if self.light_emission_config.get("light_source_pointing"):
            pointing_vector = self.light_emission_config["light_source_pointing"]
        else:
            pointing_vector = self._calibration_pointing_direction(x_cal, y_cal, z_cal)[0]
        flasher_wavelength = self.calibration_model.get_parameter_value_with_unit(
            "flasher_wavelength"
        )
        angular_distribution = self._get_angular_distribution_string_for_sim_telarray()

        return [
            f"-x {x_cal.to(u.cm).value}",
            f"-y {y_cal.to(u.cm).value}",
            f"-z {z_cal.to(u.cm).value}",
            f"-d {','.join(map(str, pointing_vector))}",
            f"-n {self.light_emission_config['flasher_photons']}",
            f"-s {int(flasher_wavelength.to(u.nm).value)}",
            f"-p {self._get_pulse_shape_string_for_sim_telarray()}",
            f"-a {angular_distribution}",
        ]

    def _make_simtel_script(self):
        """
        Return the command to run sim_telarray using the output from the previous step.

        Returns
        -------
        str
            The command to run sim_telarray
        """
        theta, phi = self._get_telescope_pointing()

        simtel_bin = self._simtel_path.joinpath("sim_telarray/bin/sim_telarray/")

        parts = [
            f"{simtel_bin}",
            f"-I{self.telescope_model.config_file_directory}",
            f"-I{simtel_bin}",
            f"-c {self.telescope_model.config_file_path}",
            "-DNUM_TELESCOPES=1",
            super().get_config_option(
                "altitude",
                self.site_model.get_parameter_value_with_unit("corsika_observation_level")
                .to(u.m)
                .value,
            ),
            super().get_config_option(
                "atmospheric_transmission",
                self.site_model.get_parameter_value("atmospheric_transmission"),
            ),
            super().get_config_option("TRIGGER_TELESCOPES", "1"),
            super().get_config_option("TELTRIG_MIN_SIGSUM", "2"),
            super().get_config_option("PULSE_ANALYSIS", "-30"),
            super().get_config_option("MAXIMUM_TELESCOPES", 1),
            super().get_config_option("telescope_theta", f"{theta}"),
            super().get_config_option("telescope_phi", f"{phi}"),
        ]

        if self.light_emission_config["light_source_type"] == "flat_fielding":
            parts.append(super().get_config_option("Bypass_Optics", "1"))

        app_name = self._get_light_emission_application_name()
        pref = self._get_prefix()
        parts += [
            super().get_config_option("power_law", "2.68"),
            super().get_config_option("input_file", f"{self.output_directory}/{app_name}.iact.gz"),
            super().get_config_option(
                "output_file", f"{self.output_directory}/{pref}{app_name}.simtel.zst"
            ),
            super().get_config_option(
                "histogram_file", f"{self.output_directory}/{pref}{app_name}.ctsim.hdata\n"
            ),
        ]

        return clear_default_sim_telarray_cfg_directories(" ".join(parts))

    def _get_simulation_output_filename(self):
        """Get the filename of the simulation output."""
        app_name = self._get_light_emission_application_name()
        pref = self._get_prefix()
        return f"{self.output_directory}/{pref}{app_name}.simtel.zst"

    def calculate_distance_focal_plane_calibration_device(self):
        """
        Calculate distance between focal plane and calibration device.

        For flasher-type light sources. Flasher position is given in mirror coordinates,
        with positive z pointing towards the camera, so the distance is focal_length - flasher_z.

        Returns
        -------
        astropy.units.Quantity
            Distance between calibration device and focal plane.
        """
        focal_length = self.telescope_model.get_parameter_value_with_unit("focal_length").to(u.m)
        flasher_z = self.calibration_model.get_parameter_value_with_unit("flasher_position")[2].to(
            u.m
        )
        return focal_length - flasher_z

    def _generate_lambertian_angular_distribution_table(self, width):
        """Generate Lambertian angular distribution table via config writer and return path."""
        base_dir = self.io_handler.get_output_directory("angular_distributions")

        def _sanitize_name(value):
            return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(value))

        tel = self.light_emission_config.get("telescope") or "telescope"
        cal = self.light_emission_config.get("light_source") or "calibration"
        fname = f"flasher_angular_distribution_{_sanitize_name(tel)}_{_sanitize_name(cal)}.dat"
        table_path = base_dir / fname
        max_angle_deg = float(width.to(u.deg).value)
        SimtelConfigWriter.write_angular_distribution_table_lambertian(
            file_path=table_path, max_angle_deg=max_angle_deg, n_samples=100
        )
        return str(table_path)

    def _get_angular_distribution_string_for_sim_telarray(self):
        """
        Get the angular distribution string for sim_telarray.

        Returns
        -------
        str
            The angular distribution string.
        """
        opt = self.calibration_model.get_parameter_value("flasher_angular_distribution")
        option_string = str(opt).lower() if opt is not None else ""
        width = self.calibration_model.get_parameter_value_with_unit(
            "flasher_angular_distribution_width"
        )

        # if option_string and width is not None:
        if option_string == "Lambertian" and width is not None:
            try:
                return self._generate_lambertian_angular_distribution_table(width)
            except (OSError, ValueError) as err:
                self._logger.warning(
                    f"Failed to write Lambertian angular distribution table: {err}; "
                    f"using token instead."
                )

        return f"{option_string}:{width.to(u.deg).value}" if width is not None else option_string

    def _get_pulse_shape_string_for_sim_telarray(self):
        """
        Get the pulse shape string for sim_telarray.

        Returns
        -------
        str
            The pulse shape string.
        """
        opt = self.calibration_model.get_parameter_value("flasher_pulse_shape")
        shape = opt[0].lower()
        width = opt[1]
        expv = opt[2]
        if shape == "gauss-exponential" and width is not None and expv is not None:
            return f"{shape}:{float(width)}:{float(expv)}"
        if shape in ("gauss", "tophat") and width is not None:
            return f"{shape}:{float(width)}"
        if shape == "exponential" and expv is not None:
            return f"{shape}:{float(expv)}"
        return shape
