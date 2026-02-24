"""Light emission simulation (e.g. illuminators or flashers)."""

import logging
import shutil
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools import settings
from simtools.io import io_handler
from simtools.job_execution import job_manager
from simtools.model.model_utils import initialize_simulation_models
from simtools.runners import runner_services
from simtools.runners.simtel_runner import SimtelRunner, sim_telarray_env_as_string
from simtools.simtel import simtel_output_validator
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import general
from simtools.utils.geometry import fiducial_radius_from_shape


class SimulatorLightEmission(SimtelRunner):
    """
    Light emission simulation (e.g. illuminators or flashers).

    Uses the sim_telarray LightEmission package to simulate the light emission.

    Parameters
    ----------
    light_emission_config : dict, optional
        Configuration for the light emission (e.g. number of events, model names)
    telescope : str, optional
        Telescope name.
    label : str, optional
        Label for the simulation
    """

    def __init__(self, light_emission_config, telescope=None, label=None):
        """Initialize SimulatorLightEmission."""
        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()
        telescope = telescope or light_emission_config.get("telescope")
        label = f"{label}_{telescope}" if label else telescope

        super().__init__(label=label, config=light_emission_config)
        self.submission_files = runner_services.RunnerServices(
            light_emission_config, run_type="sub", label=label
        )

        self.telescope_model, self.site_model, self.calibration_model = (
            initialize_simulation_models(
                label=label,
                site=light_emission_config.get("site"),
                telescope_name=telescope,
                calibration_device_name=light_emission_config.get("light_source"),
                calibration_device_type=light_emission_config.get("light_source_type"),
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

        if config.get("flasher_photons") is not None:
            photons = general.parse_typed_sequence(config["flasher_photons"], int)
            if len(photons) == 1:
                photon_value = photons[0]
                self.calibration_model.overwrite_model_parameter("flasher_photons", photon_value)
                config["flasher_photons"] = photon_value
            else:
                config["flasher_photons"] = photons
        else:
            config["flasher_photons"] = self.calibration_model.get_parameter_value(
                "flasher_photons"
            )

        if config.get("light_source_position") is not None:
            config["light_source_position"] = (
                np.array(config["light_source_position"], dtype=float) * u.m
            )

        return config

    @staticmethod
    def _repeat_or_map_events(events, n_photon_levels):
        """Apply ff-1m event mapping rules to event counts."""
        if n_photon_levels == 1:
            return [events[0]]
        if len(events) == n_photon_levels:
            return events
        if len(events) == 1:
            return [events[0]] * n_photon_levels
        raise ValueError(
            "Invalid number_of_events list length. Use one value or one value per photon intensity."
        )

    def _build_flasher_event_and_photon_sequences(self):
        """Build ff-1m-compatible events/photons sequences."""
        photons_int = general.parse_typed_sequence(
            self.light_emission_config.get("flasher_photons"), int
        )

        events = general.parse_typed_sequence(
            self.light_emission_config.get("number_of_events", 1), int
        )
        events_int = self._repeat_or_map_events(events, len(photons_int))

        return events_int, photons_int

    def simulate(self):
        """Simulate light emission."""
        run_script = self.prepare_run()
        job_manager.submit(
            run_script,
            out_file=self.submission_files.get_file_name("sub_out"),
            err_file=self.submission_files.get_file_name("sub_err"),
        )

    def prepare_run(self):
        """
        Prepare the bash run script containing the light-emission command.

        Returns
        -------
        Path
            Full path of the run script.
        """
        script_file = self.submission_files.get_file_name(file_type="sub_script")
        output_file = self.runner_service.get_file_name(file_type="sim_telarray_output")
        if output_file.exists():
            raise FileExistsError(
                f"sim_telarray output file exists, cancelling simulation: {output_file}"
            )
        lines = self.make_run_command()
        script_file.write_text("".join(lines), encoding="utf-8")
        return script_file

    def make_run_command(self, run_number=None, input_file=None):  # pylint: disable=unused-argument
        """Light emission and sim_telarray run command."""
        iact_output = self.runner_service.get_file_name(file_type="iact_output")
        return [
            "#!/usr/bin/env bash\n",
            f"{self._make_light_emission_command(iact_output)}\n\n",
            (
                f"[ -s '{iact_output}' ] || "
                f"{{ echo 'LightEmission did not produce IACT file' >&2; exit 1; }}\n\n"
            ),
            f"{self._make_simtel_script()}\n\n",
            f"rm -f '{iact_output}'\n\n",
        ]

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

        telescope_position_file = (
            self.io_handler.get_output_directory("light_emission") / "telescope_position.dat"
        )
        telescope_position_file.write_text(f"{x_tel} {y_tel} {z_tel} {radius}\n", encoding="utf-8")
        return telescope_position_file

    def _get_illuminator_position(self):
        """Return illuminator position (x, y, z) in ground coordinates."""
        pos = self.light_emission_config.get("light_source_position")
        if pos is None:
            pos = self.calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
        return pos

    def _get_illuminator_pointing_vector(self, pos=None):
        """Return illuminator pointing vector; prefer explicit config if available."""
        pointing_vector = self.light_emission_config.get("light_source_pointing")
        if pointing_vector is not None:
            return pointing_vector
        if pos is None:
            pos = self._get_illuminator_position()
        x_cal, y_cal, z_cal = pos
        return self._calibration_pointing_direction(x_cal, y_cal, z_cal)[0]

    @staticmethod
    def _should_use_telpos_file(pointing_vector):
        """Decide whether to use telpos file based on pointing vector.

        Rule: do not use telpos only if pointing is (0, 0, -1) (within tolerance).
        """
        try:
            vec = np.asarray(pointing_vector, dtype=float)
        except (TypeError, ValueError):
            return True
        if vec.size < 3 or not np.all(np.isfinite(vec[:3])):
            return True
        is_default_down = np.allclose(vec[:3], [0.0, 0.0, -1.0], atol=1e-6)
        return not is_default_down

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

    def _make_light_emission_command(self, iact_output):
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
        config_directory = self.io_handler.get_model_configuration_directory(
            model_version=self.site_model.model_version
        )
        obs_level = self.site_model.get_parameter_value_with_unit("corsika_observation_level")

        app = self._get_light_emission_application_name()
        cmd = [
            str(settings.config.sim_telarray_path / "LightEmission" / app),
            *self._get_site_command(app, config_directory, obs_level),
            *self._get_light_source_command(),
        ]

        if self.light_emission_config["light_source_type"] == "illuminator":
            cmd += [
                "-A",
                f"{config_directory}/"
                f"{self.telescope_model.get_parameter_value('atmospheric_profile')}",
            ]

        cmd += ["-o", str(iact_output)]
        log_file = self.runner_service.get_file_name(file_type="light_emission_log")
        return " ".join(cmd) + f" 2>&1 | gzip > {log_file}\n"

    def _get_site_command(self, app_name, config_directory, corsika_observation_level):
        """Return site command with altitude, atmosphere and telescope_position handling."""
        if app_name in ("ff-1m",):
            atmo_id = self._prepare_flasher_atmosphere_files(config_directory)
            return [
                "-I.",
                f"-I{settings.config.sim_telarray_path / 'cfg'}",
                f"-I{config_directory}",
                f"--altitude {corsika_observation_level.to(u.m).value}",
                f"--atmosphere {atmo_id}",
            ]
        # default path (not used for flasher now, but kept for completeness)
        cmd = [f"-h  {corsika_observation_level.to(u.m).value} "]

        if self.light_emission_config.get("light_source_type") == "illuminator":
            pointing_vector = self._get_illuminator_pointing_vector()
            if self._should_use_telpos_file(pointing_vector):
                self._logger.info(
                    "Using telescope position file for illuminator setup "
                    f"(pointing={pointing_vector})."
                )
                cmd.append(f"--telpos-file {self._write_telescope_position_file()}")
        return cmd

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
        events, photons = self._build_flasher_event_and_photon_sequences()
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

        if shape_name == "Gauss-Exponential":
            if width_ns <= 0 or exp_ns <= 0:
                raise ValueError(
                    "Gauss-Exponential pulse shape requires positive width"
                    " and exponential decay values"
                )
            try:
                tel = self.light_emission_config.get("telescope") or "telescope"
                cal = self.light_emission_config.get("light_source") or "calibration"
                fname = (
                    f"flasher_pulse_shape_{self._sanitize_name(tel)}_{self._sanitize_name(cal)}.dat"
                )
                table_path = self.io_handler.get_output_directory("light_emission") / fname
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
            f"--events {','.join(str(event) for event in events)}",
            f"--photons {','.join(str(photon) for photon in photons)}",
            f"--bunchsize {self.calibration_model.get_parameter_value('flasher_bunch_size')}",
            f"--xy {flasher_xyz[0].to(u.cm).value},{flasher_xyz[1].to(u.cm).value}",
            f"--distance {dist_cm}",
            f"--camera-radius {camera_radius}",
            f"--spectrum {int(flasher_wavelength.to(u.nm).value)}",
            f"--lightpulse {pulse_arg}",
            f"--angular-distribution {angular_distribution}",
        ]

    @staticmethod
    def _sanitize_name(value):
        return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(value))

    def _add_illuminator_command_options(self):
        """Get illuminator-specific command options for light emission script."""
        pos = self._get_illuminator_position()
        x_cal, y_cal, z_cal = pos
        pointing_vector = self._get_illuminator_pointing_vector(pos)
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
        simtel_bin = str(settings.config.sim_telarray_exe)

        parts = [
            simtel_bin,
            f"-I{self.telescope_model.config_file_directory}",
            f"-I{simtel_bin}",
            f"-c {self.telescope_model.config_file_path}",
            "-DNUM_TELESCOPES=1",
        ]

        options = [
            (
                "altitude",
                self.site_model.get_parameter_value_with_unit("corsika_observation_level")
                .to(u.m)
                .value,
            ),
            (
                "atmospheric_transmission",
                self.site_model.get_parameter_value("atmospheric_transmission"),
            ),
            ("TRIGGER_TELESCOPES", "1"),
            ("TELTRIG_MIN_SIGSUM", "2"),
            ("PULSE_ANALYSIS", "-30"),
            ("MAXIMUM_TELESCOPES", 1),
            ("telescope_theta", f"{theta}"),
            ("telescope_phi", f"{phi}"),
        ]

        if self.light_emission_config["light_source_type"] == "flat_fielding":
            options.append(("Bypass_Optics", "1"))

        input_file = self.runner_service.get_file_name(file_type="iact_output")
        output_file = self.runner_service.get_file_name(file_type="sim_telarray_output")
        histo_file = self.runner_service.get_file_name(file_type="sim_telarray_histogram")

        options += [
            ("power_law", "2.68"),
            ("input_file", f"{input_file}"),
            ("output_file", f"{output_file}"),
            ("histogram_file", f"{histo_file}"),
        ]

        parts += [f"-C {key}={value}" for key, value in options]

        log_file = self.runner_service.get_file_name(file_type="sim_telarray_log")

        return sim_telarray_env_as_string() + " ".join(parts) + f" 2>&1 | gzip > {log_file}\n"

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

    def _generate_lambertian_angular_distribution_table(self):
        """Generate Lambertian angular distribution table via config writer and return path.

        Uses a pure cosine profile normalized to 1 at 0 deg and spans 0..90 deg by default.
        """
        tel = self._sanitize_name(self.light_emission_config.get("telescope") or "telescope")
        cal = self._sanitize_name(self.light_emission_config.get("light_source") or "calibration")
        fname = f"flasher_angular_distribution_{tel}_{cal}.dat"
        return SimtelConfigWriter.write_angular_distribution_table_lambertian(
            file_path=self.io_handler.get_output_directory("light_emission") / fname,
            max_angle_deg=90.0,
            n_samples=100,
        )

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
        if option_string == "lambertian":
            try:
                return self._generate_lambertian_angular_distribution_table()
            except (OSError, ValueError) as err:
                self._logger.warning(
                    f"Failed to write Lambertian angular distribution table: {err};"
                    f" using token instead."
                )
                return option_string

        if option_string == "isotropic":
            return option_string

        width = self.calibration_model.get_parameter_value_with_unit(
            "flasher_angular_distribution_width"
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
        # Map internal shapes to sim_telarray expected tokens
        # 'tophat' corresponds to a simple (flat) pulse in sim_telarray.
        shape_token_map = {
            "tophat": "simple",
        }
        shape_out = shape_token_map.get(shape, shape)
        width = opt[1]
        expv = opt[2]
        if shape_out == "gauss-exponential" and width is not None and expv is not None:
            return f"{shape_out}:{float(width)}:{float(expv)}"
        if shape_out in ("gauss", "simple") and width is not None:
            return f"{shape_out}:{float(width)}"
        if shape_out == "exponential" and expv is not None:
            return f"{shape_out}:{float(expv)}"
        return shape_out

    def validate_simulations(self):
        """Validate that the simulations were successful."""
        simtel_output_validator.validate_sim_telarray(
            data_files=Path(self.runner_service.get_file_name(file_type="sim_telarray_output")),
            log_files=Path(self.runner_service.get_file_name(file_type="sim_telarray_log")),
            array_models=None,
        )
