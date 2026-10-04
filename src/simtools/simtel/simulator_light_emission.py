"""Light emission simulation (e.g. illuminators or flashers)."""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools import settings
from simtools.application.model_reader import require_model_reader
from simtools.io import io_handler
from simtools.job_execution import job_manager
from simtools.model.calibration_model import CalibrationModel
from simtools.model.model_utils import (
    initialize_simulation_models,
    read_overwrite_model_parameter_dict,
)
from simtools.model_repository.asset_names import get_simtel_table_file_name
from simtools.runners import runner_services
from simtools.runners.simtel_runner import SimtelRunner, sim_telarray_env_as_string
from simtools.simtel import simtel_output_validator
from simtools.simulation.configuration import get_light_source_writer
from simtools.utils import general


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

    @staticmethod
    def get_available_wavelengths_from_config(config, model_reader=None):
        """
        Get available wavelengths from model configuration without full initialization.

        This is a lightweight method that only initializes the calibration model
        to retrieve wavelengths, without telescope/site models or file I/O.

        Parameters
        ----------
        config : dict
            Configuration dictionary with site, light_source, and model_version

        Returns
        -------
        list of astropy.units.Quantity
            List of available wavelengths with units

        Raises
        ------
        ValueError
            If required configuration keys are missing
        """
        site = config.get("site")
        model_version = config.get("model_version")
        light_source = config.get("light_source")

        if site is None or model_version is None or light_source is None:
            raise ValueError(
                "Missing required configuration keys for wavelength query: "
                f"site={site}, model_version={model_version}, light_source={light_source}"
            )

        calibration_kwargs = {
            "site": site,
            "calibration_device_model_name": light_source,
            "model_version": model_version,
            "label": f"temp_wavelength_query_{light_source}",
            "overwrite_model_parameter_dict": read_overwrite_model_parameter_dict(),
        }
        calibration_kwargs["model_reader"] = require_model_reader(model_reader)
        calibration_model = CalibrationModel(**calibration_kwargs)

        wavelengths = calibration_model.get_parameter_value_with_unit("flasher_wavelength")
        return general.ensure_list(wavelengths)

    def __init__(self, light_emission_config, telescope=None, label=None, model_reader=None):
        """Initialize SimulatorLightEmission."""
        self._logger = logging.getLogger(__name__)
        self.configuration_writers = {}
        self.model_reader = require_model_reader(model_reader)
        self.io_handler = io_handler.IOHandler()
        telescope = telescope or light_emission_config.get("telescope")
        label = f"{label}_{telescope}" if label else telescope

        # Add wavelength to label if present (always include for consistency)
        if (
            "wavelength" in light_emission_config
            and light_emission_config["wavelength"] is not None
        ):
            wl = light_emission_config["wavelength"]
            wl_str = f"{wl.to(u.nm).value:.0f}nm"
            label = f"{label}_{wl_str}"

        super().__init__(label=label, config=light_emission_config)
        self.submission_files = runner_services.RunnerServices(
            light_emission_config, run_type="sub", label=label
        )

        model_kwargs = {
            "label": label,
            "site": light_emission_config.get("site"),
            "telescope_name": telescope,
            "calibration_device_name": light_emission_config.get("light_source"),
            "calibration_device_type": light_emission_config.get("light_source_type"),
            "model_version": light_emission_config.get("model_version"),
        }
        model_kwargs["model_reader"] = self.model_reader
        if light_emission_config.get("model_directory") is not None:
            model_kwargs["model_directory"] = light_emission_config["model_directory"]
        self.telescope_model, self.site_model, self.calibration_model = (
            initialize_simulation_models(**model_kwargs)
        )
        self.telescope_model.write_sim_telarray_config_file(additional_models=self.site_model)

        self.light_emission_config = self._initialize_light_emission_configuration(
            light_emission_config
        )

    def get_available_wavelengths(self):
        """
        Get available wavelengths from this simulator's calibration model.

        Returns
        -------
        list of astropy.units.Quantity
            List of available wavelengths with units
        """
        wavelengths = self.calibration_model.get_parameter_value_with_unit("flasher_wavelength")
        return general.ensure_list(wavelengths)

    def _initialize_light_emission_configuration(self, config):
        """Initialize light emission configuration."""
        flasher_type = self.calibration_model.get_parameter_value("flasher_type")
        if flasher_type:
            config["light_source_type"] = flasher_type.lower()
            if config.get("run_mode") is None:
                config["run_mode"] = config["light_source_type"]

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

        if config.get("wavelength") is not None:
            requested_wavelength = config["wavelength"].to(u.nm)

            # Get allowed wavelengths from the model
            allowed_wavelengths_with_unit = self.calibration_model.get_parameter_value_with_unit(
                "flasher_wavelength"
            )
            # Ensure it's a list (could be scalar Quantity)
            allowed_wavelengths_with_unit = general.ensure_list(allowed_wavelengths_with_unit)
            # Convert to nm for comparison
            allowed_wavelengths = [wl.to(u.nm) for wl in allowed_wavelengths_with_unit]

            # Validate that requested wavelength is one of the allowed wavelengths
            # Use a small tolerance for floating point comparison
            tolerance = 0.5 * u.nm
            if not any(abs(requested_wavelength - wl) < tolerance for wl in allowed_wavelengths):
                allowed_values = [wl.value for wl in allowed_wavelengths]
                raise ValueError(
                    f"Wavelength {requested_wavelength.value} nm is not supported by the "
                    f"illuminator. Allowed wavelengths are: {allowed_values} nm"
                )

            # Find the closest match and use that exact value
            closest_wavelength = min(
                allowed_wavelengths, key=lambda wl: abs(wl - requested_wavelength)
            )

            # Override the model parameter with the selected wavelength
            self.calibration_model.overwrite_model_parameter(
                "flasher_wavelength", closest_wavelength
            )
            config["wavelength"] = closest_wavelength

            self._logger.info(f"Using wavelength: {closest_wavelength.value} nm")

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

    def build_flasher_event_and_photon_sequences(self):
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
            f"{get_light_source_writer(self).make_command(iact_output)}\n\n",
            (
                f"[ -s '{iact_output}' ] || "
                f"{{ echo 'LightEmission did not produce IACT file' >&2; exit 1; }}\n\n"
            ),
            f"{self._make_simtel_script()}\n\n",
            f"rm -f '{iact_output}'\n\n",
        ]

    def _get_telescope_pointing(self):
        """
        Return telescope pointing based on light source type.

        For flat_fielding sims, avoid calibration pointing entirely; default angles to (0,0).
        For fixed light-source positions, keep vertical-up telescope pointing.
        Otherwise (layout/default), derive telescope angles from calibration geometry so
        axis-offset corrections are reflected in source and telescope pointing.

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
        illuminator_position = self.get_illuminator_position()
        _, angles = self._calibration_pointing_direction(*illuminator_position)
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
        x_tel, y_tel, z_tel = self._get_telescope_position_ground_with_axis_offset(
            x_cal=x_cal * u.m,
            y_cal=y_cal * u.m,
            z_cal=z_cal * u.m,
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

    def _get_telescope_position_ground_with_axis_offset(self, x_cal=None, y_cal=None, z_cal=None):
        """Return telescope position with illuminator-axis offset correction in ground coordinates.

        The first axes_offsets component is a horizontal offset from the azimuth axis towards the
        reflector. For illuminator simulations, offsets are applied in the telescope frame using
        the current pointing direction.
        """
        x_tel, y_tel, z_tel = self.telescope_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )

        axes_offsets = self.telescope_model.get_parameter_value_with_unit("axes_offsets")

        offset_1 = axes_offsets[0].to(u.m).value
        offset_2 = axes_offsets[1].to(u.m).value if len(axes_offsets) > 1 else 0.0
        if np.isclose(offset_1, 0.0) and np.isclose(offset_2, 0.0):
            return x_tel, y_tel, z_tel

        if x_cal is None or y_cal is None or z_cal is None:
            x_cal, y_cal, z_cal = self.calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )

        x_tel_m = x_tel.to(u.m).value
        y_tel_m = y_tel.to(u.m).value
        z_tel_m = z_tel.to(u.m).value

        delta_x = x_cal.to(u.m).value - x_tel_m
        delta_y = y_cal.to(u.m).value - y_tel_m
        delta_z = z_cal.to(u.m).value - z_tel_m

        norm_horizontal = np.hypot(delta_x, delta_y)
        # Angle from North (y-axis) towards East (x-axis)
        phi = np.arctan2(delta_x, delta_y)
        el = np.arctan2(delta_z, norm_horizontal)

        # Unit vectors in telescope-pointing convention (phi, el).
        u_pointing = np.array(
            [
                np.cos(el) * np.sin(phi),
                np.cos(el) * np.cos(phi),
                np.sin(el),
            ],
            dtype=float,
        )
        u_xy = np.array([np.sin(phi), np.cos(phi), 0.0], dtype=float)
        altitude_axis = np.array([np.cos(phi), -np.sin(phi), 0.0], dtype=float)

        v_perpendicular = np.cross(altitude_axis, u_pointing)
        norm_v = np.linalg.norm(v_perpendicular)
        v_perpendicular /= norm_v

        corrected_position = np.array([x_tel_m, y_tel_m, z_tel_m], dtype=float)
        corrected_position += offset_1 * u_xy + offset_2 * v_perpendicular

        return (
            corrected_position[0] * u.m,
            corrected_position[1] * u.m,
            corrected_position[2] * u.m,
        )

    def write_telescope_position_file(self, illuminator_position=None):
        """
        Write the telescope positions to a telescope_position file.

        The file will contain lines in the format: x y z r in cm

        Returns
        -------
        Path
            The path to the generated telescope_position file.
        """
        if illuminator_position is None:
            illuminator_position = self.get_illuminator_position()
        x_cal, y_cal, z_cal = illuminator_position
        x_tel, y_tel, z_tel = self._get_telescope_position_ground_with_axis_offset(
            x_cal=x_cal,
            y_cal=y_cal,
            z_cal=z_cal,
        )
        x_tel, y_tel, z_tel = [coord.to(u.cm).value for coord in (x_tel, y_tel, z_tel)]

        radius = self.telescope_model.get_parameter_value_with_unit("telescope_sphere_radius")
        radius = radius.to(u.cm).value  # Convert radius to cm

        tel = self.get_file_name_token(self.light_emission_config.get("telescope") or "telescope")
        cal = self.get_file_name_token(
            self.light_emission_config.get("light_source") or "calibration"
        )
        telescope_position_file = (
            self.io_handler.get_output_directory("light_emission")
            / f"telescope_position_{tel}_{cal}.dat"
        )
        telescope_position_file.write_text(f"{x_tel} {y_tel} {z_tel} {radius}\n", encoding="utf-8")
        return telescope_position_file

    def get_illuminator_position(self):
        """Return the illuminator emission position in ground coordinates."""
        pos = self.light_emission_config.get("light_source_position")
        if pos is not None:
            return pos

        x_pos, y_pos, z_pos = self.calibration_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )
        tower_height = self.calibration_model.get_parameter_value_with_unit(
            "illuminator_tower_height"
        )
        return x_pos, y_pos, z_pos + tower_height

    def get_illuminator_pointing_vector(self, pos=None):
        """Return illuminator pointing vector; prefer explicit config if available."""
        pointing_vector = self.light_emission_config.get("light_source_pointing")
        if pointing_vector is not None:
            return pointing_vector
        if pos is None:
            pos = self.get_illuminator_position()
        x_cal, y_cal, z_cal = pos
        return self._calibration_pointing_direction(x_cal, y_cal, z_cal)[0]

    @staticmethod
    def uses_telescope_position_file(pointing_vector):
        """Decide whether to use telpos file based on pointing vector.

        Rule: do not use telpos only if pointing is (0, 0, -1) (within tolerance).
        """
        try:
            vec = np.asarray(pointing_vector, dtype=float)
        except TypeError, ValueError:
            return True
        if vec.size < 3 or not np.all(np.isfinite(vec[:3])):
            return True
        is_default_down = np.allclose(vec[:3], [0.0, 0.0, -1.0], atol=1e-6)
        return not is_default_down

    @staticmethod
    def get_file_name_token(value):
        """Return a filename token for a light source or telescope name."""
        return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(value))

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
            f"-c {self.telescope_model.config_file_path}",
            "-DNUM_TELESCOPES=1",
        ]

        atmospheric_transmission = self.site_model.get_parameter_value("atmospheric_transmission")
        if str(atmospheric_transmission).lower().endswith(".ecsv"):
            parameter_data = self.site_model.parameters.get("atmospheric_transmission", {})
            atmospheric_transmission = get_simtel_table_file_name(parameter_data) or (
                f"atmospheric_transmission-{Path(self.telescope_model.config_file_path).stem}.dat"
            )

        options = [
            (
                "altitude",
                self.site_model.get_parameter_value_with_unit("corsika_observation_level")
                .to(u.m)
                .value,
            ),
            (
                "atmospheric_transmission",
                atmospheric_transmission,
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

    def validate_simulations(self):
        """Validate that the simulations were successful."""
        simtel_output_validator.validate_sim_telarray(
            data_files=Path(self.runner_service.get_file_name(file_type="sim_telarray_output")),
            log_files=Path(self.runner_service.get_file_name(file_type="sim_telarray_log")),
            array_models=None,
        )
