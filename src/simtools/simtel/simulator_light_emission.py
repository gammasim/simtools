"""Simulation using the light emission package for calibration devices."""

import logging
import shutil
import stat
import subprocess
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.io import io_handler
from simtools.runners.simtel_runner import SimtelRunner
from simtools.utils.general import clear_default_sim_telarray_cfg_directories

__all__ = ["SimulatorLightEmission"]


class SimulatorLightEmission(SimtelRunner):
    """
    Interface with sim_telarray to perform light emission package simulations.

    The light emission package is used to simulate an artificial light source, used for calibration.
    """

    def __init__(
        self,
        *,
        telescope_model,
        calibration_model=None,
        flasher_model=None,
        site_model=None,
        light_emission_config=None,
        light_source_setup=None,
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
        light_source_setup : str, optional
            Setup for light source positioning ("variable" or "layout")
        simtel_path : Path, optional
            Path to the sim_telarray installation
        light_source_type : str, optional
            Type of light source: 'illuminator', or 'flasher'
        label : str, optional
            Label for the simulation
        test : bool, optional
            Whether this is a test run
        """
        super().__init__(simtel_path=simtel_path, label=label, corsika_config=None)

        self._logger = logging.getLogger(__name__)

        self._telescope_model = telescope_model
        self._calibration_model = calibration_model
        self._flasher_model = flasher_model
        self._site_model = site_model
        self.light_emission_config = light_emission_config or {}
        self.light_source_setup = light_source_setup
        self.light_source_type = light_source_type or "illuminator"
        self.test = test

        self.io_handler = io_handler.IOHandler()
        self.output_directory = self.io_handler.get_output_directory(self.label)

        self.number_events = self.light_emission_config["number_events"]

        if self._calibration_model is not None:
            self.flasher_photons = (
                self._calibration_model.get_parameter_value("flasher_photons")
                if not self.test
                else 1e8
            )
        elif self._flasher_model is not None:
            self.flasher_photons = (
                self._flasher_model.get_parameter_value("flasher_photons") if not self.test else 1e8
            )
        else:
            self.flasher_photons = 1e8

        # Ensure sim_telarray config exists on disk
        if hasattr(self._telescope_model, "write_sim_telarray_config_file"):
            self._telescope_model.write_sim_telarray_config_file(additional_model=site_model)

        # Runtime variables
        self.distance = None

    def _get_prefix(self) -> str:
        prefix = self.light_emission_config.get("output_prefix", "")
        if prefix is not None:
            return f"{prefix}_"
        return ""

    def _infer_application(self) -> tuple[str, str]:
        """Infer the LightEmission application and mode from type/setup.

        Returns
        -------
        tuple[str, str]
            (app_name, mode)
        """
        if self.light_source_type == "flasher":
            return ("ff-1m", "flasher")
        # default to illuminator xyzls, mode from setup
        mode = self.light_source_setup or "layout"
        return ("xyzls", mode)

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
            "number_events": {
                "len": 1,
                "unit": None,
                "default": 1,
                "names": ["number_events"],
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
                "default": 60 * u.cm,
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

    def _prepare_flasher_atmosphere_files(self, config_directory: Path) -> int:
        """Prepare canonical atmosphere aliases for ff-1m and return model id 1."""
        atmo_name = self._site_model.get_parameter_value("atmospheric_profile")
        self._logger.debug(f"Using atmosphere profile: {atmo_name}")

        src_path = config_directory.joinpath(atmo_name)
        for canonical in ("atmprof1.dat", "atm_profile_model_1.dat"):
            dst = config_directory.joinpath(canonical)
            if dst.exists() or dst.is_symlink():
                try:
                    dst.unlink()
                except OSError:
                    pass
            try:
                dst.symlink_to(src_path)
            except OSError:
                try:
                    shutil.copy2(src_path, dst)
                except OSError as copy_err:
                    self._logger.warning(
                        f"Failed to create atmosphere alias {dst.name}: {copy_err}"
                    )
        return 1

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

        config_directory = self.io_handler.get_model_configuration_directory(
            label=self.label, model_version=self._site_model.model_version
        )
        telpos_file = self._write_telpos_file()

        app_name, _ = self._infer_application()

        parts: list[str] = []
        # application path
        parts.append(str(self._simtel_path.joinpath("sim_telarray/LightEmission/")))
        parts.append(f"/{app_name}")

        corsika_observation_level = self._site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        )
        parts.append(
            self._build_altitude_atmo_block(
                app_name, config_directory, corsika_observation_level, telpos_file
            )
        )

        parts.append(self._build_source_specific_block(x_tel, y_tel, z_tel, config_directory))

        if self.light_source_type == "illuminator":
            parts.append(f" -A {config_directory}/")
            parts.append(f"{self._telescope_model.get_parameter_value('atmospheric_profile')}")

        parts.append(f" -o {self.output_directory}/{app_name}.iact.gz")
        parts.append("\n")

        return "".join(parts)

    def _build_altitude_atmo_block(
        self, app_name, config_directory: Path, corsika_observation_level, telpos_file: Path
    ) -> str:
        """Return CLI segment for altitude/atmosphere and telpos handling."""
        if app_name in ("ff-1m",):
            seg = []
            seg.append(" -I.")
            seg.append(f" -I{self._simtel_path.joinpath('sim_telarray/cfg')}")
            seg.append(f" -I{config_directory}")
            seg.append(f" --altitude {corsika_observation_level.to(u.m).value}")
            atmo_id = self._prepare_flasher_atmosphere_files(config_directory)
            seg.append(f" --atmosphere {atmo_id}")
            return "".join(seg)
        # default path (not used for flasher now, but kept for completeness)
        return f" -h  {corsika_observation_level.to(u.m).value} --telpos-file {telpos_file}"

    def _build_source_specific_block(self, _x_tel, _y_tel, _z_tel, _config_directory: Path) -> str:
        """Return CLI segment for light-source specific flags."""
        if self.light_source_type == "flasher":
            return self._add_flasher_command_options("")
        if self.light_source_type == "illuminator":
            return self._add_illuminator_command_options("")
        self._logger.warning("Unknown light_source_type '%s'", self.light_source_type)
        return ""

    def _add_flasher_command_options(self, command):
        """Add flasher-specific options to the script (uniform ff-1m)."""
        return self._add_flasher_options(command)

    def _add_flasher_options(self, command):
        """Add flasher options for all telescope types (ff-1m style)."""
        # For MST/LST we used to use ff-1m; now apply same for all telescopes
        flasher_xy = self._flasher_model.get_parameter_value_with_unit("flasher_position")
        flasher_distance = self._flasher_model.get_parameter_value_with_unit("flasher_depth")
        # Camera radius required for application, Radius of fiducial sphere enclosing camera
        camera_radius = (
            self._telescope_model.get_parameter_value_with_unit("camera_body_diameter")
            .to(u.cm)
            .value
            / 2
        )
        spectrum = self._flasher_model.get_parameter_value_with_unit("spectrum")
        pulse = self._flasher_model.get_parameter_value("lightpulse")
        angular = self._flasher_model.get_parameter_value("angular_distribution")
        bunch_size = self._flasher_model.get_parameter_value("bunch_size")

        # Convert to plain numbers for CLI
        fx = flasher_xy[0].to(u.cm).value
        fy = flasher_xy[1].to(u.cm).value
        dist_cm = flasher_distance.to(u.cm).value
        spec_nm = int(spectrum.to(u.nm).value)

        command += f" --events {self.number_events}"
        command += f" --photons {self.flasher_photons}"
        command += f" --bunchsize {bunch_size}"
        command += f" --xy {fx},{fy}"
        command += f" --distance {dist_cm}"
        command += f" --camera-radius {camera_radius}"
        command += f" --spectrum {spec_nm}"
        command += f" --lightpulse {pulse}"
        command += f" --angular-distribution {angular}"
        return command

    def _add_illuminator_command_options(self, command):
        """
        Add illuminator-specific command options to the light emission script.

        Parameters
        ----------
        command : str
            The command string to add options to

        Returns
        -------
        str
            The updated command string
        """
        if self.light_source_setup == "variable":
            command += f" -x {self.light_emission_config['x_pos']['default'].to(u.cm).value}"
            command += f" -y {self.light_emission_config['y_pos']['default'].to(u.cm).value}"
            command += f" -z {self.light_emission_config['z_pos']['default'].to(u.cm).value}"
            command += (
                f" -d {','.join(map(str, self.light_emission_config['direction']['default']))}"
            )
            command += f" -n {self.flasher_photons}"

        elif self.light_source_setup == "layout":
            x_cal, y_cal, z_cal = self._calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
            command += f" -x {x_cal.to(u.cm).value}"
            command += f" -y {y_cal.to(u.cm).value}"
            command += f" -z {z_cal.to(u.cm).value}"
            pointing_vector = self.calibration_pointing_direction()[0]
            command += f" -d {','.join(map(str, pointing_vector))}"

            command += f" -n {self.flasher_photons}"
            self._logger.info(f"Photons per run: {self.flasher_photons} ")

            flasher_wavelength = self._calibration_model.get_parameter_value_with_unit(
                "flasher_wavelength"
            )
            command += f" -s {int(flasher_wavelength.to(u.nm).value)}"

            pulse_shape = self._calibration_model.get_parameter_value_with_unit(
                "flasher_pulse_shape"
            )
            pulse_width = self._calibration_model.get_parameter_value_with_unit(
                "flasher_pulse_width"
            )

            command += f" -p {pulse_shape}:{pulse_width.to(u.ns).value}"
            # TODO should this be a parameter
            command += " -a isotropic"

        return command

    def _make_simtel_script(self):
        """
        Return the command to run sim_telarray using the output from the previous step.

        Returns
        -------
        str
            The command to run sim_telarray
        """
        # For flasher sims, avoid calibration pointing entirely; default angles to (0,0)
        if self.light_source_type == "flasher":
            angles = [0, 0]
        else:
            _, angles = self.calibration_pointing_direction()

        simtel_bin = self._simtel_path.joinpath("sim_telarray/bin/sim_telarray/")
        # Build command without prefix; caller will add SIM_TELARRAY_CONFIG_PATH once
        command = f"{simtel_bin} "
        command += f"-I{self._telescope_model.config_file_directory} "
        command += f"-I{simtel_bin} "
        command += f"-c {self._telescope_model.config_file_path} "
        self._remove_line_from_config(self._telescope_model.config_file_path, "array_triggers")
        self._remove_line_from_config(self._telescope_model.config_file_path, "axes_offsets")

        command += "-DNUM_TELESCOPES=1 "

        command += super().get_config_option(
            "altitude",
            self._site_model.get_parameter_value_with_unit("corsika_observation_level")
            .to(u.m)
            .value,
        )
        command += super().get_config_option(
            "atmospheric_transmission",
            self._site_model.get_parameter_value("atmospheric_transmission"),
        )
        command += super().get_config_option("TRIGGER_TELESCOPES", "1")

        command += super().get_config_option("TELTRIG_MIN_SIGSUM", "2")
        command += super().get_config_option("PULSE_ANALYSIS", "-30")
        command += super().get_config_option("MAXIMUM_TELESCOPES", 1)

        if self.light_source_type == "variable":
            command += super().get_config_option("telescope_theta", 0)
            command += super().get_config_option("telescope_phi", 0)
        else:
            command += super().get_config_option("telescope_theta", f"{angles[0]}")
            command += super().get_config_option("telescope_phi", f"{angles[1]}")

        # For flasher runs, bypass reflections on primary mirror
        if self.light_source_type == "flasher":
            command += super().get_config_option("Bypass_Optics", "1")

        command += super().get_config_option("power_law", "2.68")
        app_name, app_mode = self._infer_application()
        pref = self._get_prefix()
        command += super().get_config_option(
            "input_file", f"{self.output_directory}/{app_name}.iact.gz"
        )
        dist_suffix = ""
        if self.light_source_setup == "variable":
            try:
                dist_val = int(self._get_distance_for_plotting().to_value(u.m))
                dist_suffix = f"_d_{dist_val}"
            except Exception:  # pylint:disable=broad-except
                dist_suffix = ""

        command += super().get_config_option(
            "output_file",
            f"{self.output_directory}/{pref}{app_name}_{app_mode}{dist_suffix}.simtel.zst",
        )
        command += super().get_config_option(
            "histogram_file",
            f"{self.output_directory}/{pref}{app_name}_{app_mode}{dist_suffix}.ctsim.hdata\n",
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

    def prepare_script(self):
        """
        Build and return bash run script containing the light-emission command.

        Returns
        -------
        Path
            Full path of the run script.
        """
        self._logger.debug("Creating run bash script")

        _script_dir = self.output_directory.joinpath("scripts")
        _script_dir.mkdir(parents=True, exist_ok=True)
        _script_file = _script_dir.joinpath(f"{self._infer_application()[0]}-lightemission.sh")
        self._logger.debug(f"Run bash script - {_script_file}")

        target_out = Path(self._get_simulation_output_filename())
        if target_out.exists():
            msg = f"Simtel output file exists already, cancelling simulation: {target_out}"
            self._logger.error(msg)
            raise FileExistsError(msg)

        command_le = self._make_light_emission_script()
        command_simtel = self._make_simtel_script()

        with _script_file.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n")

            file.write(f"{command_le}\n\n")
            app_name, _ = self._infer_application()
            file.write(
                f"[ -s '{self.output_directory}/{app_name}.iact.gz' ] || "
                f"{{ echo 'LightEmission did not produce IACT file' >&2; exit 1; }}\n\n"
            )
            file.write(f"{command_simtel}\n\n")

            # Cleanup intermediate IACT file at the end of the run
            file.write(f"rm -f '{self.output_directory}/{app_name}.iact.gz'\n\n")

        _script_file.chmod(_script_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        return _script_file

    def _get_simulation_output_filename(self):
        """Get the filename of the simulation output."""
        dist_suffix = ""
        if self.light_source_setup == "variable":
            try:
                dist_val = int(self._get_distance_for_plotting().to_value(u.m))
                dist_suffix = f"_d_{dist_val}"
            except Exception:  # pylint:disable=broad-except
                dist_suffix = ""
        app_name, app_mode = self._infer_application()
        pref = self._get_prefix()
        return f"{self.output_directory}/{pref}{app_name}_{app_mode}{dist_suffix}.simtel.zst"

    def _get_distance_for_plotting(self):
        """Get the distance to be used for plotting as an astropy Quantity.

        For flasher runs, use the flasher_depth (cm) from the flasher model.
        For illuminator runs, use the configured z_pos quantity.
        Otherwise, fall back to self.distance if set, or 0 m.
        """
        if self.light_source_type == "flasher" and self._flasher_model is not None:
            return self._flasher_model.get_parameter_value_with_unit("flasher_depth").to(u.m)

        def _as_meters(val):
            if isinstance(val, u.Quantity):
                return val.to(u.m)
            try:
                return float(val) * u.m
            except (TypeError, ValueError):
                return None

        cfg = self.light_emission_config or {}
        z = cfg.get("z_pos")
        if isinstance(z, dict):
            z_def = z.get("default")
            z_val = z_def[0] if isinstance(z_def, list | tuple) and z_def else z_def
            z_q = _as_meters(z_val)
            if z_q is not None:
                return z_q

        d_q = _as_meters(getattr(self, "distance", None))
        if d_q is not None:
            return d_q

        return 0 * u.m

    def run_simulation(self) -> Path:
        """Run the light emission simulation and return the output simtel file path."""
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
            self._logger.warning(f"Expected simtel output not found: {out}")
        return out

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
        """Calculate distance(s) between telescope and calibration device."""
        if self.light_source_setup == "layout":
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
            self._logger.info(f"Distance between telescope and calibration device: {distance} m")
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
        """Simulate light emission for variable distances and return output files list."""
        if args_dict["distances_ls"] is not None:
            self.update_light_emission_config(
                "z_pos", self.distance_list(args_dict["distances_ls"])
            )
        self._logger.info(
            f"Simulating for distances: {self.light_emission_config['z_pos']['default']}"
        )
        outputs: list[Path] = []
        distances = self.calculate_distance_telescope_calibration_device()

        for current_distance, z_pos in zip(
            distances, self.light_emission_config["z_pos"]["default"]
        ):
            self.update_light_emission_config("z_pos", z_pos)
            self.distance = current_distance
            outputs.append(self.run_simulation())
        return outputs

    def simulate_layout_positions(self, args_dict):  # pylint: disable=unused-argument
        """Simulate light emission for layout positions and return output files list."""
        # args_dict kept for API symmetry; explicitly mark as unused
        del args_dict
        self.distance = self.calculate_distance_telescope_calibration_device()[0]
        # Single distance for layout
        return [self.run_simulation()]
