"""Light emission simulation (e.g. illuminators or flashers)."""

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
    Light emission simulation (e.g. illuminators or flashers).

    Interface with sim_telarray to perform light emission package simulations.
    The light emission package is used to simulate an artificial light source, used for calibration.

    Parameters
    ----------
    telescope_model : TelescopeModel
        Model of the telescope to be simulated
    calibration_model : CalibrationModel
        Model of the calibration device to be simulated
    site_model : SiteModel, optional
        Model of the site
    light_emission_config : dict, optional
        Configuration for the light emission
    simtel_path : Path, optional
        Path to the sim_telarray installation
    label : str, optional
        Label for the simulation
    test : bool, optional
        Whether this is a test run
    """

    def __init__(
        self,
        telescope_model,
        calibration_model,
        site_model=None,
        light_emission_config=None,
        simtel_path=None,
        label=None,
        test=False,
    ):
        """Initialize SimtelRunner."""
        super().__init__(simtel_path=simtel_path, label=label, corsika_config=None)

        self._logger = logging.getLogger(__name__)

        self.telescope_model = telescope_model
        self.calibration_model = calibration_model
        self.site_model = site_model
        self.light_emission_config = light_emission_config or {}
        self.light_source_setup = self.light_emission_config.get("light_source_setup")
        self.test = test

        self.io_handler = io_handler.IOHandler()
        self.output_directory = self.io_handler.get_output_directory(self.label)

        self.light_source_type = self.calibration_model.get_parameter_value("flasher_type")
        self.flasher_photons = (
            self.calibration_model.get_parameter_value("flasher_photons") if not self.test else 1e8
        )

        # Ensure sim_telarray config exists on disk
        if hasattr(self.telescope_model, "write_sim_telarray_config_file"):
            self.telescope_model.write_sim_telarray_config_file(additional_model=site_model)

        # Runtime variables
        # TODO check if needed
        self.distance = None

    def _get_prefix(self) -> str:
        prefix = self.light_emission_config.get("output_prefix", "")
        if prefix is not None:
            return f"{prefix}_"
        return ""

    def _infer_application(self) -> tuple[str, str]:
        """Infer the LightEmission application and mode from type.

        Returns
        -------
        tuple[str, str]
            (app_name, mode)
        """
        if self.light_source_type == "flasher":
            return ("ff-1m", "flasher")
        # default to illuminator xyzls, mode from setup
        return ("xyzls", self.light_source_setup or "layout")

    def calibration_pointing_direction(self):
        """
        Calculate the pointing of the calibration device towards the telescope.

        This is for calibration devices not installed on telescopes (e.g. illuminators).

        Returns
        -------
        list
            The pointing vector from the calibration device to the telescope.
        """
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
        telescope_position_file = self.output_directory.joinpath("telescope_position.dat")
        x_tel, y_tel, z_tel = self.telescope_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )
        x_tel, y_tel, z_tel = [coord.to(u.cm).value for coord in (x_tel, y_tel, z_tel)]

        radius = self.telescope_model.get_parameter_value_with_unit("telescope_sphere_radius")
        radius = radius.to(u.cm).value  # Convert radius to cm
        with telescope_position_file.open("w", encoding="utf-8") as file:
            file.write(f"{x_tel} {y_tel} {z_tel} {radius}\n")

        return telescope_position_file

    def _prepare_flasher_atmosphere_files(self, config_directory: Path) -> int:
        """Prepare canonical atmosphere aliases for ff-1m and return model id 1."""
        atmosphere_file = self.site_model.get_parameter_value("atmospheric_profile")
        self._logger.debug(f"Using atmosphere profile: {atmosphere_file}")

        src_path = config_directory.joinpath(atmosphere_file)
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
        x_tel, y_tel, z_tel = self.telescope_model.get_parameter_value_with_unit(
            "array_element_position_ground"
        )

        config_directory = self.io_handler.get_model_configuration_directory(
            label=self.label, model_version=self.site_model.model_version
        )
        app_name, _ = self._infer_application()

        parts: list[str] = []
        # application path
        parts.append(str(self._simtel_path.joinpath("sim_telarray/LightEmission/")))
        parts.append(f"/{app_name}")

        corsika_observation_level = self.site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        )
        parts.append(
            self._build_altitude_atmo_block(app_name, config_directory, corsika_observation_level)
        )

        parts.append(self._build_source_specific_block(x_tel, y_tel, z_tel, config_directory))

        if self.light_source_type == "illuminator":
            parts.append(f" -A {config_directory}/")
            parts.append(f"{self.telescope_model.get_parameter_value('atmospheric_profile')}")

        parts.append(f" -o {self.output_directory}/{app_name}.iact.gz")
        parts.append("\n")

        return "".join(parts)

    def _build_altitude_atmo_block(self, app_name, config_directory, corsika_observation_level):
        """Return CLI segment for altitude/atmosphere and telescope_position handling."""
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
        return (
            f" -h  {corsika_observation_level.to(u.m).value} "
            f"--telpos-file {self._write_telescope_position_file()}"
        )

    def _build_source_specific_block(self, _x_tel, _y_tel, _z_tel, _config_directory: Path) -> str:
        """Return CLI segment for light-source specific flags."""
        if self.light_source_type.lower() == "flasher":
            return self._add_flasher_command_options("")
        if self.light_source_type.lower() == "illuminator":
            return self._add_illuminator_command_options("")
        raise ValueError(f"Unknown light_source_type '{self.light_source_type}'")

    def _add_flasher_command_options(self, command):
        """Add flasher-specific options to the script (uniform ff-1m)."""
        return self._add_flasher_options(command)

    def _add_flasher_options(self, command):
        """
        Add flasher options for all telescope types (ff-1m style).

        TODO fixed wavelength, no spectral distribution

        """
        flasher_xyz = self.calibration_model.get_parameter_value_with_unit("flasher_position")
        # Camera radius required for application, radius of fiducial sphere enclosing camera
        # TODO this might not be a diameter for squared cameras
        camera_radius = (
            self.telescope_model.get_parameter_value_with_unit("camera_body_diameter")
            .to(u.cm)
            .value
            / 2
        )
        flasher_wavelength = self.calibration_model.get_parameter_value_with_unit(
            "flasher_wavelength"
        )
        bunch_size = self.calibration_model.get_parameter_value("flasher_bunch_size")
        dist_cm = self.calculate_distance_focal_plane_calibration_device().to(u.cm).value

        command += f" --events {self.light_emission_config['number_events']}"
        command += f" --photons {self.flasher_photons}"
        command += f" --bunchsize {bunch_size}"
        command += f" --xy {flasher_xyz[0].to(u.cm).value},{flasher_xyz[1].to(u.cm).value}"
        command += f" --distance {dist_cm}"
        command += f" --camera-radius {camera_radius}"
        command += f" --spectrum {int(flasher_wavelength.to(u.nm).value)}"
        command += f" --lightpulse {self._get_pulse_shape_string_for_sim_telarray()}"
        command += " --angular-distribution "
        command += f"{self._get_angular_distribution_string_for_sim_telarray()}"
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
            x_cal, y_cal, z_cal = self.calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
            command += f" -x {x_cal.to(u.cm).value}"
            command += f" -y {y_cal.to(u.cm).value}"
            command += f" -z {z_cal.to(u.cm).value}"
            pointing_vector = self.calibration_pointing_direction()[0]
            command += f" -d {','.join(map(str, pointing_vector))}"

            command += f" -n {self.flasher_photons}"
            self._logger.info(f"Photons per run: {self.flasher_photons} ")

            flasher_wavelength = self.calibration_model.get_parameter_value_with_unit(
                "flasher_wavelength"
            )
            command += f" -s {int(flasher_wavelength.to(u.nm).value)}"

            pulse_shape = self.calibration_model.get_parameter_value_with_unit(
                "flasher_pulse_shape"
            )
            pulse_width = self.calibration_model.get_parameter_value_with_unit(
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
        command += f"-I{self.telescope_model.config_file_directory} "
        command += f"-I{simtel_bin} "
        command += f"-c {self.telescope_model.config_file_path} "
        self._remove_line_from_config(self.telescope_model.config_file_path, "array_triggers")
        self._remove_line_from_config(self.telescope_model.config_file_path, "axes_offsets")

        command += "-DNUM_TELESCOPES=1 "

        command += super().get_config_option(
            "altitude",
            self.site_model.get_parameter_value_with_unit("corsika_observation_level")
            .to(u.m)
            .value,
        )
        command += super().get_config_option(
            "atmospheric_transmission",
            self.site_model.get_parameter_value("atmospheric_transmission"),
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
                dist_val = int(self._get_distance_for_file_name().to_value(u.m))
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
                dist_val = int(self._get_distance_for_file_name().to_value(u.m))
                dist_suffix = f"_d_{dist_val}"
            except Exception:  # pylint:disable=broad-except
                dist_suffix = ""
        app_name, app_mode = self._infer_application()
        pref = self._get_prefix()
        return f"{self.output_directory}/{pref}{app_name}_{app_mode}{dist_suffix}.simtel.zst"

    def _get_distance_for_file_name(self):
        """
        Get the distance to be used for file names as an astropy Quantity.

        For flasher-type light sources, use the calculated distance between focal plane and
        calibration device. For illuminator-type light sources, use geometrical distance
        between telescope and calibration device.

        TODO - this is a mess

        Returns
        -------
        astropy.Quantity
            Distance in meters (default: 0 m)
        """
        if self.light_source_type == "flasher":
            return self.calculate_distance_focal_plane_calibration_device()

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
        """
        Calculate distance(s) between telescope and calibration device.

        For illuminator-type light sources.

        Returns
        -------
        list
            List of distances between calibration device and telescope in meters.
        """
        if self.light_source_setup == "layout":
            # Layout positions: Use DB coordinates
            x_cal, y_cal, z_cal = self.calibration_model.get_parameter_value_with_unit(
                "array_element_position_ground"
            )
            x_cal, y_cal, z_cal = [coord.to(u.m).value for coord in (x_cal, y_cal, z_cal)]
            x_tel, y_tel, z_tel = self.telescope_model.get_parameter_value_with_unit(
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

    def calculate_distance_focal_plane_calibration_device(self):
        """
        Calculate distance between focal plane and calibration device.

        For flasher-type light sources.
        """
        distance = self.telescope_model.get_parameter_value_with_unit("focal_length")
        # TODO check sign
        distance += self.calibration_model.get_parameter_value_with_unit("flasher_position")[2]
        return distance

    def simulate_variable_distances(self, args_dict):
        """
        Simulate light emission for variable distances.

        For Illuminator-type light sources.

        Parameters
        ----------
        args_dict : dict
            Dictionary of arguments

        Returns
        -------
        list[Path]
            List of output simtel file paths.
        """
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
        # TODO is this function needed to be with unused arguments?
        # args_dict kept for API symmetry; explicitly mark as unused
        del args_dict
        self.distance = self.calculate_distance_telescope_calibration_device()[0]
        # Single distance for layout
        return [self.run_simulation()]

    def _get_angular_distribution_string_for_sim_telarray(self):
        """
        Get the angular distribution string for sim_telarray.

        Returns
        -------
        str
            The angular distribution string.
        """
        option_string = self.calibration_model.get_parameter_value("angular_distribution")
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
        option_string = self.calibration_model.get_parameter_value("flasher_pulse_shape")
        width = self.calibration_model.get_parameter_value_with_unit("flasher_pulse_width")
        return f"{option_string}:{width.to(u.ns).value}" if width is not None else option_string
