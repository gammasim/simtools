import logging
import os

import astropy.units as u
import eventio as eio
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.image import tailcuts_clean
from ctapipe.io import EventSource
from ctapipe.visualization import CameraDisplay
from matplotlib.colors import LinearSegmentedColormap

import simtools.utils.general as gen
from simtools.io_operations import io_handler
from simtools.simtel.simtel_runner import SimtelRunner

__all__ = ["SimulatorLightEmission"]


class SimulatorLightEmission(SimtelRunner):
    """
    SimulatorLightEmission is the interface with sim_telarray to perform
    light emission package simulations.


    Parameters
    ----------
    telescope_model:
        TelescopeModel instance to define site, telescope model etc.
    calibration_model:
        CalibrationModel instance to define calibration device
    site_model:
        SiteModel instance to define the site specific parameters
    default_le_config: dict
        defines parameters for running the sim_telarray light emission application.
    le_application: str
        Name of the application. Default sim_telarray application running
        the sim_telarray LightEmission package is xyzls.
    output_directory: str or Path
        Simtools light-emission output directory.
    simtel_source_path: str or Path
        Location of sim_telarray installation.
    config_data: dict
        Dict containing the configurable parameters.
    config_file: str or Path
        Path of the yaml file containing the configurable parameters.
    force_simulate: bool
        Remove existing files and force re-running of the light emission simulation.
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
        """
        Initialize SimtelRunner
        """
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
        self.distance = np.sqrt(
            self.default_le_config["x_pos"]["default"] ** 2
            + self.default_le_config["y_pos"]["default"] ** 2
            + self.default_le_config["z_pos"]["default"] ** 2
        )
        self.light_source_type = light_source_type
        self._telescope_model.export_config_file()

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Builds a LightEmission object from kwargs only.
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
        print("args", args)

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

        Returns:
        list: The pointing vector from the calibration device to the telescope.
        """
        # x_cal = self._calibration_model.get_parameter_value("x_pos")
        # y_cal = self._calibration_model.get_parameter_value("y_pos")
        # z_cal = self._calibration_model.get_parameter_value("z_pos")
        x_cal = self.default_le_config["x_pos_ILLN-01"]["default"]
        y_cal = self.default_le_config["y_pos_ILLN-01"]["default"]
        z_cal = self.default_le_config["z_pos_ILLN-01"]["default"]

        cal_vect = np.array([x_cal, y_cal, z_cal])

        # x_tel = self._telescope_model.get_parameter_value("x_pos")
        # y_tel = self._telescope_model.get_parameter_value("y_pos")
        # z_tel = self._telescope_model.get_parameter_value("z_pos")
        x_tel = self.default_le_config["x_pos"]["real"]
        y_tel = self.default_le_config["y_pos"]["real"]
        z_tel = self.default_le_config["z_pos"]["real"]
        tel_vect = np.array([x_tel, y_tel, z_tel])

        direction_vector = cal_vect - tel_vect
        pointing_vector = direction_vector / np.linalg.norm(direction_vector)

        return pointing_vector.tolist()

    def _make_light_emission_script(self, **kwargs):  # pylint: disable=unused-argument
        command = f" rm {self.output_directory}/"
        command += f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n"
        command += str(self._simtel_source_path.joinpath("sim_telarray/LightEmission/"))
        command += f"/{self.le_application[0]}"
        command += " -n 1e10"

        command += f" -x {self.default_le_config['x_pos']['default'].value}"
        command += f" -y {self.default_le_config['y_pos']['default'].value}"
        command += f" -z {self.default_le_config['z_pos']['default'].value}"
        if self.le_application[1] == "variable":
            command += f" -d {','.join(map(str, self.default_le_config['direction']['default']))}"
        elif self.le_application[1] == "static":
            command += f" -d {','.join(map(str, self.calibration_pointing_direction()))}"

        if self.light_source_type == "led":
            command += f" -n {self._calibration_model.get_parameter_value('photons_per_run')}"

            # currently we use the same wavelength
            command += f" -s {self._calibration_model.get_parameter_value('laser_wavelength')}"

            # pulse
            command += (
                f" -p Gauss:{self._calibration_model.get_parameter_value('led_pulse_sigtime')}"
            )
            # {self._calibration_model.get_parameter_value('led_pulse_offset')}"
            # TODO further parameters require different application

            # command += f" -s {self._calibration_model.get_parameter_value('led_var_photons')}"
            # command += f" -s {self._calibration_model.get_parameter_value('pedestal_events')}"

        elif self.light_source_type == "laser":
            command += f" -n {self._calibration_model.get_parameter_value('photons_per_run')}"

            command += f" -s {self._calibration_model.get_parameter_value('laser_wavelength')}"
            command += f" -N {self._calibration_model.get_parameter_value('laser_events')}"
            # command += (
            #    f" -s {self._calibration_model.get_parameter_value('laser_external_trigger')}"
            # )
            command += f" -s {self._calibration_model.get_parameter_value('laser_pulse_exptime')}"
            command += f" -s {self._calibration_model.get_parameter_value('laser_pulse_offset')}"
            command += f" -s {self._calibration_model.get_parameter_value('laser_pulse_sigtime')}"
            command += f" -p {self._calibration_model.get_parameter_value('laser_pulse_twidth')}"

            # command += f" -a {self._calibration_model.get_parameter_value('beam_shape').value}"
            # command += f"{self._calibration_model.get_parameter_value('beam_width').value}"
            # command += f" -p {self._calibration_model.get_parameter_value('pulse_shape').value}"
            # command += f" {self._calibration_model.get_parameter_value('pulse_width').value}"

        command += f" -A {self.output_directory}/model/"
        command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"
        command += f" -o {self.output_directory}/{self.le_application[0]}.iact.gz"
        command += "\n"

        return command

    def _make_simtel_script(self, **kwargs):  # pylint: disable=unused-argument
        """Return the command to run simtel_array."""

        # LightEmission
        command = f"{self._simtel_source_path.joinpath('sim_telarray/bin/sim_telarray/')}"
        command += f" -c {self._telescope_model.get_config_file()}"
        # command += " -c /workdir/sim_telarray/sim_telarray/cfg/CTA/CTA-ULTRA6-MST-NectarCam.cfg"

        def remove_line_from_config(file_path, line_prefix):
            with open(file_path, "r", encoding="utf-8") as file:
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
        # command += super()._config_option("maximum_telescopes", "1")
        # command += super()._config_option("trigger_telescopes", "1")
        command += super()._config_option(
            "atmospheric_transmission",
            self._telescope_model.get_parameter_value("atmospheric_transmission"),
        )
        # command += super()._config_option("trigger_current_limit", "1e10")
        command += super()._config_option("show", "all")
        # command += super()._config_option("random_state", "none")
        # command += super()._config_option("ONLY_TRIGGERED_TELESCOPES", "0")
        # command += super()._config_option("ONLY_TRIGGERED_ARRAYS", "0")
        command += super()._config_option("TRIGGER_CURRENT_LIMIT", "20")
        command += super()._config_option("TRIGGER_TELESCOPES", "1")
        # command += super()._config_option("ARRAY_TRIGGER", "None")
        command += super()._config_option("TELTRIG_MIN_SIGSUM", "7.8")

        command += super()._config_option("PULSE_ANALYSIS", "-30")
        command += super()._config_option("SUM_BEFORE_PEAK", "-30")
        command += super()._config_option("DEFAULT_TRIGGER", "AnalogSum")

        # command += super()._config_option("ARRAY_WINDOW", "1000")
        # command += super()._config_option("FAKE_TRIGGER", "1")

        # from light_emission_default config
        # command += super()._config_option(
        #    "telescope_theta",
        #    self.config.zenith_angle + self.config.off_axis_angle,
        # )
        command += super()._config_option("telescope_theta", "0")
        command += super()._config_option("telescope_phi", "0")
        command += super()._config_option("power_law", "2.68")
        # command += super()._config_option("FADC_BINS", str(int(self.config.fadc_bins))) #in config
        command += super()._config_option(
            "input_file", f"{self.output_directory}/{self.le_application[0]}.iact.gz"
        )
        command += super()._config_option(
            "output_file",
            f"{self.output_directory}/"
            f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n",
        )

        return command

    def _create_postscript(self, **kwargs):  # pylint: disable=unused-argument
        """
        writes out post-script file using read_cta
        """
        postscript_dir = self.output_directory.joinpath("postscripts")
        postscript_dir.mkdir(parents=True, exist_ok=True)

        command = str(self._simtel_source_path.joinpath("hessioxxx/bin/read_cta"))
        command += " --min-tel 1 --min-trg-tel 1"
        command += " -q --integration-scheme 4 --integration-window 7,3 -r 5"
        command += " --plot-with-sum-only"
        command += " --plot-with-pixel-amp --plot-with-pixel-id"
        # command += f" --plot-with-title 'tel {self._telescope_model.name}"
        # command += "dist: {self.default_le_config['z_pos']['default'].value/100}'"

        command += (
            f" -p {postscript_dir}/"
            f"{self.le_application[0]}_{self.le_application[1]}_d_{self.distance.to(u.m).value}.ps"
        )
        command += (
            f" {self.output_directory}/"
            f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz\n"
        )
        # command += f"ps2pdf {self.output_directory}/{self.le_application}.ps
        #  {self.output_directory}/{self.le_application}.pdf"
        return command

    def plot_simtel(self):
        """
        plot true p.e. in camera frame using eventio
        """

        def camera_rotation(pixel_x, pixel_y, cam_rot):
            pixel_x_derot = pixel_x * np.cos(cam_rot) - pixel_y * np.sin(cam_rot)
            pixel_y_derot = pixel_x * np.sin(cam_rot) + pixel_y * np.cos(cam_rot)

            return pixel_x_derot, pixel_y_derot

        simtel_file = eio.SimTelFile(
            f"{self.output_directory}/{self.le_application[0]}_{self.le_application[1]}.simtel.gz"
        )
        for array_event in simtel_file:
            array_event_s = array_event
            photo_electrons = array_event["photoelectrons"]

        pixel_x = simtel_file.telescope_descriptions[1]["camera_settings"]["pixel_x"]
        pixel_y = simtel_file.telescope_descriptions[1]["camera_settings"]["pixel_y"]
        cam_rot = simtel_file.telescope_descriptions[1]["camera_settings"]["cam_rot"]
        n_pixels = simtel_file.telescope_descriptions[1]["camera_settings"]["n_pixels"]

        n_pe = photo_electrons[0]["photoelectrons"]

        pixels_clean = array_event_s["telescope_events"][1]["pixel_lists"][1]["pixels"]
        tel_name = simtel_file.telescope_meta[1][b"CAMERA_CONFIG_NAME"].decode("utf-8")

        pixel_x_derot, pixel_y_derot = camera_rotation(pixel_x, pixel_y, cam_rot)

        palette = ["#1B1A1D", "#69809F", "#B3C4D5", "#F45B3B", "#ff0000"]
        cmap = LinearSegmentedColormap.from_list("camera", palette, N=200)
        cmap.set_bad("#4f4f4f")
        norm = matplotlib.colors.LogNorm(vmin=0.1, vmax=200)

        fig, ax = plt.subplots(1, dpi=300)
        ax.scatter(
            pixel_y_derot,
            pixel_x_derot,
            color=cmap(norm(n_pe)),
            marker=(6, 0, -np.rad2deg(cam_rot)),
            edgecolor="grey",
            linewidths=0.5,
        )

        ax.text(-1, 1.45, f"(from .. of {photo_electrons[0]['n_pe']} true p.e.)", size="xx-small")
        plt.title(f"Simulation of {tel_name}", pad=35)

        ax.text(
            -1,
            1.35,
            f"Number of pixels after cleaning {pixels_clean}",
            horizontalalignment="left",
            size="xx-small",
        )
        ax.text(
            -1,
            1.25,
            f"$N_{{\\mathrm{{pixels}}}}=$ {n_pixels}",
            horizontalalignment="left",
            size="xx-small",
        )

        ax.set_axis_off()
        ax.set_aspect("equal")
        fig.savefig(f"{self.output_directory}/{self.le_application[0]}_test.pdf")

    def plot_simtel_ctapipe(self, return_cleaned=0):
        """
        reads in simtel file and plots reconstructed photo electrons via ctapipe
        """
        filename = (
            f"{self.output_directory}/"
            f"{self.le_application[0]}_{self.le_application[1]}.simtel.gz"
        )
        source = EventSource(filename, max_events=1)
        event = None
        for event in source:
            print(event.index.event_id)
        tel_id = sorted(event.r1.tel.keys())[0]

        calib = CameraCalibrator(subarray=source.subarray)

        calib(event)
        geometry = source.subarray.tel[1].camera.geometry

        image = event.dl1.tel[tel_id].image
        cleaned = image.copy()

        if return_cleaned:
            mask = tailcuts_clean(
                geometry, image, picture_thresh=7, boundary_thresh=5, min_number_picture_neighbors=0
            )
            cleaned[~mask] = 0

        fig, ax = plt.subplots(1, 1, dpi=300)
        title = f"CT{tel_id}, run {event.index.obs_id} event {event.index.event_id}"
        disp = CameraDisplay(geometry, image=cleaned, norm="symlog", ax=ax)
        disp.cmap = "RdBu_r"
        disp.add_colorbar(fraction=0.02, pad=-0.1)
        disp.set_limits_percent(100)
        ax.set_title(title, pad=20)
        ax.annotate(
            f"tel type: {source.subarray.tel[1].type.name}\n"
            f"optics: {source.subarray.tel[1].optics.name}\n"
            f"camera: {source.subarray.tel[1].camera_name}\n"
            f"distance: {self.default_le_config['z_pos']['default'].to(u.m)}",
            (0, 0),
            (0.1, 1),
            xycoords="axes fraction",
            va="top",
            size=7,
        )
        ax.annotate(
            f"dl1 image,\ntotal $p.e._{{reco}}$: {np.round(np.sum(image))}\n",
            (0, 0),
            (0.75, 1),
            xycoords="axes fraction",
            va="top",
            ha="left",
            size=7,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    def prepare_script(self, generate_postscript=False):
        """
        Builds and returns the full path of the bash run script
        containing the light-emission command.

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
                command_plot = self._create_postscript()
                file.write("# Generate postscript\n\n")
                file.write(f"{command_plot}\n\n")
                file.write("# End\n\n")

        os.system(f"chmod ug+x {self._script_file}")
        return self._script_file
