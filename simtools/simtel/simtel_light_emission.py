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
        TelescopeModel class to define site, telescope model etc.
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
        default_le_config,
        le_application,
        simtel_source_path,
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
                "default_le_config",
                "le_application",
                "simtel_source_path",
                "label",
            ],
            **kwargs,
        )
        print(config_data)

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

    def _make_light_emission_script(self, **kwargs):  # pylint: disable=unused-argument
        command = f" rm {self.output_directory}/{self.le_application}.simtel.gz\n"
        command += str(self._simtel_source_path.joinpath("sim_telarray/LightEmission/"))
        command += f"/{self.le_application}"
        # command += f" -a {self.default_le_config['beam_shape']['default']}:"
        # command += f"{self.default_le_config['beam_width']['default'].value}"
        # command += f" -p {self.default_le_config['pulse_shape']['default']}:"
        # command += f"{self.default_le_config['pulse_width']['default'].value}"
        command += f" -n {self.photons_per_run}"
        command += f" -x {self.default_le_config['x_pos']['default'].value}"
        command += f" -y {self.default_le_config['y_pos']['default'].value}"
        command += f" -z {self.default_le_config['z_pos']['default'].value}"
        command += f" -d {','.join(map(str, self.default_le_config['direction']['default']))}"
        command += f" -A {self.output_directory}/model/"
        command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"
        command += f" -o {self.output_directory}/{self.le_application}.iact.gz"
        command += "\n"
        return command

    def _make_simtel_script(self, **kwargs):  # pylint: disable=unused-argument
        """Return the command to run simtel_array."""

        # LightEmission
        command = f"{self._simtel_source_path.joinpath('sim_telarray/bin/sim_telarray/')}"
        command += f" -c {self._telescope_model.get_config_file()}"
        command += " -DNUM_TELESCOPES=1"
        command += " -I../cfg/CTA"
        command += "iobuf_maximum=1000000000"
        command += super()._config_option(
            "altitude", self._telescope_model.get_parameter_value("altitude")
        )
        command += super()._config_option("maximum_telescopes", "1")
        command += super()._config_option("trigger_telescopes", "1")
        command += super()._config_option(
            "atmospheric_transmission",
            self._telescope_model.get_parameter_value("atmospheric_transmission"),
        )
        # from light_emission_default config
        # command += super()._config_option(
        #    "telescope_theta",
        #    self.config.zenith_angle + self.config.off_axis_angle,
        # )
        # command += super()._config_option("telescope_theta", "70")
        command += super()._config_option("telescope_phi", "0")
        command += super()._config_option("power_law", "2.68")
        command += super()._config_option("FADC_BINS", str(int(self.config.fadc_bins)))
        command += super()._config_option(
            "input_file", f"{self.output_directory}/{self.le_application}.iact.gz"
        )
        command += super()._config_option(
            "output_file", f"{self.output_directory}/{self.le_application}.simtel.gz\n"
        )

        return command

    def _create_postscript(self, **kwargs):  # pylint: disable=unused-argument
        """
        writes out post-script file using read_cta
        """
        command = str(self._simtel_source_path.joinpath("hessioxxx/bin/read_cta"))
        command += " --min-tel 1 --min-trg-tel 1"
        command += " -q --integration-scheme 4 --integration-window 7,3 -r 5"
        command += " --plot-with-sum-only"
        command += " --plot-with-pixel-amp --plot-with-pixel-id"
        # command += f" --plot-with-title 'tel {self._telescope_model.name}"
        # command += "dist: {self.default_le_config['z_pos']['default'].value/100}'"

        command += f" -p {self.output_directory}/{self.le_application}.ps"
        command += f" {self.output_directory}/{self.le_application}.simtel.gz\n"
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

        simtel_file = eio.SimTelFile(f"{self.output_directory}/{self.le_application}.simtel.gz")
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
        fig.savefig(f"{self.output_directory}/{self.le_application}_test.pdf")

    def plot_simtel_ctapipe(self, return_cleaned=0):
        """
        reads in simtel file and plots reconstructed photo electrons via ctapipe
        """
        filename = f"{self.output_directory}/{self.le_application}.simtel.gz"
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
        self._script_file = self._script_dir.joinpath(f"{self.le_application}-lightemission")
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
