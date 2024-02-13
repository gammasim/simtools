import logging
import os

import astropy.units as u

import simtools.utils.general as gen
from simtools.io_operations import io_handler
from simtools.simtel.simtel_runner import SimtelRunner

__all__ = ["SimulatorLightEmission"]


class SimulatorLightEmission(SimtelRunner):
    """
    SimulatorLightEmission is the interface with sim_telarray to perform
    light emission package simulations.

    Configurable parameters:
        zenith_angle:
            len: 1
            unit: deg
            default: 20 deg

    Parameters
    ----------
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
        output_dir,
        label=None,
        simtel_source_path=None,
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
        self._base_directory = self.io_handler.get_output_directory(self.label)
        print("base directory: ", self._base_directory)
        # Loading config_data, currently use default config
        self.config = gen.validate_config_data(
            gen.collect_data_from_file_or_dict(config_file, config_data),
            self.light_emission_default_configuration(),
        )

        # LightEmission - default parameters
        self._rep_number = 0
        self.runs = 1
        self.photons_per_run = 100000 if not test else 10000

        self.le_application = le_application
        self.default_le_config = default_le_config
        self.output_dir = output_dir
        # self._load_required_files(force_simulate)

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
                "output_dir",
                "label",
                "simtel_source_path",
            ],
            **kwargs,
        )
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
        # ./xyzls -a Gauss:3 -p Gauss:0.1 -n 1e5,1e6,1e7
        command = f" rm {self.output_dir}/{self.le_application}.simtel.gz\n"
        command += str(self._simtel_source_path.joinpath("sim_telarray/LightEmission/"))
        command += f"/{self.le_application}"
        command += f" -a {self.default_le_config['beam_shape']['default']}:"
        command += f"{self.default_le_config['beam_width']['default'].value}"
        command += f" -p {self.default_le_config['pulse_shape']['default']}:"
        command += f"{self.default_le_config['pulse_width']['default'].value}"
        command += " -n 1e7"
        # command += f" -A {self._simtel_source_path.joinpath('sim_telarray/
        # cfg/common/atmprof1.dat')}"
        command += f" -A {self.output_dir}/model/"
        command += f"{self._telescope_model.get_parameter_value('atmospheric_profile')}"
        command += f" -o {self.output_dir}/{self.le_application}.iact.gz"
        command += "\n"
        print(command)
        return command

    def _make_simtel_script(self, **kwargs):  # pylint: disable=unused-argument
        """Return the command to run simtel_array."""

        # LightEmission
        command = f"{self._simtel_source_path.joinpath('sim_telarray/bin/sim_telarray/')}"
        command += f" -c {self._simtel_source_path.joinpath('sim_telarray/cfg/CTA')}/"
        command += "CTA-PROD6-MST-NectarCam.cfg"
        # command += f"{self._telescope_model.get_config_file()}" # general selection
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
        command += super()._config_option("telescope_phi", "0")
        command += super()._config_option("power_law", "2.68")
        command += super()._config_option("FADC_BINS", str(int(self.config.fadc_bins)))
        command += super()._config_option(
            "input_file", f"{self.output_dir}/{self.le_application}.iact.gz"
        )
        command += super()._config_option(
            "output_file", f"{self.output_dir}/{self.le_application}.simtel.gz\n"
        )

        return command

    def _make_plot_script(self, **kwargs):  # pylint: disable=unused-argument
        command = str(self._simtel_source_path.joinpath("hessioxxx/bin/read_cta_nr"))
        command += f" -p {self.output_dir}/{self.le_application}.ps"
        command += f" {self.output_dir}/{self.le_application}.simtel.gz\n"
        # command += f"ps2pdf {self.output_dir}/{self.le_application}.ps
        #  {self.output_dir}/{self.le_application}.pdf"
        return command

    def prepare_script(self, test=False, plot=False, extra_commands=None):
        """
        Builds and returns the full path of the bash run script
        containing the light-emission command.

        Parameters
        ----------
        plot: bool
            If output should be plotted.

        extra_commands: str
            Additional commands for running simulations given in config.yml.

        Returns
        -------
        Path
            Full path of the run script.
        """
        self._logger.debug("Creating run bash script")

        self._script_dir = self._base_directory.joinpath("scripts")
        self._script_dir.mkdir(parents=True, exist_ok=True)
        self._script_file = self._script_dir.joinpath(f"{self.le_application}-lightemission")
        self._logger.debug(f"Run bash script - {self._script_file}")

        self._logger.debug(f"Extra commands to be added to the run script {extra_commands}")

        command_le = self._make_light_emission_script()
        command_simtel = self._make_simtel_script()
        command_plot = self._make_plot_script()

        with self._script_file.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")

            # if extra_commands is not None:
            #    file.write("# Writing extras\n")
            #    for line in extra_commands:
            #        file.write(f"{line}\n")
            #    file.write("# End of extras\n\n")

            file.write(f"{command_le}\n\n")
            file.write(f"{command_simtel}\n\n")
            if plot:
                file.write(f"{command_plot}\n\n")

            #  TODO: Add functionality to run several telescope configs at once

        if test:
            #  TODO: Add
            pass

        os.system(f"chmod ug+x {self._script_file}")
        return self._script_file
