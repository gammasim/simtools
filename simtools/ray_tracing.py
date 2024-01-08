import gzip
import logging
import shlex
import shutil
import subprocess
from copy import copy
from math import pi, tan
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

import simtools.utils.general as gen
from simtools.io_operations import io_handler
from simtools.model.model_utils import compute_telescope_transmission
from simtools.model.telescope_model import TelescopeModel
from simtools.psf_analysis import PSFImage
from simtools.simtel.simtel_runner_ray_tracing import SimtelRunnerRayTracing
from simtools.utils import names
from simtools.visualization import visualize

__all__ = ["RayTracing"]


class RayTracing:
    """
    Class for handling ray tracing simulations and analysis.

    Parameters
    ----------
    telescope_model: TelescopeModel
        Instance of the TelescopeModel class.
    label: str
        Instance label.
    simtel_source_path: str (or Path)
        Location of sim_telarray installation.
    config_data: dict.
        Dict containing the configurable parameters.
    config_file: str or Path
        Path of the yaml file containing the configurable parameters.
    """

    YLABEL = {
        "d80_cm": r"$D_{80}$",
        "d80_deg": r"$D_{80}$",
        "eff_area": "Eff. mirror area",
        "eff_flen": "Eff. focal length",
    }

    def __init__(
        self,
        telescope_model,
        simtel_source_path,
        label=None,
        config_data=None,
        config_file=None,
    ):
        """
        Initialize RayTracing class.
        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initializing RayTracing class")

        self._simtel_source_path = Path(simtel_source_path)
        self._io_handler = io_handler.IOHandler()

        self._telescope_model = self._validate_telescope_model(telescope_model)

        self.config = gen.validate_config_data(
            gen.collect_data_from_file_or_dict(config_file, config_data),
            SimtelRunnerRayTracing.ray_tracing_default_configuration(False),
        )

        # Due to float representation, round the off-axis angles so the values in results table
        # are the same as provided.
        self.config = self.config._replace(off_axis_angle=np.around(self.config.off_axis_angle, 5))

        self.label = label if label is not None else self._telescope_model.label

        self._output_directory = self._io_handler.get_output_directory(
            label=self.label, sub_dir="ray-tracing"
        )

        # Loading relevant attributes in case of single mirror mode.
        if self.config.single_mirror_mode:
            # Recalculating source distance.
            self._logger.debug(
                "Single mirror mode is activated - "
                "source distance is being recalculated to 2 * flen"
            )
            mir_flen = self._telescope_model.get_parameter_value("mirror_focal_length")
            self._source_distance = 2 * float(mir_flen) * u.cm.to(u.km)  # km

            # Setting mirror numbers.
            if self.config.mirror_numbers[0] == "all":
                self._mirror_numbers = list(
                    range(0, self._telescope_model.mirrors.number_of_mirrors)
                )
            else:
                self._mirror_numbers = self.config.mirror_numbers
        else:
            self._source_distance = self.config.source_distance

        self._psf_images = None
        self._results = None
        self._has_results = False

        # Results file
        file_name_results = names.ray_tracing_results_file_name(
            self._telescope_model.site,
            self._telescope_model.name,
            self._source_distance,
            self.config.zenith_angle,
            self.label,
        )
        self._output_directory.joinpath("results").mkdir(parents=True, exist_ok=True)
        self._file_results = self._output_directory.joinpath("results").joinpath(file_name_results)

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Builds a RayTracing object from kwargs only.
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
                "label",
                "simtel_source_path",
            ],
            **kwargs,
        )
        return cls(**args, config_data=config_data)

    def __repr__(self):
        """
        String representation of RayTracing class.
        """
        return f"RayTracing(label={self.label})\n"

    def _validate_telescope_model(self, tel):
        """Validate TelescopeModel"""
        if isinstance(tel, TelescopeModel):
            self._logger.debug("RayTracing contains a valid TelescopeModel")
            return tel

        msg = "Invalid TelescopeModel"
        self._logger.error(msg)
        raise ValueError(msg)

    def simulate(self, test=False, force=False):
        """
        Simulate RayTracing using SimtelRunnerRayTracing.

        Parameters
        ----------
        test: bool
            Test flag will make it faster by simulating much fewer photons.
        force: bool
            Force flag will remove existing files and simulate again.
        """
        all_mirrors = self._mirror_numbers if self.config.single_mirror_mode else [0]
        for this_off_axis in self.config.off_axis_angle:
            for this_mirror in all_mirrors:
                self._logger.info(
                    f"Simulating RayTracing for off_axis={this_off_axis}, mirror={this_mirror}"
                )
                simtel = SimtelRunnerRayTracing(
                    simtel_source_path=self._simtel_source_path,
                    telescope_model=self._telescope_model,
                    test=test,
                    config_data={
                        "zenith_angle": self.config.zenith_angle * u.deg,
                        "source_distance": self._source_distance * u.km,
                        "off_axis_angle": this_off_axis * u.deg,
                        "mirror_numbers": this_mirror,
                        "use_random_focal_length": self.config.use_random_focal_length,
                        "single_mirror_mode": self.config.single_mirror_mode,
                    },
                    force_simulate=force,
                )
                simtel.run(test=test, force=force)

                photons_file_name = names.ray_tracing_file_name(
                    self._telescope_model.site,
                    self._telescope_model.name,
                    self._source_distance,
                    self.config.zenith_angle,
                    this_off_axis,
                    this_mirror if self.config.single_mirror_mode else None,
                    self.label,
                    "photons",
                )
                photons_file = self._output_directory.joinpath(photons_file_name)

                self._logger.debug("Using gzip to compress the photons file.")

                with open(photons_file, "rb") as f_in:
                    with gzip.open(
                        photons_file.with_suffix(photons_file.suffix + ".gz"), "wb"
                    ) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                photons_file.unlink()

    def analyze(
        self,
        export=True,
        force=False,
        use_rx=False,
        no_tel_transmission=False,
        containment_fraction=0.8,
    ):
        """
        Analyze RayTracing, meaning read simtel files, compute psfs and eff areas and store the
        results in _results.

        Parameters
        ----------
        export: bool
            If True, results will be exported to a file automatically. Alternatively,
            export_results function can be used.
        force: bool
            If True, existing results files will be removed and analysis will be done again.
        use_rx: bool
            If True, calculations are done using the rx binary provided by sim_telarray. If False,
            calculations are done internally, by the module psf_analysis.
        no_tel_transmission: bool
            If True, the telescope transmission is not applied.
        containment_fraction: float
            Containment fraction for PSF containment calculation. Allowed values are in the
            inverval [0,1]
        """

        do_analyze = not self._file_results.exists() or force

        focal_length = float(self._telescope_model.get_parameter_value("focal_length"))
        tel_transmission_pars = (
            self._telescope_model.get_telescope_transmission_parameters()
            if not no_tel_transmission
            else [1, 0, 0, 0]
        )

        cm_to_deg = 180.0 / pi / focal_length

        self._psf_images = {}
        if do_analyze:
            _rows = []
        else:
            self._read_results()

        all_mirrors = self._mirror_numbers if self.config.single_mirror_mode else [0]
        for this_off_axis in self.config.off_axis_angle:
            for this_mirror in all_mirrors:
                self._logger.debug(f"Analyzing RayTracing for off_axis={this_off_axis}")
                if self.config.single_mirror_mode:
                    self._logger.debug(f"mirror_number={this_mirror}")

                photons_file_name = names.ray_tracing_file_name(
                    self._telescope_model.site,
                    self._telescope_model.name,
                    self._source_distance,
                    self.config.zenith_angle,
                    this_off_axis,
                    this_mirror if self.config.single_mirror_mode else None,
                    self.label,
                    "photons",
                )

                photons_file = self._output_directory.joinpath(photons_file_name + ".gz")
                tel_transmission = compute_telescope_transmission(
                    tel_transmission_pars, this_off_axis
                )
                image = PSFImage(focal_length, None)
                image.read_photon_list_from_simtel_file(photons_file)
                self._psf_images[this_off_axis] = copy(image)

                if not do_analyze:
                    continue

                if use_rx:
                    containment_diameter_cm, centroid_x, centroid_y, eff_area = self._process_rx(
                        photons_file
                    )
                    containment_diameter_deg = containment_diameter_cm * cm_to_deg
                    image.set_psf(containment_diameter_cm, fraction=containment_fraction, unit="cm")
                    image.centroid_x = centroid_x
                    image.centroid_y = centroid_y
                    image.set_effective_area(eff_area * tel_transmission)
                else:
                    containment_diameter_cm = image.get_psf(containment_fraction, "cm")
                    containment_diameter_deg = image.get_psf(containment_fraction, "deg")
                    centroid_x = image.centroid_x
                    centroid_y = image.centroid_y
                    eff_area = image.get_effective_area() * tel_transmission

                eff_flen = (
                    np.nan if this_off_axis == 0 else centroid_x / tan(this_off_axis * pi / 180.0)
                )
                _current_results = (
                    this_off_axis * u.deg,
                    containment_diameter_cm * u.cm,
                    containment_diameter_deg * u.deg,
                    eff_area * u.m * u.m,
                    eff_flen * u.cm,
                )
                if self.config.single_mirror_mode:
                    _current_results += (this_mirror,)
                _rows.append(_current_results)

        if do_analyze:
            _columns = ["Off-axis angle"]
            _columns.extend(list(self.YLABEL.keys()))
            if self.config.single_mirror_mode:
                _columns.append("mirror_number")
            self._results = QTable(rows=_rows, names=_columns)

        self._has_results = True
        if export:
            self.export_results()

    def _process_rx(self, file, containment_fraction=0.8):
        """
        Process sim_telarray photon list with rx binary and return the results
        (containment_diameter_cm, centroids and eff area).

        Parameters
        ----------
        file: str or Path
            Photon list file.
        containment_fraction: float
            Containment fraction for PSF containment calculation. Allowed values are in the
            inverval [0,1]

        Returns
        -------
        (containment_diameter_cm, x_mean, y_mean, eff_area)

        """

        try:
            rx_output = subprocess.Popen(  # pylint: disable=consider-using-with
                shlex.split(
                    f"{self._simtel_source_path}/sim_telarray/bin/rx "
                    f"-f {containment_fraction:.2f} -v"
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            with gzip.open(file, "rb") as _stdin:
                with rx_output.stdin:
                    shutil.copyfileobj(_stdin, rx_output.stdin)
                    try:
                        rx_output = rx_output.communicate()[0].splitlines()[-1:][0].split()
                    except IndexError:
                        self._logger.error(f"Invalid output from rx: {rx_output}")
                        raise
        except FileNotFoundError:
            self._logger.error(f"Photon list file not found: {file}")
            raise
        containment_diameter_cm = 2 * float(rx_output[0])
        x_mean = float(rx_output[1])
        y_mean = float(rx_output[2])
        eff_area = float(rx_output[5])
        return containment_diameter_cm, x_mean, y_mean, eff_area

    def export_results(self):
        """Export results to a csv file."""
        if not self._has_results:
            self._logger.error("Cannot export results because it does not exist")
        else:
            self._logger.info(f"Exporting results to {self._file_results}")
            astropy.io.ascii.write(self._results, self._file_results, format="ecsv", overwrite=True)

    def _read_results(self):
        """Read existing results file and store it in _results."""
        self._results = astropy.io.ascii.read(self._file_results, format="ecsv")
        self._has_results = True

    def plot(self, key, save=False, **kwargs):
        """
        Plot key vs off-axis angle and save the figure in pdf.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen
        save: bool
            If True, figure will be saved.
        **kwargs:
            kwargs for plt.plot

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)

        self._logger.info(f"Plotting {key} vs off-axis angle")

        plot = visualize.plot_table(
            self._results["Off-axis angle", key], self.YLABEL[key], no_legend=True, **kwargs
        )

        if save:
            plot_file_name = names.ray_tracing_plot_file_name(
                key,
                self._telescope_model.site,
                self._telescope_model.name,
                self._source_distance,
                self.config.zenith_angle,
                self.label,
            )
            self._output_directory.joinpath("figures").mkdir(exist_ok=True)
            plot_file = self._output_directory.joinpath("figures").joinpath(plot_file_name)
            self._logger.info(f"Saving fig in {plot_file}")
            plot.savefig(plot_file)

    def plot_histogram(self, key, **kwargs):
        """
        Plot histogram of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen
        **kwargs:
            kwargs for plt.hist

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)

        ax = plt.gca()
        ax.hist([r.value for r in self._results[key]], **kwargs)

    def get_mean(self, key):
        """
        Get mean value of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen

        Returns
        -------
        float
            Mean value of key.

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)
        return np.mean(self._results[key])

    def get_std_dev(self, key):
        """
        Get std dev of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen

        Returns
        -------
        float
            Std deviation of key.

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)
        return np.std(self._results[key])

    def images(self):
        """
        Get list of PSFImages.

        Returns
        -------
        List of PSFImage's
        """
        images = []
        for this_off_axis in self.config.off_axis_angle:
            if self._psf_images and this_off_axis in self._psf_images:
                images.append(self._psf_images[this_off_axis])
        if len(images) == 0:
            self._logger.error("No image found")
            return None
        return images
