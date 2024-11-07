"""Ray tracing simulations and analysis."""

import gzip
import logging
import shlex
import shutil
import subprocess
from collections import namedtuple
from copy import copy
from math import pi, tan
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

from simtools.io_operations import io_handler
from simtools.model.model_utils import compute_telescope_transmission
from simtools.psf_analysis import PSFImage
from simtools.simtel.simulator_ray_tracing import SimulatorRayTracing
from simtools.utils import names
from simtools.visualization import visualize

__all__ = ["RayTracing"]

INVALID_KEY_TO_PLOT = "Invalid key to plot"


class RayTracing:
    """
    Ray tracing simulations and analysis.

    Parameters
    ----------
    telescope_model: TelescopeModel
        telescope model
    simtel_path: str (or Path)
        Location of sim_telarray installation.
    label: str
        label used for output file naming.
    zenith_angle: astropy.units.Quantity
        Zenith angle.
    off_axis_angle: list of astropy.units.Quantity
        Off-axis angles.
    source_distance: astropy.units.Quantity
        Source distance.
    single_mirror_mode: bool
        Single mirror mode flag.
    use_random_focal_length: bool
        Use random focal length flag.
    mirror_numbers: list, str
        List of mirror numbers (or 'all').
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
        simtel_path,
        label=None,
        zenith_angle=20.0 * u.deg,
        off_axis_angle=[0.0] * u.deg,
        source_distance=10.0 * u.km,
        single_mirror_mode=False,
        use_random_focal_length=False,
        mirror_numbers="all",
    ):
        """Initialize RayTracing class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initializing RayTracing class")

        self.simtel_path = Path(simtel_path)
        self._io_handler = io_handler.IOHandler()

        self.telescope_model = telescope_model
        self.label = label if label is not None else self.telescope_model.label

        self.config = self._initialize_config(
            zenith_angle,
            off_axis_angle,
            source_distance,
            single_mirror_mode,
            use_random_focal_length,
            mirror_numbers,
        )

        self.output_directory = self._io_handler.get_output_directory(
            label=self.label, sub_dir="ray-tracing"
        )
        self.output_directory.joinpath("results").mkdir(parents=True, exist_ok=True)
        self._file_results = self.output_directory.joinpath("results").joinpath(
            self._generate_file_name(file_type="ray-tracing", suffix=".ecsv")
        )
        self._psf_images = None
        self._results = None

    def __repr__(self):
        """Return string representation of RayTracing class."""
        return f"RayTracing(label={self.label})\n"

    def _initialize_config(
        self,
        zenith_angle,
        off_axis_angle,
        source_distance,
        single_mirror_mode,
        use_random_focal_length,
        mirror_numbers,
    ):
        """Initialize ray tracing configuration to namedtuple."""
        config_data = namedtuple(
            "Config",
            [
                "zenith_angle",
                "off_axis_angle",
                "source_distance",
                "single_mirror_mode",
                "use_random_focal_length",
                "mirror_numbers",
            ],
        )
        if single_mirror_mode:
            source_distance, mirror_numbers = self._initialize_single_mirror_mode(mirror_numbers)
        # round the off-axis angles so the values in results table are the same as provided.
        off_axis_angle = np.around(off_axis_angle, 5)

        return config_data(
            zenith_angle=zenith_angle.to("deg").value,
            off_axis_angle=off_axis_angle.to("deg").value,
            source_distance=source_distance.to("km").value,
            single_mirror_mode=single_mirror_mode,
            use_random_focal_length=use_random_focal_length,
            mirror_numbers=mirror_numbers,
        )

    def _initialize_single_mirror_mode(self, mirror_numbers):
        """
        Initialize single mirror mode.

        Parameters
        ----------
        mirror_numbers: list, str
            List of mirror numbers (or 'all').

        Returns
        -------
        tuple
            Tuple containing source distance and mirror numbers.
        """
        self._logger.debug(
            "Single mirror mode is activated - "
            "source distance is being recalculated to 2 * focal length "
            " (this is not correct for dual-mirror telescopes)."
        )
        mir_focal_length = self.telescope_model.get_parameter_value("mirror_focal_length")
        source_distance = (2 * float(mir_focal_length) * u.cm).to(u.km)

        if "all" in mirror_numbers:
            mirror_numbers = list(range(0, self.telescope_model.mirrors.number_of_mirrors))

        return source_distance, mirror_numbers

    def simulate(self, test=False, force=False):
        """
        Simulate RayTracing using SimulatorRayTracing.

        Parameters
        ----------
        test: bool
            Test flag will make it faster by simulating much fewer photons.
        force: bool
            Force flag will remove existing files and simulate again.
        """
        all_mirrors = self.config.mirror_numbers if self.config.single_mirror_mode else [0]
        for this_off_axis in self.config.off_axis_angle:
            for this_mirror in all_mirrors:
                self._logger.info(
                    f"Simulating RayTracing for off_axis={this_off_axis}, mirror={this_mirror}"
                )
                simtel = SimulatorRayTracing(
                    simtel_path=self.simtel_path,
                    telescope_model=self.telescope_model,
                    test=test,
                    config_data=self.config._replace(
                        off_axis_angle=this_off_axis,
                        mirror_numbers=this_mirror,
                    ),
                    force_simulate=force,
                )
                simtel.run(test=test)

                photons_file = self.output_directory.joinpath(
                    self._generate_file_name(
                        file_type="photons",
                        suffix=".lis",
                        off_axis_angle=this_off_axis,
                        mirror_number=this_mirror if self.config.single_mirror_mode else None,
                    )
                )
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
        Ray tracing analysis.

        Involves the following: read simtel files, compute PSFs and eff areas, store the
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
            interval [0,1]
        """
        do_analyze = not self._file_results.exists() or force
        if not do_analyze:
            self._read_results()

        focal_length = float(self.telescope_model.get_parameter_value("focal_length"))
        tel_transmission_pars = self._get_telescope_transmission_params(no_tel_transmission)
        cm_to_deg = 180.0 / pi / focal_length

        self._psf_images = {}

        # Call the helper function to process off-axis angles and mirrors
        _rows = self._process_off_axis_and_mirror(
            self._get_all_mirrors(),
            focal_length,
            tel_transmission_pars,
            cm_to_deg,
            do_analyze,
            use_rx,
            containment_fraction,
        )

        if do_analyze:
            self._store_results(_rows)
        if export:
            self.export_results()

    def _process_off_axis_and_mirror(
        self,
        all_mirrors,
        focal_length,
        tel_transmission_pars,
        cm_to_deg,
        do_analyze,
        use_rx,
        containment_fraction,
    ):
        """
        Process off-axis angles and mirrors for RayTracing analysis.

        Parameters
        ----------
        all_mirrors: list
            List of mirror numbers to analyze.
        focal_length: float
            Focal length of the telescope.
        tel_transmission_pars: list
            Telescope transmission parameters.
        cm_to_deg: float
            Conversion factor from centimeters to degrees.
        do_analyze: bool
            Flag indicating whether to perform analysis or not.
        use_rx: bool
            Flag indicating whether to use the RX method for analysis.
        containment_fraction: float
            Containment fraction for PSF containment calculation.

        Returns
        -------
        list
            List of results for each combination of off-axis angle and mirror.
        """
        _rows = []

        for this_off_axis in self.config.off_axis_angle:
            for this_mirror in all_mirrors:
                self._logger.debug(f"Analyzing RayTracing for off_axis={this_off_axis}")

                photons_file = self.output_directory.joinpath(
                    self._generate_file_name(
                        "photons", ".lis", off_axis_angle=this_off_axis, mirror_number=this_mirror
                    )
                    + ".gz"
                )

                tel_transmission = compute_telescope_transmission(
                    tel_transmission_pars, this_off_axis
                )
                image = self._create_psf_image(photons_file, focal_length, this_off_axis)

                if do_analyze:
                    _current_results = self._analyze_image(
                        image,
                        photons_file,
                        this_off_axis,
                        use_rx,
                        cm_to_deg,
                        containment_fraction,
                        tel_transmission,
                    )

                    if self.config.single_mirror_mode:
                        _current_results += (this_mirror,)

                    _rows.append(_current_results)

        return _rows

    def _get_telescope_transmission_params(self, no_tel_transmission):
        """
        Get telescope transmission parameters.

        Parameters
        ----------
        no_tel_transmission: bool
            Flag indicating whether to apply telescope transmission or not.

        Returns
        -------
        list
            Telescope transmission parameters.
        """
        return (
            self.telescope_model.get_parameter_value("telescope_transmission")
            if not no_tel_transmission
            else [1, 0, 0, 0]
        )

    def _get_all_mirrors(self):
        """
        Get number of mirrors.

        Returns
        -------
        list
            List of number of mirrors.
        """
        return self.config.mirror_numbers if self.config.single_mirror_mode else [0]

    def _get_photons_file(self, this_off_axis, this_mirror):
        """
        Get path to photons file for a given off-axis angle and mirror.

        Parameters
        ----------
        this_off_axis: float
            Off-axis angle.
        this_mirror: int or None
            Mirror number.

        Returns
        -------
        Path
            Path to the photons file.
        """
        photons_file_name = names.generate_file_name(
            file_type="photons",
            suffix=".lis",
            site=self.telescope_model.site,
            telescope_model_name=self.telescope_model.name,
            source_distance=self.config.source_distance,
            zenith_angle=self.config.zenith_angle,
            off_axis_angle=this_off_axis,
            mirror_number=this_mirror if self.config.single_mirror_mode else None,
            label=self.label,
        )
        return self.output_directory.joinpath(photons_file_name + ".gz")

    def _create_psf_image(self, photons_file, focal_length, this_off_axis):
        """
        Create PSF image from photons file.

        Parameters
        ----------
        photons_file: Path
            Path to the photons file.
        focal_length: float
            Focal length of the telescope.
        this_off_axis: float
            Off-axis angle.

        Returns
        -------
        PSFImage
            PSF image object.
        """
        image = PSFImage(focal_length, None)
        image.read_photon_list_from_simtel_file(photons_file)
        self._psf_images[this_off_axis] = copy(image)
        return image

    def _analyze_image(
        self,
        image,
        photons_file,
        this_off_axis,
        use_rx,
        cm_to_deg,
        containment_fraction,
        tel_transmission,
    ):
        """
        Analyze PSF image.

        Parameters
        ----------
        image: PSFImage
            PSF image object.
        photons_file: Path
            Path to the photons file.
        this_off_axis: float
            Off-axis angle.
        use_rx: bool
            Flag indicating whether to use the RX method for analysis.
        cm_to_deg: float
            Conversion factor from centimeters to degrees.
        containment_fraction: float
            Containment fraction for PSF containment calculation.
        tel_transmission: float
            Telescope transmission factor.

        Returns
        -------
        tuple
            Tuple containing analyzed results.
        """
        if use_rx:
            containment_diameter_cm, centroid_x, centroid_y, eff_area = self._process_rx(
                photons_file
            )
            containment_diameter_deg = containment_diameter_cm * cm_to_deg
            image.set_psf(containment_diameter_cm, fraction=containment_fraction, unit="cm")
            image.centroid_x = centroid_x
            image.centroid_y = centroid_y
            eff_area = eff_area * tel_transmission
            image.set_effective_area(eff_area)
        else:
            containment_diameter_cm = image.get_psf(containment_fraction, "cm")
            containment_diameter_deg = image.get_psf(containment_fraction, "deg")
            centroid_x = image.centroid_x
            eff_area = image.get_effective_area() * tel_transmission

        eff_flen = np.nan if this_off_axis == 0 else centroid_x / tan(this_off_axis * pi / 180.0)
        return (
            this_off_axis * u.deg,
            containment_diameter_cm * u.cm,
            containment_diameter_deg * u.deg,
            eff_area * u.m * u.m,
            eff_flen * u.cm,
        )

    def _store_results(self, _rows):
        """
        Store analysis results.

        Parameters
        ----------
        _rows: list
            List of rows containing analysis results.
        """
        _columns = ["Off-axis angle"]
        _columns.extend(list(self.YLABEL.keys()))
        if self.config.single_mirror_mode:
            _columns.append("mirror_number")
        self._results = QTable(rows=_rows, names=_columns)

    def _process_rx(self, file, containment_fraction=0.8):
        """
        Process sim_telarray photon list with rx binary and return results.

        Parameters
        ----------
        file: str or Path
            Photon list file.
        containment_fraction: float
            Containment fraction for PSF containment calculation. Allowed values are in the
            interval [0,1]

        Returns
        -------
        (containment_diameter_cm, x_mean, y_mean, eff_area)

        """
        try:
            rx_output = subprocess.Popen(  # pylint: disable=consider-using-with
                shlex.split(
                    f"{self.simtel_path}/sim_telarray/bin/rx -f {containment_fraction:.2f} -v"
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            with gzip.open(file, "rb") as _stdin:
                with rx_output.stdin:
                    shutil.copyfileobj(_stdin, rx_output.stdin)
                    try:
                        rx_output = rx_output.communicate()[0].splitlines()[-1:][0].split()
                    except IndexError as e:
                        raise IndexError(f"Unexpected output format from rx: {rx_output}") from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Photon list file not found: {file}") from e
        containment_diameter_cm = 2 * float(rx_output[0])
        x_mean = float(rx_output[1])
        y_mean = float(rx_output[2])
        eff_area = float(rx_output[5])
        return containment_diameter_cm, x_mean, y_mean, eff_area

    def export_results(self):
        """Export results to a csv file."""
        if self._results:
            self._logger.info(f"Exporting results to {self._file_results}")
            astropy.io.ascii.write(self._results, self._file_results, format="ecsv", overwrite=True)
        else:
            self._logger.error("No results to export")

    def _read_results(self):
        """Read existing results file and store it in _results."""
        self._results = astropy.io.ascii.read(self._file_results, format="ecsv")

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
        self._logger.info(f"Plotting {key} vs off-axis angle")

        try:
            plot = visualize.plot_table(
                self._results["Off-axis angle", key], self.YLABEL[key], no_legend=True, **kwargs
            )
        except KeyError as exc:
            msg = INVALID_KEY_TO_PLOT
            self._logger.error(msg)
            raise exc

        if save:
            plot_file_name = self._generate_file_name(
                file_type="ray-tracing",
                suffix=".pdf",
                extra_label=key,
            )
            self.output_directory.joinpath("figures").mkdir(exist_ok=True)
            plot_file = self.output_directory.joinpath("figures").joinpath(plot_file_name)
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
            msg = INVALID_KEY_TO_PLOT
            self._logger.error(msg)
            raise KeyError(msg)

        ax = plt.gca()
        ax.hist(self._results[key], **kwargs)

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
            msg = INVALID_KEY_TO_PLOT
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
            msg = INVALID_KEY_TO_PLOT
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
            self._logger.warning("No image found")
            return None
        return images

    def _generate_file_name(
        self, file_type, suffix, off_axis_angle=None, mirror_number=None, extra_label=None
    ):
        """Generate file name for output files."""
        return names.generate_file_name(
            file_type=file_type,
            suffix=suffix,
            site=self.telescope_model.site,
            telescope_model_name=self.telescope_model.name,
            source_distance=self.config.source_distance,
            zenith_angle=self.config.zenith_angle,
            off_axis_angle=off_axis_angle,
            mirror_number=mirror_number if self.config.single_mirror_mode else None,
            label=self.label,
            extra_label=extra_label,
        )
