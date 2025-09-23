"""Ray tracing simulations and analysis."""

import gzip
import logging
import shutil
from copy import copy
from math import pi, tan
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

from simtools.io import io_handler
from simtools.model.model_utils import compute_telescope_transmission
from simtools.ray_tracing.psf_analysis import PSFImage
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
    site_model: SiteModel
        site model
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
    random_focal_length_seed: int
        Seed for the random number generator used for focal length variation.
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
        site_model,
        simtel_path,
        label=None,
        zenith_angle=20.0 * u.deg,
        off_axis_angle=[0.0] * u.deg,
        source_distance=10.0 * u.km,
        single_mirror_mode=False,
        use_random_focal_length=False,
        random_focal_length_seed=None,
        mirror_numbers="all",
    ):
        """Initialize RayTracing class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing RayTracing class {single_mirror_mode}")

        self.simtel_path = Path(simtel_path)
        self._io_handler = io_handler.IOHandler()

        self.telescope_model, self.site_model = telescope_model, site_model
        self.label = label if label is not None else self.telescope_model.label

        self.zenith_angle = zenith_angle.to("deg").value
        self.off_axis_angle = np.around(off_axis_angle.to("deg").value, 5)
        self.single_mirror_mode = single_mirror_mode
        self.use_random_focal_length = use_random_focal_length
        self.random_focal_length_seed = random_focal_length_seed
        self.mirrors = self._initialize_mirror_configuration(source_distance, mirror_numbers)
        self.output_directory = self._io_handler.get_output_directory(sub_dir="ray_tracing")
        self.output_directory.joinpath("results").mkdir(parents=True, exist_ok=True)
        self._file_results = self.output_directory.joinpath("results").joinpath(
            self._generate_file_name(file_type="ray_tracing", suffix=".ecsv")
        )
        self._psf_images = {}
        self._results = None

    def __repr__(self):
        """Return string representation of RayTracing class."""
        return f"RayTracing(label={self.label})\n"

    def _initialize_mirror_configuration(
        self,
        source_distance,
        mirror_numbers,
    ):
        """
        Initialize mirror configuration.

        Note the difference between single mirror mode and nominal mode.
        In single mirror mode, a 'mirror' is a mirror panel.
        In nominal mode, a 'mirror' is a telescope.

        Parameters
        ----------
        source_distance: astropy.units.Quantity
            Source distance.
        mirror_numbers: list, str
            List of mirror numbers (or 'all').

        Returns
        -------
        dict
            Dictionary containing mirror numbers, focal lengths, and source distances.

        """
        # initialize a mirror as a mirror panel
        if self.single_mirror_mode:
            return self._initialize_single_mirror_mode(mirror_numbers)
        # initialize a mirror as a telescope optical system
        return {
            0: {
                "source_distance": source_distance.to("km").value,
                "focal_length": self.telescope_model.get_parameter_value("focal_length"),
            }
        }

    def _initialize_single_mirror_mode(self, mirror_numbers):
        """
        Initialize single mirror mode.

        Parameters
        ----------
        mirror_numbers: list, str
            List of mirror numbers (or 'all').

        Returns
        -------
        dict
            Dictionary containing mirror numbers, focal lengths, and source distances.
        """
        self._logger.debug(
            "Single mirror mode is activated - "
            "source distance is being recalculated to 2 * focal length "
            " (this is not correct for dual-mirror telescopes)."
        )
        if "all" in mirror_numbers:
            mirror_numbers = list(range(0, self.telescope_model.mirrors.number_of_mirrors))
        mirrors = {mirror: {"focal_length": 0, "source_distance": 0.0} for mirror in mirror_numbers}

        for mirror in mirror_numbers:
            _, _, _, _focal_length, _ = self.telescope_model.mirrors.get_single_mirror_parameters(
                mirror
            )
            if np.isnan(_focal_length) or np.isclose(_focal_length, 0, rtol=1.0e-3):
                _focal_length = self._get_mirror_panel_focal_length() * u.cm
            if np.isnan(_focal_length) or np.isclose(_focal_length, 0, rtol=1.0e-3):
                raise ValueError("Focal length is invalid (NaN or close to zero)")
            mirrors[mirror]["focal_length"] = _focal_length
            mirrors[mirror]["source_distance"] = 2 * _focal_length.to("km").value

        return mirrors

    def _get_mirror_panel_focal_length(self):
        """
        Return mirror panel focal length with possible random addition.

        Returns
        -------
        float
            Focal length.
        """
        _focal_length = self.telescope_model.get_parameter_value("mirror_focal_length")
        if self.use_random_focal_length:
            rng = np.random.default_rng(self.random_focal_length_seed)
            _random_focal_length = self.telescope_model.get_parameter_value("random_focal_length")
            if _random_focal_length[0] > 0.0:
                _focal_length += rng.normal(loc=0, scale=_random_focal_length[0])
            else:
                _focal_length += rng.uniform(
                    low=-1.0 * _random_focal_length[1], high=1.0 * _random_focal_length[1]
                )
        return _focal_length

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
        for this_off_axis in self.off_axis_angle:
            for mirror_number, mirror_data in self.mirrors.items():
                self._logger.info(
                    f"Simulating RayTracing for off_axis={this_off_axis}, mirror={mirror_number}"
                )
                simtel = SimulatorRayTracing(
                    simtel_path=self.simtel_path,
                    telescope_model=self.telescope_model,
                    site_model=self.site_model,
                    test=test,
                    config_data={
                        "zenith_angle": self.zenith_angle,
                        "off_axis_angle": this_off_axis,
                        "source_distance": mirror_data["source_distance"],
                        "single_mirror_mode": self.single_mirror_mode,
                        "use_random_focal_length": self.use_random_focal_length,
                        "mirror_numbers": mirror_number,
                    },
                    force_simulate=force,
                )
                simtel.run(test=test)

                photons_file = self.output_directory.joinpath(
                    self._generate_file_name(
                        file_type="photons",
                        suffix=".lis",
                        off_axis_angle=this_off_axis,
                        mirror_number=mirror_number if self.single_mirror_mode else None,
                    )
                )
                self._logger.debug(f"Using gzip to compress the photons file {photons_file}.")

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

        tel_transmission_pars = self._get_telescope_transmission_params(no_tel_transmission)

        self._psf_images = {}

        _rows = self._process_off_axis_and_mirror(
            tel_transmission_pars,
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
        tel_transmission_pars,
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

        for this_off_axis in self.off_axis_angle:
            for mirror_number, mirror_data in self.mirrors.items():
                self._logger.debug(f"Analyzing RayTracing for off_axis={this_off_axis}")

                photons_file = self.output_directory.joinpath(
                    self._generate_file_name(
                        "photons", ".lis", off_axis_angle=this_off_axis, mirror_number=mirror_number
                    )
                    + ".gz"
                )

                tel_transmission = compute_telescope_transmission(
                    tel_transmission_pars, this_off_axis
                )
                image = self._create_psf_image(
                    photons_file,
                    mirror_data["focal_length"],
                    this_off_axis,
                    containment_fraction,
                    use_rx,
                )

                if do_analyze:
                    _current_results = self._analyze_image(
                        image,
                        this_off_axis,
                        containment_fraction,
                        tel_transmission,
                    )

                    if self.single_mirror_mode:
                        _current_results += (mirror_number,)

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

    def _create_psf_image(
        self, photons_file, focal_length, this_off_axis, containment_fraction, use_rx=False
    ):
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
        containment_fraction: float
            Containment fraction for PSF containment calculation.
        use_rx: bool
            Flag indicating whether to use the RX method for analysis.

        Returns
        -------
        PSFImage
            PSF image object.
        """
        image = PSFImage(
            focal_length=focal_length,
            containment_fraction=containment_fraction,
            simtel_path=self.simtel_path,
        )
        image.process_photon_list(photons_file, use_rx)
        self._psf_images[this_off_axis] = copy(image)
        return image

    def _analyze_image(
        self,
        image,
        this_off_axis,
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
        return (
            this_off_axis * u.deg,
            image.get_psf(containment_fraction, "cm") * u.cm,
            image.get_psf(containment_fraction, "deg") * u.deg,
            image.get_effective_area(tel_transmission) * u.m * u.m,
            np.nan if this_off_axis == 0 else image.centroid_x / tan(this_off_axis * pi / 180.0),
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
        if self.single_mirror_mode:
            _columns.append("mirror_number")
        self._results = QTable(rows=_rows, names=_columns)

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

    def plot(self, key, save=False, d80=None, **kwargs):
        """
        Plot key vs off-axis angle and save the figure in pdf.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen
        save: bool
            If True, figure will be saved.
        d80: float
            d80 for cumulative PSF plot.
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
            raise KeyError(INVALID_KEY_TO_PLOT) from exc

        if save:
            plot_file_name = self._generate_file_name(
                file_type="ray_tracing",
                suffix=".pdf",
                extra_label=key,
            )
            self.output_directory.joinpath("figures").mkdir(exist_ok=True)
            plot_file = self.output_directory.joinpath("figures").joinpath(plot_file_name)
            self._logger.info(f"Saving fig in {plot_file}")
            plot.savefig(plot_file)

            for off_axis_key, image in self._psf_images.items():
                image_file_name = self._generate_file_name(
                    file_type="ray_tracing",
                    off_axis_angle=off_axis_key,
                    suffix=".pdf",
                    extra_label=f"image_{key}",
                )
                image_file = self.output_directory.joinpath("figures").joinpath(image_file_name)
                self._logger.info(f"Saving PSF image to {image_file}")
                image.plot_image(file_name=image_file)

                image_cumulative_file_name = self._generate_file_name(
                    file_type="ray_tracing",
                    off_axis_angle=off_axis_key,
                    suffix=".pdf",
                    extra_label=f"cumulative_psf_{key}",
                )
                image_cumulative_file = self.output_directory.joinpath("figures").joinpath(
                    image_cumulative_file_name
                )
                self._logger.info(f"Saving cumulative PSF to {image_cumulative_file}")
                image.plot_cumulative(file_name=image_cumulative_file, d80=d80)

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
        try:
            ax = plt.gca()
            ax.hist(self._results[key], **kwargs)
        except KeyError as exc:
            raise KeyError(INVALID_KEY_TO_PLOT) from exc

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
        try:
            return np.mean(self._results[key])
        except KeyError as exc:
            raise KeyError(INVALID_KEY_TO_PLOT) from exc

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
        try:
            return np.std(self._results[key])
        except KeyError as exc:
            raise KeyError(INVALID_KEY_TO_PLOT) from exc

    def images(self):
        """
        Get list of PSFImages.

        Returns
        -------
        List of PSFImages
        """
        images = []
        for this_off_axis in self.off_axis_angle:
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
        file_type_prefix = file_type if file_type == "ray_tracing" else f"ray_tracing_{file_type}"
        return names.generate_file_name(
            file_type=file_type_prefix,
            suffix=suffix,
            site=self.telescope_model.site,
            telescope_model_name=self.telescope_model.name,
            source_distance=None if self.single_mirror_mode else self.mirrors[0]["source_distance"],
            zenith_angle=self.zenith_angle,
            off_axis_angle=off_axis_angle,
            mirror_number=mirror_number if self.single_mirror_mode else None,
            label=self.label,
            extra_label=extra_label,
        )
