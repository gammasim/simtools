"""
Ray tracing simulations and analysis.

Simulates light propagation through telescope optics using simtel_array,
processes photon lists to compute PSF (D80 containment diameter),
effective mirror area, and effective focal length as functions of off-axis angle.
"""

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
    label: str
        label used for output file naming.
    zenith_angle: astropy.units.Quantity
        Zenith angle.
    off_axis_angle: list of tuples or astropy.units.Quantity, optional
        Off-axis angles as (x, y) tuples in degrees, or list of scalar angles
        for backward compatibility. If scalar values provided, cardinal offsets
        will be generated in N, S, E, W directions.
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
    offset_file: Path or str, optional
        Path to ECSV file containing x, y offset columns (in degrees).
    offset_directions: list, optional
        Cardinal directions for offset generation: ['N', 'S', 'E', 'W'].
        Only used when generating from scalar offsets.
    """

    YLABEL = {
        "psf_cm": "PSF",
        "psf_deg": "PSF",
        "eff_area": "Eff. mirror area",
        "eff_flen": "Eff. focal length",
    }

    def __init__(
        self,
        telescope_model,
        site_model,
        label=None,
        zenith_angle=20.0 * u.deg,
        off_axis_angle=[0.0] * u.deg,
        source_distance=10.0 * u.km,
        single_mirror_mode=False,
        use_random_focal_length=False,
        random_focal_length_seed=None,
        mirror_numbers="all",
        offset_file=None,
        offset_directions=None,
    ):
        """Initialize RayTracing class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing RayTracing class {single_mirror_mode}")

        self._io_handler = io_handler.IOHandler()

        self.telescope_model, self.site_model = telescope_model, site_model
        self.label = label if label is not None else self.telescope_model.label

        self.zenith_angle = zenith_angle.to("deg").value
        self.elevation_angle = 90.0 - self.zenith_angle

        # Process off-axis angles: convert to list of (x, y) tuples
        if offset_file is not None:
            self.off_axis_angle = self._load_offset_file(offset_file)
            self._logger.info(f"Loaded {len(self.off_axis_angle)} offsets from {offset_file}")
        else:
            self.off_axis_angle = self._process_offset_angles(off_axis_angle, offset_directions)

        self.single_mirror_mode = single_mirror_mode
        self.use_random_focal_length = use_random_focal_length
        self.random_focal_length_seed = random_focal_length_seed
        self.mirrors = self._initialize_mirror_configuration(source_distance, mirror_numbers)
        self.output_directory = self._io_handler.get_output_directory()
        self.output_directory.joinpath("results").mkdir(parents=True, exist_ok=True)
        self._file_results = self.output_directory.joinpath("results").joinpath(
            self._generate_file_name(file_type="ray_tracing", suffix=".ecsv")
        )
        self._psf_images = {}
        self._results = None

    def _process_offset_angles(self, off_axis_angle, offset_directions):
        """
        Process off-axis angles and convert to (x, y) tuples.

        Parameters
        ----------
        off_axis_angle: astropy.units.Quantity
            Scalar angles for backward compatibility.
        offset_directions: list or None
            Cardinal directions ['N', 'S', 'E', 'W'] for offset generation.

        Returns
        -------
        list
            List of (x, y) tuples in degrees.
        """
        # Convert to numpy array of degrees
        angles_deg = np.around(off_axis_angle.to("deg").value, 5)

        if not isinstance(angles_deg, np.ndarray):
            angles_deg = np.atleast_1d(angles_deg)

        # If offsets are already tuples, return as-is
        if len(angles_deg) > 0 and isinstance(angles_deg[0], (tuple, list)):
            return [tuple(float(x) for x in offset) for offset in angles_deg]

        # Default to all four cardinal directions if not specified
        if offset_directions is None:
            offset_directions = ["N", "S", "E", "W"]

        # Generate cardinal direction offsets
        offsets = []
        direction_map = {
            "N": (0.0, 1.0),
            "S": (0.0, -1.0),
            "E": (1.0, 0.0),
            "W": (-1.0, 0.0),
        }

        for angle in angles_deg:
            for direction in offset_directions:
                if direction not in direction_map:
                    self._logger.warning(f"Unknown direction {direction}, skipping")
                    continue
                dx, dy = direction_map[direction]
                offsets.append((dx * angle, dy * angle))

        return offsets

    def _load_offset_file(self, offset_file_path):
        """
        Load offsets from ECSV file.

        Expected columns: x, y (in degrees)

        Parameters
        ----------
        offset_file_path: Path or str
            Path to ECSV file.

        Returns
        -------
        list
            List of (x, y) tuples in degrees.
        """
        offset_file = Path(offset_file_path)
        if not offset_file.exists():
            msg = f"Offset file not found: {offset_file}"
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            table = astropy.io.ascii.read(offset_file, format="ecsv")
            offsets = [(float(row["x"]), float(row["y"])) for row in table]
            self._logger.info(f"Loaded {len(offsets)} offsets from {offset_file}")
            return offsets
        except Exception as e:
            msg = f"Error reading offset file {offset_file}: {e}"
            self._logger.error(msg)
            raise RuntimeError(msg) from e

    def _calculate_phi_from_offset(self, off_x, off_y):
        """
        Calculate telescope azimuth (phi) offset from x, y offsets considering zenith angle.

        Parameters
        ----------
        off_x: float
            Azimuth offset direction component in degrees.
        off_y: float
            Elevation offset direction component in degrees.

        Returns
        -------
        float
            Azimuth offset phi in degrees (change from original phi=0).
        """
        if off_x == 0.0 and off_y == 0.0:
            return 0.0

        zenith_rad = np.radians(self.zenith_angle)
        return off_x * np.cos(zenith_rad)

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
        Run ray tracing simulations using sim_telarray.

        Generates photon lists for each off-axis angle and mirror configuration,
        simulating light propagation through telescope optics.
        Output files are automatically compressed with gzip.

        Parameters
        ----------
        test: bool
            Test flag will make it faster by simulating much fewer photons.
        force: bool
            Force flag will remove existing files and simulate again.
        """
        for off_x, off_y in self.off_axis_angle:
            for mirror_number, mirror_data in self.mirrors.items():
                self._logger.info(
                    f"Simulating RayTracing for off_axis=({off_x:.3f}, {off_y:.3f}), "
                    f"mirror={mirror_number}"
                )

                # Calculate theta (radial distance) and phi (azimuth)
                theta_offset = np.sqrt(off_x**2 + off_y**2)
                phi = self._calculate_phi_from_offset(off_x, off_y)

                simtel = SimulatorRayTracing(
                    telescope_model=self.telescope_model,
                    site_model=self.site_model,
                    label=self.label,
                    test=test,
                    config_data={
                        "zenith_angle": self.zenith_angle,
                        "off_axis_x": off_x,
                        "off_axis_y": off_y,
                        "off_axis_theta": theta_offset,
                        "off_axis_phi": phi,
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
                        off_axis_x=off_x,
                        off_axis_y=off_y,
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
        Analyze ray tracing simulation results.

        Processes photon lists to compute PSF containment diameters (D80 by default),
        effective mirror area (detected_photons * total_area / total_photons),
        and effective focal length (centroid_radius / tan(off_axis_angle)).

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

        for off_x, off_y in self.off_axis_angle:
            for mirror_number, mirror_data in self.mirrors.items():
                self._logger.debug(f"Analyzing RayTracing for off_axis=({off_x:.3f}, {off_y:.3f})")

                photons_file = self.output_directory.joinpath(
                    self._generate_file_name(
                        "photons",
                        ".lis",
                        off_axis_x=off_x,
                        off_axis_y=off_y,
                        mirror_number=mirror_number,
                    )
                    + ".gz"
                )

                # Calculate theta and phi for transmission calculation
                theta_offset = np.sqrt(off_x**2 + off_y**2)

                tel_transmission = compute_telescope_transmission(
                    tel_transmission_pars, theta_offset
                )
                image = self._create_psf_image(
                    photons_file,
                    mirror_data["focal_length"],
                    containment_fraction,
                    use_rx,
                )

                # Store PSF image with (x, y) tuple key
                self._psf_images[(off_x, off_y)] = copy(image)

                if do_analyze:
                    _current_results = self._analyze_image(
                        image,
                        off_x,
                        off_y,
                        theta_offset,
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

    def _create_psf_image(self, photons_file, focal_length, containment_fraction, use_rx=False):
        """
        Create PSF image from photons file.

        Parameters
        ----------
        photons_file: Path
            Path to the photons file.
        focal_length: float
            Focal length of the telescope.
        theta_offset: float
            Radial off-axis angle in degrees.
        containment_fraction: float
            Containment fraction for PSF containment calculation.
        use_rx: bool
            Flag indicating whether to use the RX method for analysis.

        Returns
        -------
        PSFImage
            PSF image object.
        """
        image = PSFImage(focal_length=focal_length, containment_fraction=containment_fraction)
        image.process_photon_list(photons_file, use_rx)
        # Note: storage key is set in _process_off_axis_and_mirror with (x, y) tuple
        return image

    def _analyze_image(
        self,
        image,
        off_x,
        off_y,
        theta_offset,
        containment_fraction,
        tel_transmission,
    ):
        """
        Extract analysis results from PSF image.

        Computes effective focal length as f_eff = r / tan(off_axis_angle),
        where r is the distance from the image centroid to the optical axis.

        Parameters
        ----------
        image: PSFImage
            PSF image object.
        off_x: float
            X offset in degrees.
        off_y: float
            Y offset in degrees.
        theta_offset: float
            Radial offset (radial distance) in degrees.
        containment_fraction: float
            Containment fraction for PSF containment calculation.
        tel_transmission: float
            Telescope transmission factor.

        Returns
        -------
        tuple
            (off_x, off_y, theta_offset, azimuth, psf_cm, psf_deg, eff_area, eff_focal_length)
        """
        r = np.hypot(image.centroid_x, image.centroid_y)
        azimuth = self._calculate_phi_from_offset(off_x, off_y)
        return (
            off_x * u.deg,
            off_y * u.deg,
            theta_offset * u.deg,
            azimuth * u.deg,
            image.get_psf(containment_fraction, "cm") * u.cm,
            image.get_psf(containment_fraction, "deg") * u.deg,
            image.get_effective_area(tel_transmission) * u.m * u.m,
            np.nan if theta_offset == 0 else r / tan(theta_offset * pi / 180.0),
        )

    def _store_results(self, _rows):
        """
        Store analysis results.

        Parameters
        ----------
        _rows: list
            List of rows containing analysis results.
        """
        _columns = ["off_x", "off_y", "off_theta", "off_azimuth"]
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

    def get_psf_mm(self, row_index: int = 0) -> float:
        """Return PSF diameter from the analysis results in mm.

        Parameters
        ----------
        row_index : int
            Row index into the results table (default: 0).

        Returns
        -------
        float
            PSF diameter in millimeters.
        """
        if self._results is None:
            raise RuntimeError("No results available; run analyze() first")
        psf = self._results["psf_cm"][row_index]

        if isinstance(psf, u.Quantity):
            psf_cm = psf.to_value(u.cm)
        else:
            psf_cm = float(psf)

        return psf_cm * 10.0

    def _read_results(self):
        """Read existing results file and store it in _results."""
        self._results = astropy.io.ascii.read(self._file_results, format="ecsv")

    def plot(self, key, save=False, psf_diameter_cm=None, **kwargs):
        """
        Plot analysis results vs radial off-axis angle.

        Visualizes computed PSF, effective area, or effective focal length
        as a function of radial off-axis angle. Optionally saves individual PSF images
        and cumulative distributions.

        Parameters
        ----------
        key: str
            psf_cm, psf_deg, eff_area or eff_flen
        save: bool
            If True, figure will be saved.
        psf_diameter_cm: float
            PSF diameter value to be marked in the cumulative PSF plot (in cm).
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
                self._results["off_theta", key], self.YLABEL[key], no_legend=True, **kwargs
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

            for (off_x, off_y), image in self._psf_images.items():
                image_file_name = self._generate_file_name(
                    file_type="ray_tracing",
                    off_axis_x=off_x,
                    off_axis_y=off_y,
                    suffix=".pdf",
                    extra_label=f"image_{key}",
                )
                image_file = self.output_directory.joinpath("figures").joinpath(image_file_name)
                self._logger.info(f"Saving PSF image to {image_file}")
                image.plot_image(file_name=image_file)

                image_cumulative_file_name = self._generate_file_name(
                    file_type="ray_tracing",
                    off_axis_x=off_x,
                    off_axis_y=off_y,
                    suffix=".pdf",
                    extra_label=f"cumulative_psf_{key}",
                )
                image_cumulative_file = self.output_directory.joinpath("figures").joinpath(
                    image_cumulative_file_name
                )
                self._logger.info(f"Saving cumulative PSF to {image_cumulative_file}")
                image.plot_cumulative(
                    file_name=image_cumulative_file, psf_diameter_cm=psf_diameter_cm
                )

    def plot_histogram(self, key, **kwargs):
        """
        Plot histogram of key.

        Parameters
        ----------
        key: str
            psf_cm, psf_deg, eff_area or eff_flen
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
            psf_cm, psf_deg, eff_area or eff_flen

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
            psf_cm, psf_deg, eff_area or eff_flen

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
        Get list of analyzed PSF images.

        Returns PSFImage objects containing photon positions, centroids,
        PSF containment diameters, and effective areas for each off-axis point.

        Returns
        -------
        List of PSFImages
        """
        images = [
            self._psf_images[(off_x, off_y)]
            for off_x, off_y in self.off_axis_angle
            if self._psf_images and (off_x, off_y) in self._psf_images
        ]
        if len(images) == 0:
            self._logger.warning("No image found")
            return None
        return images

    def _generate_file_name(
        self,
        file_type,
        suffix,
        off_axis_x=None,
        off_axis_y=None,
        mirror_number=None,
        extra_label=None,
    ):
        """Generate file name for output files with (x, y) offsets."""
        file_type_prefix = file_type if file_type == "ray_tracing" else f"ray_tracing_{file_type}"
        return names.generate_file_name(
            file_type=file_type_prefix,
            suffix=suffix,
            site=self.telescope_model.site,
            telescope_model_name=self.telescope_model.name,
            source_distance=None if self.single_mirror_mode else self.mirrors[0]["source_distance"],
            zenith_angle=self.zenith_angle,
            off_axis_x=off_axis_x,
            off_axis_y=off_axis_y,
            mirror_number=mirror_number if self.single_mirror_mode else None,
            label=self.label,
            extra_label=extra_label,
        )
