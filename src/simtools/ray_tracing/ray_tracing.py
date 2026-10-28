"""
Ray tracing simulations and analysis.

Traces star light through the telescope optics with the configured ray-tracing
backend (``simtools.ray_tracing.backends.interface``) and computes the PSF (D80
containment diameter), effective mirror area, and effective focal length as
functions of the off-axis angle.
"""

import logging
from copy import copy
from math import pi, tan
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import numpy as np
from astropy.table import QTable

from simtools import constants, settings
from simtools.io import io_handler
from simtools.model.model_utils import compute_telescope_transmission
from simtools.ray_tracing import parameter_changes
from simtools.ray_tracing.backends import interface
from simtools.ray_tracing.backends.interface import StarImageRequest
from simtools.simtel import simtel_seeds
from simtools.utils import names, random
from simtools.visualization import visualize

INVALID_KEY_TO_PLOT = "Invalid key to plot"
DEFAULT_MAX_WORKERS = 8
BACKEND_METADATA = "ray_tracing_backend"


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

        self._io_handler = io_handler.IOHandler()

        self.telescope_model, self.site_model = telescope_model, site_model
        self.label = label if label is not None else self.telescope_model.label

        self.zenith_angle = zenith_angle.to("deg").value
        self.azimuth_angle = 0.0 * u.deg
        self._logger.debug(
            f"Telescope pointing set to zenith={self.zenith_angle:.3f} deg, "
            f"azimuth={self.azimuth_angle:.3f} deg"
        )

        # Process off-axis angles: convert to list of (x, y) tuples
        if offset_file is not None:
            self.off_axis_angle = self._load_offset_file(offset_file)
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
        self.psf_images = {}
        self._backend = None
        self._results = None

    def _process_offset_angles(self, off_axis_angle, offset_directions):
        """
        Process off-axis angles and convert to (x, y) tuples.

        Parameters
        ----------
        off_axis_angle: astropy.units.Quantity or list
            Scalar angles for backward compatibility, or list of (x, y) tuples.
        offset_directions: list or None
            Cardinal directions ['N', 'S', 'E', 'W'] for offset generation.

        Returns
        -------
        list
            List of (x, y) tuples in degrees.
        """
        # Check if input is already a list of tuples
        if self._is_list_of_tuples(off_axis_angle):
            return self._convert_tuples_to_degrees(off_axis_angle)

        # Convert scalar angles to numpy array
        angles_deg = self._extract_scalar_angles(off_axis_angle)

        # Generate offsets from scalar angles
        if offset_directions is None:
            offset_directions = ["N", "S", "E", "W"]

        return self._generate_offsets_from_angles(angles_deg, offset_directions)

    def _is_list_of_tuples(self, off_axis_angle):
        """
        Check if input is a list of tuples or lists.

        Parameters
        ----------
        off_axis_angle: any
            Input to check.

        Returns
        -------
        bool
            True if input is a list of tuples/lists.
        """
        if not isinstance(off_axis_angle, (list, tuple)):
            return False
        return len(off_axis_angle) > 0 and isinstance(off_axis_angle[0], (tuple, list))

    def _convert_tuples_to_degrees(self, offset_tuples):
        """
        Convert tuple offsets to degrees, handling units.

        Parameters
        ----------
        offset_tuples: list
            List of (x, y) tuples, possibly with units.

        Returns
        -------
        list
            List of (x, y) tuples in degrees.
        """
        result = []
        for offset in offset_tuples:
            converted_offset = tuple(
                x.to("deg").value if isinstance(x, u.Quantity) else float(x) for x in offset
            )
            result.append(converted_offset)
        return result

    def _extract_scalar_angles(self, off_axis_angle):
        """
        Extract scalar angles and convert to degrees.

        Parameters
        ----------
        off_axis_angle: astropy.units.Quantity or list
            Input angles.

        Returns
        -------
        numpy.ndarray
            Angles in degrees.
        """
        if isinstance(off_axis_angle, (list, tuple)):
            angles_deg = np.asarray(off_axis_angle)
        else:
            angles_deg = off_axis_angle.to("deg").value
        return np.atleast_1d(np.around(angles_deg, 5))

    def _generate_offsets_from_angles(self, angles_deg, offset_directions):
        """
        Generate (x, y) offsets for given angles and directions.

        Parameters
        ----------
        angles_deg: numpy.ndarray
            Angles in degrees.
        offset_directions: list
            Cardinal directions.

        Returns
        -------
        list
            List of (x, y) tuples.
        """
        direction_map = {
            "N": (0.0, 1.0),
            "S": (0.0, -1.0),
            "E": (1.0, 0.0),
            "W": (-1.0, 0.0),
        }

        offsets = []
        for angle in angles_deg:
            # avoid generating multiple identical offsets at zero angle
            if np.isclose(angle, 0.0, rtol=0.0, atol=1e-12):
                offsets.append((0.0, 0.0))
                continue
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
            raise FileNotFoundError(f"Offset file not found: {offset_file}")
        table = astropy.io.ascii.read(offset_file, format="ecsv")

        try:
            offsets = [(float(row["x"]), float(row["y"])) for row in table]
        except KeyError as e:
            raise RuntimeError(f"Error reading offset file {offset_file}") from e
        self._logger.info(f"Loaded {len(offsets)} offsets from {offset_file}")
        return offsets

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

    def simulate(
        self, test=False, force=False, compress_photons=True, max_workers=DEFAULT_MAX_WORKERS
    ):
        """
        Run simulations using the configured ray-tracing backend.

        Traces one image for each off-axis angle (and mirror in single-mirror mode).

        Parameters
        ----------
        test: bool
            Test flag will make it faster by simulating much fewer photons.
        force: bool
            Simulate again even if the backend keeps earlier results.
        compress_photons: bool
            Compress photon files the backend writes.
        max_workers: int
            Maximum number of concurrent simulations of the backend. Defaults to 8.

        Raises
        ------
        ValueError
            If ``max_workers`` is not positive.

        """
        if max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self._get_backend().simulate(
            self._requests(test),
            test=test,
            force=force,
            compress_photons=compress_photons,
            max_workers=max_workers,
        )

    def _get_backend(self):
        """Return the ray-tracing backend (created on first use)."""
        if self._backend is None:
            self._backend = interface.get_backend()(
                self.telescope_model,
                self.site_model,
                self.label,
                self.zenith_angle,
                self.single_mirror_mode,
            )
        return self._backend

    def _requests(self, test=False, for_simulation=True):
        """Star images to trace: one per off-axis angle (and mirror in single-mirror mode).

        Seeds and parameter changes are only needed to simulate; the analysis
        identifies the images.
        """
        photons = self._photons_per_instance(test)
        seeds = ()
        if for_simulation:
            seeds = self._instance_seeds(single_instance=test)
        return [
            self._request(off_x, off_y, mirror_number, photons or 100000, seeds, for_simulation)
            for off_x, off_y in self.off_axis_angle
            for mirror_number in self.mirrors
        ]

    @staticmethod
    def _photons_per_instance(test):
        """Photons per instance: 5000 in test mode, else ``number_of_photons`` (or None)."""
        if test:
            return 5000
        photons = settings.config.args.get("number_of_photons")
        if photons is not None and (
            isinstance(photons, bool) or not isinstance(photons, int) or photons < 1
        ):
            raise ValueError("number_of_photons must be a positive integer")
        return photons

    def _request(self, off_x, off_y, mirror_number, photons, seeds, for_simulation):
        """Return the request of one image."""
        source_distance = self.mirrors[mirror_number]["source_distance"]
        changes = {}
        if self.single_mirror_mode and for_simulation:
            changes = parameter_changes.single_mirror(
                self.telescope_model.get_single_mirror_list_file(
                    mirror_number, self.use_random_focal_length
                ),
                self.telescope_model.get_parameter_value("mirror_focal_length"),
                source_distance * u.km.to(u.cm),
            )
        return StarImageRequest(
            off_axis_x=off_x,
            off_axis_y=off_y,
            source_distance_km=source_distance,
            seeds=tuple(self._mirror_seeds(seeds, mirror_number)),
            photons_per_instance=photons,
            mirror_number=mirror_number if self.single_mirror_mode else None,
            parameter_changes=changes,
        )

    def _instance_seeds(self, single_instance=False):
        """(Instrument seed, simulation seed) of each random instrument instance.

        A single instance in single-mirror mode or if ``single_instance`` is set (test
        mode); otherwise ``random_instrument_instances``.
        Both backends run one simulation per instance with these seeds.
        """
        instances = settings.config.args.get(
            "random_instrument_instances", constants.RAY_TRACING_INSTRUMENT_INSTANCES
        )
        if single_instance or self.single_mirror_mode:
            instances = 1
        return simtel_seeds.instance_seed_pairs(instances)

    def _mirror_seeds(self, seeds, mirror_number):
        """Seeds of one mirror in single-mirror mode (each mirror has its own errors)."""
        if not self.single_mirror_mode:
            return seeds
        return [
            tuple(
                random.seeds(
                    fixed_seed=(seed, mirror_number + 1), max_seed=constants.SIMTEL_MAX_SEED
                )
                for seed in pair
            )
            for pair in seeds
        ]

    def analyze(
        self,
        export=True,
        force=False,
        use_rx=False,
        no_tel_transmission=False,
        containment_fraction=0.8,
        save_photons=False,
        keep_images=True,
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
        save_photons: bool
            If True, keep the photons of the images (as files of the backend); by default,
            photon files are removed after the analysis.
        keep_images: bool
            If True, keep the PSF image of each off-axis angle (see ``images``). Images hold
            all photon positions; without them only the results table is kept.
        """
        do_analyze = not self._file_results.exists() or force
        if not do_analyze:
            self._read_results()
            do_analyze = (
                self._results.meta.get(BACKEND_METADATA) != settings.config.ray_tracing_backend
            )

        tel_transmission_pars = self._get_telescope_transmission_params(no_tel_transmission)

        self.psf_images = {}

        _rows = self._process_off_axis_and_mirror(
            tel_transmission_pars,
            do_analyze,
            use_rx,
            containment_fraction,
            save_photons,
            keep_images,
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
        save_photons,
        keep_images=True,
    ):
        """
        Analyze the image of each off-axis angle and mirror.

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
        save_photons: bool
            If True, keep the photons of the images (backend-specific files).
        keep_images: bool
            If True, keep the PSF image of each off-axis angle.

        Returns
        -------
        list
            List of results for each combination of off-axis angle and mirror.
        """
        _rows = []
        for request in self._requests(for_simulation=False):
            mirror_number = 0 if request.mirror_number is None else request.mirror_number
            self._logger.debug(
                f"Analyzing RayTracing for off_axis=({request.off_axis_x:.3f}, "
                f"{request.off_axis_y:.3f})"
            )
            image, traced_structure = self._get_backend().psf_image(
                request,
                self.mirrors[mirror_number]["focal_length"],
                containment_fraction,
                use_rx=use_rx,
                save_photons=save_photons,
            )
            theta_offset = np.sqrt(request.off_axis_x**2 + request.off_axis_y**2)
            # A traced telescope structure replaces telescope_transmission.
            tel_transmission = (
                1.0
                if traced_structure
                else compute_telescope_transmission(tel_transmission_pars, theta_offset)
            )
            if keep_images:
                self.psf_images[(request.off_axis_x, request.off_axis_y)] = copy(image)
            current_result = self._analyze_mirror_image(
                image,
                request.off_axis_x,
                request.off_axis_y,
                mirror_number,
                theta_offset,
                containment_fraction,
                tel_transmission,
                do_analyze,
            )
            if current_result is not None:
                _rows.append(current_result)
        return _rows

    def _analyze_mirror_image(
        self,
        image,
        off_x,
        off_y,
        mirror_number,
        theta_offset,
        containment_fraction,
        tel_transmission,
        do_analyze,
    ):
        """Return analysis results for an image when analysis was requested."""
        if not do_analyze:
            return None
        current_results = self._analyze_image(
            image,
            off_x,
            off_y,
            theta_offset,
            containment_fraction,
            tel_transmission,
        )
        if self.single_mirror_mode:
            return (*current_results, mirror_number)
        return current_results

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
            (
                off_x,
                off_y,
                theta_offset,
                psf_cm,
                psf_deg,
                eff_area,
                centroid_x_error,
                centroid_y_error,
                eff_focal_length,
                eff_focal_length_error,
            )
        """
        r = np.hypot(image.centroid_x, image.centroid_y)
        centroid_x_error = float(getattr(image, "centroid_x_error", 0.0) or 0.0)
        centroid_y_error = float(getattr(image, "centroid_y_error", 0.0) or 0.0)

        if np.isclose(theta_offset, 0.0):
            eff_focal_length = np.nan
            eff_focal_length_error = np.nan
        else:
            tan_theta = tan(theta_offset * pi / 180.0)
            eff_focal_length = r / tan_theta
            if np.isclose(r, 0.0):
                radius_error = np.hypot(centroid_x_error, centroid_y_error)
            else:
                radius_error = np.sqrt(
                    ((image.centroid_x / r) * centroid_x_error) ** 2
                    + ((image.centroid_y / r) * centroid_y_error) ** 2
                )
            eff_focal_length_error = radius_error / tan_theta

        effective_area = image.get_effective_area(tel_transmission)
        if effective_area is None:
            effective_area = np.nan
        return (
            off_x * u.deg,
            off_y * u.deg,
            theta_offset * u.deg,
            image.get_psf(containment_fraction, "cm") * u.cm,
            image.get_psf(containment_fraction, "deg") * u.deg,
            effective_area * u.m * u.m,
            centroid_x_error * u.cm,
            centroid_y_error * u.cm,
            eff_focal_length,
            eff_focal_length_error,
        )

    def _store_results(self, _rows):
        """
        Store analysis results.

        Parameters
        ----------
        _rows: list
            List of rows containing analysis results.
        """
        _columns = [
            "off_x",
            "off_y",
            "off axis angle",
            "psf_cm",
            "psf_deg",
            "eff_area",
            "centroid_x_err",
            "centroid_y_err",
            "eff_flen",
            "eff_flen_err",
        ]
        if self.single_mirror_mode:
            _columns.append("mirror_number")
        self._results = QTable(rows=_rows, names=_columns)
        self._results.meta[BACKEND_METADATA] = settings.config.ray_tracing_backend

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
            kwargs for the plot

        Raises
        ------
        KeyError
            If key is not among the valid options.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the requested plot.
        """
        self._logger.info(f"Plotting {key} vs off-axis angle")

        try:
            columns = ["off axis angle", key]
            if key == "eff_flen":
                columns.append("eff_flen_err")
                plot = visualize.plot_1d(
                    {"Effective focal length": self._results[columns]},
                    xtitle="Off-axis angle",
                    ytitle=self.YLABEL[key],
                    no_legend=kwargs.pop("no_legend", True),
                    **kwargs,
                )
            else:
                plot = visualize.plot_table(
                    self._results[columns],
                    self.YLABEL[key],
                    no_legend=kwargs.pop("no_legend", True),
                    **kwargs,
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

            for (off_x, off_y), image in self.psf_images.items():
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
        return plot

    def plot_histogram(self, key, **kwargs):
        """
        Plot histogram of key.

        Parameters
        ----------
        key: str
            psf_cm, psf_deg, eff_area or eff_flen
        **kwargs:
            kwargs for the histogram plot

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        try:
            visualize.plot_histogram(self._results[key], **kwargs)
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
            self.psf_images[(off_x, off_y)]
            for off_x, off_y in self.off_axis_angle
            if self.psf_images and (off_x, off_y) in self.psf_images
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
