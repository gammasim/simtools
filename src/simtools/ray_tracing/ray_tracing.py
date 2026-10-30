"""
Ray tracing simulations and analysis.

Traces star light through the telescope optics with the configured ray-tracing backend
(``simtools.ray_tracing.backends.interface``) and computes the PSF (D80 containment
diameter), effective mirror area and effective focal length versus off-axis angle.
"""

import logging
from copy import copy
from math import pi, tan
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import numpy as np
from astropy.table import QTable

from simtools import constants
from simtools.context import RunContext
from simtools.model.model_utils import compute_telescope_transmission
from simtools.ray_tracing import parameter_changes
from simtools.ray_tracing.backends import interface
from simtools.ray_tracing.backends.interface import StarImageRequest
from simtools.utils import names, random
from simtools.visualization import visualize

INVALID_KEY_TO_PLOT = "Invalid key to plot"
DEFAULT_MAX_WORKERS = 8
BACKEND_METADATA = "ray_tracing_backend"
_CARDINAL_OFFSETS = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}


class RayTracing:
    """
    Ray tracing simulations and analysis.

    Parameters
    ----------
    telescope_model, site_model : TelescopeModel, SiteModel
        Telescope and site models.
    label : str
        Label used for output file naming.
    zenith_angle : astropy.units.Quantity
        Zenith angle.
    off_axis_angle : list of tuples or astropy.units.Quantity, optional
        Off-axis angles as (x, y) tuples in degrees, or scalar angles. Scalar angles
        generate offsets in the cardinal directions ``offset_directions``.
    source_distance : astropy.units.Quantity
        Source distance.
    single_mirror_mode : bool
        Trace each mirror panel instead of the telescope optics.
    use_random_focal_length : bool
        Use random focal length variation of the mirror panels.
    random_focal_length_seed : int
        Seed for the random focal length variation.
    mirror_numbers : list, str
        List of mirror numbers (or 'all').
    offset_file : Path or str, optional
        ECSV file with x, y offset columns (in degrees).
    offset_directions : list, optional
        Cardinal directions ['N', 'S', 'E', 'W'] for scalar offsets.
    context : RunContext, optional
        Run context.
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
        context=None,
    ):
        """Initialize RayTracing class."""
        self._logger = logging.getLogger(__name__)

        self.context = context or RunContext.from_environment()
        self._io_handler = self.context.paths

        self.telescope_model, self.site_model = telescope_model, site_model
        self.label = label if label is not None else self.telescope_model.label

        self.zenith_angle = zenith_angle.to("deg").value
        self.azimuth_angle = 0.0 * u.deg
        self._logger.debug(
            f"Telescope pointing set to zenith={self.zenith_angle:.3f} deg, "
            f"azimuth={self.azimuth_angle:.3f} deg"
        )

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
        """Return (x, y) offsets in degrees from (x, y) tuples or from scalar angles."""
        if isinstance(off_axis_angle, (list, tuple)) and (
            len(off_axis_angle) > 0 and isinstance(off_axis_angle[0], (tuple, list))
        ):
            return [
                tuple(x.to("deg").value if isinstance(x, u.Quantity) else float(x) for x in offset)
                for offset in off_axis_angle
            ]

        if isinstance(off_axis_angle, (list, tuple)):
            angles_deg = np.asarray(off_axis_angle)
        else:
            angles_deg = off_axis_angle.to("deg").value
        angles_deg = np.atleast_1d(np.around(angles_deg, 5))
        if offset_directions is None:
            offset_directions = ["N", "S", "E", "W"]
        return self._cardinal_offsets(angles_deg, offset_directions)

    def _cardinal_offsets(self, angles_deg, offset_directions):
        """Return (x, y) offsets for each angle along the given cardinal directions."""
        offsets = []
        for angle in angles_deg:
            # avoid generating multiple identical offsets at zero angle
            if np.isclose(angle, 0.0, rtol=0.0, atol=1e-12):
                offsets.append((0.0, 0.0))
                continue
            for direction in offset_directions:
                if direction not in _CARDINAL_OFFSETS:
                    self._logger.warning(f"Unknown direction {direction}, skipping")
                    continue
                dx, dy = _CARDINAL_OFFSETS[direction]
                offsets.append((dx * angle, dy * angle))
        return offsets

    def _load_offset_file(self, offset_file_path):
        """Load (x, y) offsets in degrees from an ECSV file with columns x and y."""
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

    def _initialize_mirror_configuration(self, source_distance, mirror_numbers):
        """Return the mirrors to trace: mirror panels in single-mirror mode, else the telescope."""
        if self.single_mirror_mode:
            return self._initialize_single_mirror_mode(mirror_numbers)
        return {
            0: {
                "source_distance": source_distance.to("km").value,
                "focal_length": self.telescope_model.get_parameter_value("focal_length"),
            }
        }

    def _initialize_single_mirror_mode(self, mirror_numbers):
        """Return focal length and source distance of each mirror panel."""
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
        """Return the mirror panel focal length, with random variation if requested."""
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
        Run the ray tracing simulations with the configured backend.

        Parameters
        ----------
        test : bool
            Simulate much fewer photons.
        force : bool
            Simulate again even if the backend keeps earlier results.
        compress_photons : bool
            Compress the photon files written by the backend.
        max_workers : int
            Maximum number of concurrent simulations of the backend.

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
            self._backend = interface.get_backend(context=self.context)(
                self.telescope_model,
                self.site_model,
                self.label,
                self.zenith_angle,
                self.single_mirror_mode,
                context=self.context,
            )
        return self._backend

    def _requests(self, test=False, for_simulation=True):
        """Star images to trace: one per off-axis angle (and mirror in single-mirror mode)."""
        photons = self._photons_per_instance(test)
        seeds = self._instance_seeds(single_instance=test) if for_simulation else ()
        return [
            self._request(off_x, off_y, mirror_number, photons or 100000, seeds, for_simulation)
            for off_x, off_y in self.off_axis_angle
            for mirror_number in self.mirrors
        ]

    def _photons_per_instance(self, test):
        """Photons per instance: 5000 in test mode, else ``number_of_photons`` (or None)."""
        if test:
            return 5000
        photons = self.context.args.get("number_of_photons")
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
        """
        Return (instrument seed, simulation seed) of each random instrument instance.

        A single instance in single-mirror mode or if ``single_instance`` is set (test
        mode); otherwise ``random_instrument_instances``.
        """
        instances = self.context.args.get(
            "random_instrument_instances", constants.RAY_TRACING_INSTRUMENT_INSTANCES
        )
        if single_instance or self.single_mirror_mode:
            instances = 1
        instrument_seed = self.context.args.get("instrument_seed")
        simulation_seed = self.context.args.get("simulation_seed")
        if isinstance(simulation_seed, list | tuple):
            if len(simulation_seed) == 2:
                instrument_seed = simulation_seed[0]
                simulation_seed = simulation_seed[1]
            elif len(simulation_seed) == 1:
                simulation_seed = simulation_seed[0]
            else:
                raise ValueError(
                    "simulation_seed must contain one value (simulation seed) or two values "
                    "(instrument seed, simulation seed)."
                )
        instrument_seed = instrument_seed or random.seeds(max_seed=constants.SIMTEL_MAX_SEED)
        simulation_seed = simulation_seed or random.seeds(max_seed=constants.SIMTEL_MAX_SEED)
        return list(
            zip(
                random.instance_seeds(
                    instrument_seed, instances, max_seed=constants.SIMTEL_MAX_SEED
                ),
                random.instance_seeds(
                    simulation_seed, instances, max_seed=constants.SIMTEL_MAX_SEED
                ),
                strict=True,
            )
        )

    def _mirror_seeds(self, seeds, mirror_number):
        """Return the seeds of one mirror in single-mirror mode (each mirror has its own)."""
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
        Analyze the ray tracing results: PSF containment, effective area and focal length.

        The effective mirror area is detected_photons * total_area / total_photons and the
        effective focal length is centroid_radius / tan(off_axis_angle).

        Parameters
        ----------
        export : bool
            If True, export the results to a file.
        force : bool
            If True, existing results files are removed and the analysis is repeated.
        use_rx : bool
            If True, use the rx binary of sim_telarray; otherwise the internal psf_analysis.
        no_tel_transmission : bool
            If True, the telescope transmission is not applied.
        containment_fraction : float
            Containment fraction for the PSF calculation, in [0, 1].
        save_photons : bool
            If True, keep the photon files of the backend.
        keep_images : bool
            If True, keep the PSF image of each off-axis angle (see ``images``).
        """
        do_analyze = not self._file_results.exists() or force
        if not do_analyze:
            self._read_results()
            do_analyze = (
                self._results.meta.get(BACKEND_METADATA)
                != self.context.software.ray_tracing_backend
            )

        tel_transmission_pars = (
            self.telescope_model.get_parameter_value("telescope_transmission")
            if not no_tel_transmission
            else [1, 0, 0, 0]
        )
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
        """Analyze the image of each off-axis angle and mirror; returns the result rows."""
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
            if not do_analyze:
                continue

            current_result = self._analyze_image(
                image,
                request.off_axis_x,
                request.off_axis_y,
                theta_offset,
                containment_fraction,
                tel_transmission,
            )
            if self.single_mirror_mode:
                current_result = (*current_result, mirror_number)
            _rows.append(current_result)
        return _rows

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
        Extract the analysis results of one PSF image.

        The effective focal length is f_eff = r / tan(off_axis_angle), where r is the
        distance from the image centroid to the optical axis.

        Returns
        -------
        tuple
            (off_x, off_y, theta_offset, psf_cm, psf_deg, eff_area, centroid_x_error,
            centroid_y_error, eff_focal_length, eff_focal_length_error), with units.
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
        """Store the analysis rows as results table."""
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
        self._results.meta[BACKEND_METADATA] = self.context.software.ray_tracing_backend

    def export_results(self):
        """Export the results to an ECSV file."""
        if self._results:
            self._logger.info(f"Exporting results to {self._file_results}")
            astropy.io.ascii.write(self._results, self._file_results, format="ecsv", overwrite=True)
        else:
            self._logger.error("No results to export")

    def get_psf_mm(self, row_index: int = 0) -> float:
        """
        Return the PSF diameter from the analysis results in mm.

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
        psf_cm = psf.to_value(u.cm) if isinstance(psf, u.Quantity) else float(psf)
        return psf_cm * 10.0

    def _read_results(self):
        """Read existing results file and store it in _results."""
        self._results = astropy.io.ascii.read(self._file_results, format="ecsv")

    def plot(self, key, save=False, psf_diameter_cm=None, **kwargs):
        """
        Plot the analysis results versus the radial off-axis angle.

        Parameters
        ----------
        key : str
            psf_cm, psf_deg, eff_area or eff_flen.
        save : bool
            If True, the figures are saved.
        psf_diameter_cm : float
            PSF diameter marked in the cumulative PSF plot (in cm).
        **kwargs
            Passed to the plot function.

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
            self._save_figures(plot, key, psf_diameter_cm)
        return plot

    def _save_figures(self, plot, key, psf_diameter_cm):
        """Save the summary plot and the PSF image plots of each off-axis angle."""
        figures_dir = self.output_directory.joinpath("figures")
        figures_dir.mkdir(exist_ok=True)
        plot_file = figures_dir.joinpath(
            self._generate_file_name(file_type="ray_tracing", suffix=".pdf", extra_label=key)
        )
        self._logger.info(f"Saving fig in {plot_file}")
        # Keep the suffix: file names contain decimal points (e.g. "za20.0deg").
        visualize.save_figure(plot, plot_file, figure_format=[plot_file.suffix[1:]], close=True)

        for (off_x, off_y), image in self.psf_images.items():
            image_file = figures_dir.joinpath(
                self._generate_file_name(
                    file_type="ray_tracing",
                    off_axis_x=off_x,
                    off_axis_y=off_y,
                    suffix=".pdf",
                    extra_label=f"image_{key}",
                )
            )
            self._logger.info(f"Saving PSF image to {image_file}")
            image.plot_image(file_name=image_file)

            cumulative_file = figures_dir.joinpath(
                self._generate_file_name(
                    file_type="ray_tracing",
                    off_axis_x=off_x,
                    off_axis_y=off_y,
                    suffix=".pdf",
                    extra_label=f"cumulative_psf_{key}",
                )
            )
            self._logger.info(f"Saving cumulative PSF to {cumulative_file}")
            image.plot_cumulative(file_name=cumulative_file, psf_diameter_cm=psf_diameter_cm)

    def plot_histogram(self, key, **kwargs):
        """
        Plot a histogram of a results column.

        Parameters
        ----------
        key : str
            psf_cm, psf_deg, eff_area or eff_flen.
        **kwargs
            Passed to the histogram plot.

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
        Return the analyzed PSF images, one per off-axis angle.

        Returns
        -------
        list of PSFImage or None
            PSF images with photon positions, centroids and PSF containment diameters.
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
        """Generate the file name of an output file with (x, y) offsets."""
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
