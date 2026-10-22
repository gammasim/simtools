"""Calculate photon incident angles on focal plane and primary/secondary mirrors.

The ray-tracing backend (``simtools.ray_tracing.backends``) provides, per photon,
the angle of incidence at the focal surface with respect to the optical axis
[deg] and, for dual-mirror telescopes, the angles of incidence on the primary
and secondary mirror [deg] and the hit points on them.
"""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable, vstack

from simtools.data_model.metadata_collector import MetadataCollector
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing import backends, parameter_changes
from simtools.ray_tracing.backends import StarImageRequest
from simtools.simtel import simtel_seeds


class IncidentAnglesCalculator:
    """Trace a star and compute incident angles at mirrors or focal surfaces.

    Parameters
    ----------
    config_data : dict
        Simulation configuration (e.g. ``site``, ``telescope``, ``model_version``,
        ``zenith_angle``, ``off_axis_angle``, ``source_distance``, ``number_of_photons``).
        Zenith defaults to 0 degrees. Off-axis angles use the PSF camera-x convention.
    output_dir : str or pathlib.Path
        Output directory where logs, scripts, photons files and results are written.
    label : str, optional
        Label used to name outputs; defaults to ``incident_angles_<telescope>`` when omitted.

    Notes
    -----
    Additional options are read from ``config_data`` when present:
    - ``perfect_mirror`` (bool, default False)
    Focal-plane incidence angles are calculated for all telescope models. Primary and
    secondary mirror incidence angles are additionally calculated
    only for dual-mirror telescopes.
    """

    def __init__(
        self,
        config_data,
        output_dir,
        label=None,
    ):
        self.logger = logging.getLogger(__name__)

        self.config_data = config_data
        self.zenith_angle_deg = u.Quantity(config_data.get("zenith_angle", 0 * u.deg)).to_value(
            u.deg
        )
        self.output_dir = Path(output_dir)
        self.label = label or f"incident_angles_{config_data['telescope']}"
        cfg = config_data
        self.perfect_mirror = cfg.get("perfect_mirror", False)
        self.results = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.scripts_dir = self.output_dir / "scripts"
        self.photons_dir = self.output_dir / "photons_files"
        self.model_dir = self.output_dir / "model" / f"incident_angles_{self.label}"
        self._intermediate_files = set()
        self.cleanup_options = {
            "patterns": (f"{self.model_dir.relative_to(self.output_dir)}/**/*",),
            "files": self._intermediate_files,
            "directories": (
                self.model_dir,
                self.model_dir.parent,
                self.scripts_dir,
                self.photons_dir,
            ),
            "exclude": set(self.model_dir.rglob("*")),
        }
        self.results_dir = self.output_dir / "incident_angles"
        for d in (self.logs_dir, self.scripts_dir, self.photons_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Initializing models for %s, %s",
            config_data["site"],
            config_data["telescope"],
        )
        self.telescope_model, self.site_model, _ = initialize_simulation_models(
            label=self.label,
            site=config_data["site"],
            telescope_name=config_data["telescope"],
            model_version=config_data["model_version"],
            model_directory=self.model_dir,
        )
        self.calculate_primary_secondary_angles = (
            self.telescope_model.get_parameter_value("mirror_class") == 2
        )

    def _label_suffix(self):
        """Build a filename suffix including telescope and off-axis angle.

        Returns
        -------
        str
            Suffix of the form ``"<label>_<telescope>_off<angle>"`` where
            ``<angle>`` is formatted without trailing zeros.
        """
        tel = str(self.config_data.get("telescope", "TEL"))
        off = float(self.config_data.get("off_axis_angle", 0.0 * u.deg).to_value(u.deg))
        return f"{self.label}_{tel}_off{off:g}"

    def run(self):
        """Trace the star with the ray-tracing backend and return an angle table.

        Returns
        -------
        astropy.table.QTable
            Table containing at least the ``angle_incidence_focal`` column
            and, when configured, primary/secondary angles and hit geometry.
        """
        request = StarImageRequest(
            off_axis_x=float(self.config_data["off_axis_angle"].to_value(u.deg)),
            off_axis_y=0.0,
            source_distance_km=self._source_distance_km(),
            seeds=tuple(simtel_seeds.instance_seed_pairs()),
            photons_per_instance=int(self.config_data["number_of_photons"]),
            parameter_changes=parameter_changes.PERFECT_MIRROR if self.perfect_mirror else {},
        )
        backend = backends.get_backend()(
            self.telescope_model, self.site_model, self.label, self.zenith_angle_deg, False
        )
        data, intermediate_files = backend.incidence_angles(
            request,
            self.output_dir,
            self._label_suffix(),
            self.calculate_primary_secondary_angles,
            keep_files=self.config_data.get("keep_photon_files", False),
        )
        self._intermediate_files.update(intermediate_files)

        self.results = QTable()
        self.results["angle_incidence_focal"] = np.asarray(data["angle_incidence_focal"]) * u.deg
        if self.calculate_primary_secondary_angles:
            for mirror in ("primary", "secondary"):
                self.results[f"angle_incidence_{mirror}"] = (
                    np.asarray(data[f"angle_incidence_{mirror}"]) * u.deg
                )
            for mirror in ("primary", "secondary"):
                x, y = (np.asarray(data[f"{mirror}_hit_{axis}"]) for axis in "xy")
                self.results[f"{mirror}_hit_radius"] = np.hypot(x, y) * u.m
            for mirror in ("primary", "secondary"):
                for axis in "xy":
                    self.results[f"{mirror}_hit_{axis}"] = (
                        np.asarray(data[f"{mirror}_hit_{axis}"]) * u.m
                    )
        self._save_results()
        return self.results

    def run_for_offsets(self, offsets):
        """Run the simulation for multiple off-axis angles.

        For each off-axis angle provided, run a full simulation, labeling output files
        accordingly.

        Parameters
        ----------
        offsets : Iterable[float]
            Off-axis angles in degrees.

        Returns
        -------
        dict[float, astropy.table.QTable]
            Mapping from off-axis angle (deg) to the resulting table.
        """
        results_by_offset = {}
        base_off = self.config_data.get("off_axis_angle", 0.0 * u.deg)

        for off in offsets:
            self.config_data["off_axis_angle"] = float(off) * u.deg
            self.logger.info(f"Running for off-axis angle {off:g} deg")
            tbl = self.run()
            results_by_offset[float(off)] = tbl.copy()

        self.config_data["off_axis_angle"] = base_off
        return results_by_offset

    def _source_distance_km(self):
        """Return source distance as a scalar value in km."""
        source_distance = self.config_data["source_distance"]
        if isinstance(source_distance, u.Quantity):
            return source_distance.to_value(u.km)
        return float(source_distance)

    def _save_results(self):
        """Save the results to an ECSV file with metadata."""
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to save")
            return
        output_file = self.results_dir / f"incident_angles_{self._label_suffix()}.ecsv"

        data = self.results["angle_incidence_focal"].to(u.deg).value
        bin_centers, hist = self._calculate_histogram(data, bins=1000)

        output_table = QTable()
        output_table["incidence_angle"] = bin_centers * u.deg
        output_table["fraction"] = hist
        output_table.meta["normalization"] = "probability density per degree"
        output_table.meta["bin_width_deg"] = 0.09

        output_table.write(output_file, format="ascii.ecsv", overwrite=True)

        MetadataCollector.dump(
            args_dict=self.config_data,
            output_file=output_file.with_suffix(".yml"),
        )

    def save_model_parameters(self, results_by_offset):
        """Write incident-angle model-parameter tables and their JSON metadata.

        Export the filter distribution and enabled mirror distributions. Both mirror
        parameters are exported only for dual-mirror optics. Densities describe unweighted rays
        completing the optical path, without requiring a camera pixel hit.

        Files are placed under a telescope-named subdirectory to match repository convention.

        Parameters
        ----------
        results_by_offset : dict
            Dictionary with results for each offset.
        """
        tables = [res for res in results_by_offset.values() if res is not None and len(res) > 0]
        if not tables:
            self.logger.warning("No results to write model parameters.")
            return

        combined_table = vstack(tables)
        telescope = self.config_data["telescope"]
        param_version = (
            self.config_data.get("parameter_version") or self.config_data["model_version"]
        )
        param_dir = self.output_dir / telescope
        param_dir.mkdir(parents=True, exist_ok=True)

        parameters = {"camera_filter_photon_incident_angle": "angle_incidence_focal"}
        mirror_class = self.telescope_model.get_parameter_value("mirror_class")
        if mirror_class == 2:
            parameters["primary_mirror_incidence_angle"] = "angle_incidence_primary"
            parameters["secondary_mirror_incidence_angle"] = "angle_incidence_secondary"

        for param_name, col_name in parameters.items():
            if col_name not in combined_table.colnames:
                raise ValueError(f"Missing incidence-angle column for {param_name}: {col_name}")

            data = combined_table[col_name].to(u.deg).value
            data = data[np.isfinite(data)]
            if len(data) == 0:
                raise ValueError(f"No finite incidence angles for {param_name}")

            table = self._build_incidence_distribution_table(data)
            table.meta["off_axis_angles_deg"] = [float(offset) for offset in results_by_offset]
            table.meta["zenith_angle_deg"] = float(self.zenith_angle_deg)
            if param_name == "camera_filter_photon_incident_angle":
                table.meta["angle_reference"] = "optical axis at focal surface"
            else:
                table.meta["angle_reference"] = "local mirror surface normal"
            ecsv_file = f"{param_name}.ecsv"
            ModelDataWriter.write_product_data(
                output_file=param_dir / ecsv_file,
                product_data=table,
                metadata=MetadataCollector(args_dict=self.config_data),
            )
            ModelDataWriter.write_model_parameter(
                parameter_name=param_name,
                value=ecsv_file,
                instrument=telescope,
                parameter_version=param_version,
                output_file=f"{param_name}.json",
                output_path=param_dir,
                metadata_input_dict=self.config_data,
            )

    @staticmethod
    def _build_incidence_distribution_table(data):
        """Build an angle probability-density table for model-parameter export."""
        bin_centers, hist = IncidentAnglesCalculator._calculate_histogram(data, bins=100)
        table = QTable()
        table["incidence_angle"] = bin_centers * u.deg
        table["fraction"] = hist
        table.meta["normalization"] = "probability density per degree"
        table.meta["bin_width_deg"] = 0.9
        table.meta["sampling"] = "unweighted rays completing the telescope optical path"
        return table

    @staticmethod
    def _calculate_histogram(data, bins=100):
        """
        Calculate the probability density of finite incidence angles in [0, 90] degrees.

        Parameters
        ----------
        data : numpy.ndarray
            Data to calculate histogram for.
        bins : int
            Number of bins.

        Returns
        -------
        tuple
            Bin centers in degrees and probability densities per degree, whose
            sum multiplied by the bin width in degrees equals one.
        """
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            return np.array([]), np.array([])
        if np.any((data < 0) | (data > 90)):
            raise ValueError("Incidence angles must be between 0 and 90 degrees.")
        hist, bin_edges = np.histogram(data, bins=bins, range=(0.0, 90.0), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist
