"""Calculate incident angles using ray tracing."""

import gzip
import logging
import re
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.ray_tracing import RayTracing


class IncidentAnglesCalculator:
    """
    Calculate incident angles on the telescope camera using ray tracing.

    Parameters
    ----------
    db_config : dict
        Configuration for the database.
    config_data : dict
        Dictionary containing configuration parameters.
        Must include 'site', 'telescope', and 'model_version'.
    output_dir : str or Path
        Directory where to save output files.
    label : str, optional
        Instance label used for file naming and organization.
    ray_tracing_config : str, optional
        Path to ray tracing configuration file.
    test : bool, optional
        Whether this is a test instance (affects file paths).
    """

    def __init__(
        self,
        simtel_path,
        db_config,
        config_data,
        output_dir,
        label=None,
        ray_tracing_config=None,
        test=False,
    ):
        """Initialize IncidentAnglesCalculator class."""
        self.logger = logging.getLogger(__name__)

        # Store parameters
        self._simtel_path = simtel_path
        self.config_data = config_data
        self.output_dir = Path(output_dir)
        self.label = label or f"incident_angles_{config_data['telescope']}"
        self.ray_tracing_config = ray_tracing_config
        self.test = test
        self.results = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load models from database
        self.logger.info(
            f"Initializing models for {config_data['site']}, {config_data['telescope']}"
        )
        self.telescope_model, self.site_model = initialize_simulation_models(
            self.label,
            db_config,
            config_data["site"],
            config_data["telescope"],
            config_data["model_version"],
        )

        # Configure ray tracing parameters from config_data
        self.ray_tracing_params = self._setup_ray_tracing_params(config_data)

    def __repr__(self):
        """Return string representation of the IncidentAnglesCalculator instance."""
        return (
            f"IncidentAnglesCalculator(label={self.label}, "
            f"telescope={self.telescope_model.name}, "
            f"site={self.site_model.site})\n"
        )

    def _setup_ray_tracing_params(self, config_data):
        """
        Set up ray tracing parameters from config data.

        Parameters
        ----------
        config_data : dict
            Dictionary containing configuration parameters.

        Returns
        -------
        dict
            Dictionary with ray tracing parameters.
        """
        # Extract parameters from config_data, using defaults if not present
        zenith_angle = config_data.get("zenith_angle")
        if zenith_angle is not None:
            zenith_angle = zenith_angle.to("deg")
        else:
            zenith_angle = 20.0 * u.deg
            self.logger.info(f"Setting zenith angle to default value {zenith_angle}")

        off_axis_angle = config_data.get("off_axis_angle")
        if off_axis_angle is not None:
            off_axis_angle = off_axis_angle.to("deg")
        else:
            off_axis_angle = [0.0] * u.deg
            self.logger.info(f"Setting off-axis angle to default value {off_axis_angle}")

        source_distance = config_data.get("source_distance")
        if source_distance is not None:
            source_distance = source_distance.to("km")
        else:
            source_distance = 10.0 * u.km
            self.logger.info(f"Setting source distance to default value {source_distance}")

        number_of_rays = config_data.get("number_of_rays", 10000)

        return {
            "zenith_angle": zenith_angle,
            "off_axis_angle": off_axis_angle,
            "source_distance": source_distance,
            "number_of_rays": number_of_rays,
        }

    def run(self):
        """
        Run the incident angle calculation.

        Returns
        -------
        astropy.table.QTable
            Table containing the incident angle data
        """
        self.logger.info("Running ray tracing simulation")

        # Export model configuration files
        self.telescope_model.write_sim_telarray_config_file()

        # Create the RayTracing object
        ray_tracing = RayTracing(
            telescope_model=self.telescope_model,
            site_model=self.site_model,
            simtel_path=self._simtel_path,
            label=self.label,
            zenith_angle=self.ray_tracing_params["zenith_angle"],
            off_axis_angle=self.ray_tracing_params["off_axis_angle"],
            source_distance=self.ray_tracing_params["source_distance"],
        )

        # Run ray tracing simulation
        self.logger.info("Simulating ray tracing...")
        ray_tracing.simulate(test=self.test, force=True)

        # Analyze ray tracing results
        self.logger.info("Analyzing ray tracing results...")
        ray_tracing.analyze(force=True)

        # Get the photons file path
        photons_file = self._get_photons_file_path(ray_tracing)

        # Parse ray tracing output
        self.logger.info(f"Parsing ray tracing output file: {photons_file}")
        data = self._parse_ray_tracing_output(photons_file)

        # Create results table
        self.results = QTable()
        self.results["x_pix"] = data["x_pix"]
        self.results["y_pix"] = data["y_pix"]
        self.results["incident_angle"] = data["incident_angle_deg"] * u.deg

        # Save results to file
        self._save_results()

        # Generate plots
        self.plot_incident_angles()

        return self.results

    def _get_photons_file_path(self, ray_tracing):
        """
        Get the path to the photons file generated by ray tracing.

        Parameters
        ----------
        ray_tracing : RayTracing
            The RayTracing instance that was used to run the simulation

        Returns
        -------
        Path
            Path to the photons file
        """
        # Typically in the ray tracing output directory with a specific pattern
        output_dir = ray_tracing.output_directory
        photons_files = list(output_dir.glob("ray_tracing_photons_*.lis.gz"))

        if not photons_files:
            photons_files = list(output_dir.glob("ray_tracing_photons_*.lis"))

        if not photons_files:
            raise FileNotFoundError(f"No photons file found in {output_dir}")

        return photons_files[0]

    def _parse_ray_tracing_output(self, output_file):
        """
        Parse the ray tracing output file to extract incident angles.

        Parameters
        ----------
        output_file : str or Path
            Path to the ray tracing output file

        Returns
        -------
        dict
            Dictionary containing the extracted data
        """
        output_file = Path(output_file)
        self.logger.info(f"Parsing ray tracing output file: {output_file}")

        # Initialize data dictionaries
        data = {
            "x_pix": [],
            "y_pix": [],
            "incident_angle_deg": [],
        }

        # Check if the file is gzipped
        is_gzipped = output_file.suffix == ".gz"

        # Regular expressions to extract information
        pixel_pattern = re.compile(r"pixel\s+(\d+)\s+\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)")

        # Open the file (handling gzipped files if necessary)
        if is_gzipped:
            open_func = gzip.open
            mode = "rt"  # Text mode for gzipped files
        else:
            open_func = open
            mode = "r"

        with open_func(output_file, mode) as f:
            for line in f:
                # Check for pixel information
                pixel_match = pixel_pattern.search(line)
                if pixel_match:
                    # current_x =
                    float(pixel_match.group(2))
                    # current_y =
                    float(pixel_match.group(3))
                    continue

                # Get incident angle information, calculate..
                # add here logic to extract incident angle

        self.logger.info(f"Found {len(data['x_pix'])} incident angle data points")

        return data

    def _save_results(self):
        """Save the results to file."""
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to save")
            return

        output_file = self.output_dir / f"incident_angles_{self.label}.ecsv"
        self.results.write(output_file, format="ascii.ecsv", overwrite=True)
        self.logger.info(f"Results saved to {output_file}")

    def plot_incident_angles(self):
        """
        Plot the incident angle distribution.

        Creates two plots:
        1. Scatter plot of pixel positions colored by incident angle
        2. Histogram of incident angles
        """
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to plot")
            return

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot of pixel positions colored by incident angle
        scatter = ax1.scatter(
            self.results["x_pix"],
            self.results["y_pix"],
            c=self.results["incident_angle"].value,
            s=10,
            cmap="viridis",
            alpha=0.8,
        )
        ax1.set_xlabel("Pixel X position")
        ax1.set_ylabel("Pixel Y position")
        ax1.set_title(f"Incident Angles - {self.telescope_model.name}")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Incident Angle (deg)")

        # Histogram of incident angles
        ax2.hist(self.results["incident_angle"].value, bins=50, alpha=0.7, color="royalblue")
        ax2.set_xlabel("Incident Angle (deg)")
        ax2.set_ylabel("Count")
        ax2.set_title("Incident Angle Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"incident_angles_{self.label}.png"
        plt.savefig(output_file, dpi=200)
        self.logger.info(f"Plot saved to {output_file}")

        # Statistics
        mean_angle = np.mean(self.results["incident_angle"])
        median_angle = np.median(self.results["incident_angle"])
        min_angle = np.min(self.results["incident_angle"])
        max_angle = np.max(self.results["incident_angle"])

        self.logger.info("Incident angle statistics:")
        self.logger.info(f"  Mean: {mean_angle:.2f}")
        self.logger.info(f"  Median: {median_angle:.2f}")
        self.logger.info(f"  Range: {min_angle:.2f} - {max_angle:.2f}")

        plt.close(fig)

    def export_results(self):
        """
        Export results to files.

        Saves both the data table and a summary text file.
        """
        if not self.results or len(self.results) == 0:
            self.logger.error("Cannot export results because they do not exist")
            return

        # Save ECSV table
        table_file = self.output_dir / f"incident_angles_{self.label}.ecsv"
        self.logger.info(f"Exporting incident angles table to {table_file}")
        self.results.write(table_file, format="ascii.ecsv", overwrite=True)

        # Save summary file
        summary_file = self.output_dir / f"incident_angles_summary_{self.label}.txt"
        self.logger.info(f"Exporting summary results to {summary_file}")

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Incident angle results for {self.telescope_model.name}\n")
            f.write(f"Site: {self.telescope_model.site}\n")
            f.write(f"Zenith angle: {self.ray_tracing_params['zenith_angle']}\n")
            f.write(f"Off-axis angle: {self.ray_tracing_params['off_axis_angle']}\n")
            f.write(f"Source distance: {self.ray_tracing_params['source_distance']}\n\n")

            f.write(f"Number of data points: {len(self.results)}\n")
            f.write(f"Mean incident angle: {np.mean(self.results['incident_angle']):.3f}\n")
            f.write(f"Median incident angle: {np.median(self.results['incident_angle']):.3f}\n")
            f.write(f"Min incident angle: {np.min(self.results['incident_angle']):.3f}\n")
            f.write(f"Max incident angle: {np.max(self.results['incident_angle']):.3f}\n")
