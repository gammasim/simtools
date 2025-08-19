"""Calculate incident angles using a sim_telarray PSF-style run."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.psf_analysis import PSFImage


class IncidentAnglesCalculator:
    """Run a PSF-style sim_telarray job and compute camera incident angles."""

    def __init__(
        self,
        simtel_path,
        db_config,
        config_data,
        output_dir,
        label=None,
        ray_tracing_config=None,
        use_real_camera=False,
        test=False,
    ):
        self.logger = logging.getLogger(__name__)

        # Store parameters
        self._simtel_path = Path(simtel_path)
        self.config_data = config_data
        self.output_dir = Path(output_dir)
        self.label = label or f"incident_angles_{config_data['telescope']}"
        self.ray_tracing_config = ray_tracing_config
        self.use_real_camera = use_real_camera
        self.test = test
        self.results: QTable | None = None

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

        # Configure run parameters
        self.rt_params = self._setup_rt_params(config_data)

    def __repr__(self):
        """Return a short, informative representation for logging/debugging."""
        return (
            f"IncidentAnglesCalculator(label={self.label}, telescope={self.telescope_model.name}, "
            f"site={self.site_model.site})\n"
        )

    def _setup_rt_params(self, config_data):
        zenith = config_data.get("zenith_angle", 20.0 * u.deg).to(u.deg)
        off = config_data.get("off_axis_angle", [0.0] * u.deg).to(u.deg)
        src_dist = config_data.get("source_distance", 10.0 * u.km).to(u.km)
        n_rays = int(config_data.get("number_of_rays", 10000))
        return {
            "zenith_angle": zenith,
            "off_axis_angle": off,
            "source_distance": src_dist,
            "number_of_rays": n_rays,
        }

    def run(self):
        """Run the job, parse photon list, compute and store incident angles."""
        self.logger.info("Running sim_telarray PSF-style simulation for incident angles")

        # Export model configuration files (include site model)
        self.telescope_model.write_sim_telarray_config_file(additional_model=self.site_model)

        photons_file, stars_file, log_file = self._prepare_psf_io_files()
        run_script = self._write_run_script(photons_file, stars_file, log_file)
        self._run_script(run_script, log_file)

        data = self._compute_incident_angles_from_photons(photons_file)
        self.results = QTable()
        self.results["x_pix"] = data["x_pix"]
        self.results["y_pix"] = data["y_pix"]
        self.results["incident_angle"] = data["incident_angle_deg"] * u.deg

        self._save_results()
        self.plot_incident_angles()
        return self.results

    def _prepare_psf_io_files(self):
        photons_file = self.output_dir / f"incident_angles_photons_{self.label}.lis"
        stars_file = self.output_dir / f"incident_angles_stars_{self.label}.lis"
        log_file = self.output_dir / f"incident_angles_{self.label}.log"

        if not photons_file.exists() or self.test:
            with photons_file.open("w", encoding="utf-8") as pf:
                pf.write(f"#{'=' * 50}\n")
                pf.write("# List of photons for Incident Angle simulations\n")
                pf.write(f"#{'=' * 50}\n")
                pf.write(f"# config_file = {self.telescope_model.config_file_path}\n")
                pf.write(f"# zenith_angle [deg] = {self.rt_params['zenith_angle'].value}\n")
                off = float(np.atleast_1d(self.rt_params["off_axis_angle"].value)[0])
                pf.write(f"# off_axis_angle [deg] = {off}\n")
                pf.write(f"# source_distance [km] = {self.rt_params['source_distance'].value}\n")

            with stars_file.open("w", encoding="utf-8") as sf:
                zen = float(self.rt_params["zenith_angle"].to_value(u.deg))
                dist = float(self.rt_params["source_distance"].to_value(u.km))
                sf.write(f"0. {90.0 - zen} 1.0 {dist}\n")

        return photons_file, stars_file, log_file

    def _write_run_script(self, photons_file: Path, stars_file: Path, log_file: Path) -> Path:
        script_path = self.output_dir / f"run_incident_angles_{self.label}.sh"
        simtel_bin = self._simtel_path / "sim_telarray/bin/sim_telarray"
        corsika_dummy = self._simtel_path / "sim_telarray/run9991.corsika.gz"

        theta = float(self.rt_params["zenith_angle"].to_value(u.deg))
        off = float(np.atleast_1d(self.rt_params["off_axis_angle"].to(u.deg).value)[0])
        star_photons = self.rt_params["number_of_rays"] if not self.test else 5000

        def cfg(par, val):
            return f"-C {par}={val}"

        opts = [
            f"-c {self.telescope_model.config_file_path}",
            f"-I{self.telescope_model.config_file_directory}",
        ]
        if self.use_real_camera:
            opts.append("-C USE_REAL_CAMERA=1")

        opts += [
            cfg("IMAGING_LIST", str(photons_file)),
            cfg("stars", str(stars_file)),
            cfg("altitude", self.site_model.get_parameter_value("corsika_observation_level")),
            cfg("telescope_theta", theta + off),
            cfg("star_photons", star_photons),
            cfg("telescope_phi", 0),
            cfg("camera_transmission", 1.0),
            cfg("nightsky_background", "all:0."),
            cfg("trigger_current_limit", "1e10"),
            cfg("telescope_random_angle", 0),
            cfg("telescope_random_error", 0),
            cfg("convergent_depth", 0),
            cfg("maximum_telescopes", 1),
            cfg("show", "all"),
            cfg("camera_filter", "none"),
        ]

        command = f"{simtel_bin} {' '.join(opts)} {corsika_dummy}"
        with script_path.open("w", encoding="utf-8") as sh:
            sh.write("#!/usr/bin/env bash\n\n")
            sh.write("set -e\nset -o pipefail\n\n")
            # Redirect all stdout/stderr to logfile
            sh.write(f"exec > '{log_file}' 2>&1\n\n")
            sh.write(f"{command}\n")
        script_path.chmod(script_path.stat().st_mode | 0o110)
        return script_path

    def _run_script(self, script_path: Path, log_file: Path) -> None:
        """Execute the generated shell script and stream output to the log file."""
        self.logger.info(f"Executing {script_path} (logging to {log_file})")
        try:
            subprocess.check_call([str(script_path)])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Incident angles run failed, see log: {log_file}") from exc

    def _compute_incident_angles_from_photons(self, photons_file: Path) -> dict:
        img = PSFImage(
            focal_length=self.telescope_model.get_parameter_value("focal_length"),
            containment_fraction=0.8,
            simtel_path=str(self._simtel_path),
        )
        img.read_photon_list_from_simtel_file(str(photons_file))

        x = np.array(img.photon_pos_x)
        y = np.array(img.photon_pos_y)
        f_cm = float(self.telescope_model.get_parameter_value("focal_length"))
        r = np.sqrt(x**2 + y**2)
        inc_rad = np.arctan2(r, f_cm)
        inc_deg = np.rad2deg(inc_rad)
        return {"x_pix": x.tolist(), "y_pix": y.tolist(), "incident_angle_deg": inc_deg.tolist()}

    def _save_results(self):
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to save")
            return
        output_file = self.output_dir / f"incident_angles_{self.label}.ecsv"
        self.results.write(output_file, format="ascii.ecsv", overwrite=True)

    def plot_incident_angles(self):
        """Create and save a scatter + histogram plot of incident angles."""
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
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

        ax2.hist(self.results["incident_angle"].value, bins=50, alpha=0.7, color="royalblue")
        ax2.set_xlabel("Incident Angle (deg)")
        ax2.set_ylabel("Count")
        ax2.set_title("Incident Angle Distribution")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = self.output_dir / f"incident_angles_{self.label}.png"
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

    def export_results(self):
        """Write results to ECSV and a plain-text summary file."""
        if self.results is None or len(self.results) == 0:
            self.logger.error("Cannot export results because they do not exist")
            return
        table_file = self.output_dir / f"incident_angles_{self.label}.ecsv"
        self.results.write(table_file, format="ascii.ecsv", overwrite=True)
        summary_file = self.output_dir / f"incident_angles_summary_{self.label}.txt"
        with summary_file.open("w", encoding="utf-8") as f:
            f.write(f"Incident angle results for {self.telescope_model.name}\n")
            f.write(f"Site: {self.telescope_model.site}\n")
            f.write(f"Zenith angle: {self.rt_params['zenith_angle']}\n")
            f.write(f"Off-axis angle: {self.rt_params['off_axis_angle']}\n")
            f.write(f"Source distance: {self.rt_params['source_distance']}\n\n")
            f.write(f"Number of data points: {len(self.results)}\n")
