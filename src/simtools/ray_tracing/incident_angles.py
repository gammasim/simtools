"""Calculate incident angles using a sim_telarray PSF-style run.

Parses the imaging list (``.lis``) produced by sim_telarray_debug_trace and uses column 24:
Angle of incidence at focal surface, with respect to the optical axis [deg].
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

from simtools.model.model_utils import initialize_simulation_models
from simtools.visualization.visualize import plot_incident_angles as _plot_incident_angles


class IncidentAnglesCalculator:
    """Run a PSF-style sim_telarray job and compute incident angles at the focal surface."""

    def __init__(
        self,
        simtel_path,
        db_config,
        config_data,
        output_dir,
        label: str | None = None,
        perfect_mirror: bool = False,
        overwrite_rdna: bool = False,
        mirror_reflection_random_angle: float | None = None,
        algn: float | None = None,
        test: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # Store parameters
        self._simtel_path = Path(simtel_path)
        self.config_data = config_data
        self.output_dir = Path(output_dir)
        self.label = label or f"incident_angles_{config_data['telescope']}"
        self.perfect_mirror = perfect_mirror
        self.overwrite_rdna = overwrite_rdna
        self.mirror_reflection_random_angle = mirror_reflection_random_angle
        self.algn = algn
        self.test = test
        self.results: QTable | None = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load models from database
        self.logger.info(
            "Initializing models for %s, %s",
            config_data["site"],
            config_data["telescope"],
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

    def __repr__(self) -> str:
        """Return a concise representation helpful for logging/debugging."""
        return f"IncidentAnglesCalculator(label={self.label})"

    def _setup_rt_params(self, config_data):
        zenith = config_data.get("zenith_angle", 20.0 * u.deg).to(u.deg)
        off = config_data.get("off_axis_angle", 0.0 * u.deg).to(u.deg)
        src_dist = config_data.get("source_distance", 10.0 * u.km).to(u.km)
        n_rays = int(config_data.get("number_of_rays", 10000))
        return {
            "zenith_angle": zenith,
            "off_axis_angle": off,
            "source_distance": src_dist,
            "number_of_rays": n_rays,
        }

    def run(self) -> QTable:
        """Run sim_telarray, parse imaging list, and return an angle table."""
        self.logger.info("Running sim_telarray PSF-style simulation for incident angles")

        # Export model configuration files (include site model)
        self.telescope_model.write_sim_telarray_config_file(additional_model=self.site_model)

        photons_file, stars_file, log_file = self._prepare_psf_io_files()
        run_script = self._write_run_script(photons_file, stars_file, log_file)
        self._run_script(run_script, log_file)

        data = self._compute_incidence_angles_from_imaging_list(photons_file)
        self.results = QTable()
        self.results["angle_incidence_focal"] = data["angle_incidence_focal_deg"] * u.deg

        self._save_results()
        _plot_incident_angles(self.results, self.output_dir, self.label, self.logger)
        return self.results

    def _prepare_psf_io_files(self):
        photons_file = self.output_dir / f"incident_angles_photons_{self.label}.lis"
        stars_file = self.output_dir / f"incident_angles_stars_{self.label}.lis"
        log_file = self.output_dir / f"incident_angles_{self.label}.log"

        if photons_file.exists():
            try:
                photons_file.unlink()
            except OSError as err:
                self.logger.warning(f"Failed to remove existing photons file {photons_file}: {err}")

        with photons_file.open("w", encoding="utf-8") as pf:
            pf.write(f"#{'=' * 50}\n")
            pf.write("# Imaging list for Incident Angle simulations\n")
            pf.write(f"#{'=' * 50}\n")
            pf.write(f"# config_file = {self.telescope_model.config_file_path}\n")
            pf.write(f"# zenith_angle [deg] = {self.rt_params['zenith_angle'].value}\n")
            pf.write(
                f"# off_axis_angle [deg] = {self.rt_params['off_axis_angle'].to_value(u.deg)}\n"
            )
            pf.write(f"# source_distance [km] = {self.rt_params['source_distance'].value}\n")

        with stars_file.open("w", encoding="utf-8") as sf:
            zen = float(self.rt_params["zenith_angle"].to_value(u.deg))
            dist = float(self.rt_params["source_distance"].to_value(u.km))
            sf.write(f"0. {90.0 - zen} 1.0 {dist}\n")

        return photons_file, stars_file, log_file

    def _write_run_script(self, photons_file: Path, stars_file: Path, log_file: Path) -> Path:
        script_path = self.output_dir / f"run_incident_angles_{self.label}.sh"
        simtel_bin = self._simtel_path / "sim_telarray/bin/sim_telarray_debug_trace"
        corsika_dummy = self._simtel_path / "sim_telarray/run9991.corsika.gz"

        theta = float(self.rt_params["zenith_angle"].to_value(u.deg))
        off = float(self.rt_params["off_axis_angle"].to_value(u.deg))
        star_photons = self.rt_params["number_of_rays"] if not self.test else 5000

        def cfg(par, val):
            return f"-C {par}={val}"

        opts = [
            f"-c {self.telescope_model.config_file_path}",
            f"-I{self.telescope_model.config_file_directory}",
        ]
        if self.perfect_mirror:
            opts.extend(
                [
                    "-DPERFECT_DISH=1",
                    "-C telescope_random_angle=0",
                    "-C telescope_random_error=0",
                    "-C random_focal_length=0",
                    "-C mirror_reflection_random_angle=0",
                    "-C mirror_align_random_distance=0",
                    "-C mirror_align_random_horizontal=0,28,0,0",
                    "-C mirror_align_random_vertical=0,28,0,0",
                ]
            )

        if self.mirror_reflection_random_angle is not None:
            opts.append(cfg("mirror_reflection_random_angle", self.mirror_reflection_random_angle))
        elif self.overwrite_rdna:
            opts.append(cfg("mirror_reflection_random_angle", 0))

        if self.algn is not None:
            opts.extend(
                [
                    cfg("mirror_align_random_horizontal", f"{self.algn},28.,0.0,0.0"),
                    cfg("mirror_align_random_vertical", f"{self.algn},28.,0.0,0.0"),
                ]
            )

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
            sh.write(f"exec > '{log_file}' 2>&1\n\n")
            sh.write(f"{command}\n")
        script_path.chmod(script_path.stat().st_mode | 0o110)
        return script_path

    def _run_script(self, script_path: Path, log_file: Path) -> None:
        self.logger.info("Executing %s (logging to %s)", script_path, log_file)
        try:
            subprocess.check_call([str(script_path)])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Incident angles run failed, see log: {log_file}") from exc

    def _compute_incidence_angles_from_imaging_list(self, photons_file: Path) -> dict:
        """Parse imaging list ``.lis`` and extract column 24: angle at focal surface [deg]."""
        angles: list[float] = []
        with photons_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 24:
                    continue
                try:
                    # Column 24
                    angles.append(float(parts[23]))
                except ValueError:
                    continue
        return {"angle_incidence_focal_deg": angles}

    def _save_results(self) -> None:
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to save")
            return
        output_file = self.output_dir / f"incident_angles_{self.label}.ecsv"
        self.results.write(output_file, format="ascii.ecsv", overwrite=True)

    def export_results(self) -> None:
        """Export the results ECSV and a short text summary file."""
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
