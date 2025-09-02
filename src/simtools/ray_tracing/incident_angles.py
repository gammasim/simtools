"""Calculate incident angles using a sim_telarray PSF-style run.

Parses the imaging list (``.lis``) produced by sim_telarray_debug_trace and uses
Angle of incidence at focal surface, with respect to the optical axis [deg],
Angle of incidence on(to) primary mirror [deg], and
Angle of incidence on(to) secondary mirror [deg] (if available).
"""

import logging
import re
import subprocess
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

from simtools.model.model_utils import initialize_simulation_models


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
        calculate_primary_secondary_angles: bool = True,
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
        self.calculate_primary_secondary_angles = calculate_primary_secondary_angles
        self.results: QTable | None = None

        # Create output directory tree
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.scripts_dir = self.output_dir / "scripts"
        self.photons_dir = self.output_dir / "photons_files"
        self.results_dir = self.output_dir / "incident_angles"
        for d in (self.logs_dir, self.scripts_dir, self.photons_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

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

    def _label_suffix(self) -> str:
        """Build a filename suffix including telescope and off-axis angle."""
        tel = str(self.config_data.get("telescope", "TEL"))
        off = float(self.rt_params.get("off_axis_angle", 0.0 * u.deg).to_value(u.deg))
        return f"{self.label}_{tel}_off{off:g}"

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
        if self.calculate_primary_secondary_angles:
            if "angle_incidence_primary_deg" in data:
                self.results["angle_incidence_primary"] = (
                    data["angle_incidence_primary_deg"] * u.deg
                )
            if "angle_incidence_secondary_deg" in data:
                self.results["angle_incidence_secondary"] = (
                    data["angle_incidence_secondary_deg"] * u.deg
                )

        self._save_results()
        return self.results

    def run_for_offsets(self, offsets: list[float]) -> dict[float, QTable]:
        """Run the simulation for multiple off-axis angles.

        For each off-axis angle provided, run a full simulation, labeling output files
        accordingly. Returns a mapping from off-axis angle (deg) to the resulting QTable.
        """
        results_by_offset: dict[float, QTable] = {}
        base_off = self.rt_params.get("off_axis_angle", 0.0 * u.deg)

        for off in offsets:
            self.rt_params["off_axis_angle"] = float(off) * u.deg
            self.logger.info(f"Running for off-axis angle {off:g} deg with label {self.label}")
            tbl = self.run()
            results_by_offset[float(off)] = tbl.copy()

        self.rt_params["off_axis_angle"] = base_off
        return results_by_offset

    def _prepare_psf_io_files(self):
        """Prepare photons, stars, and log file paths for a PSF-style incident angle simulation."""
        suffix = self._label_suffix()
        photons_file = self.photons_dir / f"incident_angles_photons_{suffix}.lis"
        stars_file = self.photons_dir / f"incident_angles_stars_{suffix}.lis"
        log_file = self.logs_dir / f"incident_angles_{suffix}.log"

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
        """Generate a run script for sim_telarray with the provided configuration and inputs."""
        script_path = self.scripts_dir / f"run_incident_angles_{self._label_suffix()}.sh"
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
        """Execute the script and log output; raise an error if execution fails."""
        self.logger.info("Executing %s (logging to %s)", script_path, log_file)
        try:
            subprocess.check_call([str(script_path)])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Incident angles run failed, see log: {log_file}") from exc

    def _compute_incidence_angles_from_imaging_list(self, photons_file: Path) -> dict:
        """
        Compute incidence angles from imaging list.

        Column positions may differ between telescope types and sim_telarray builds.
        Parse header lines ("#   Column N: ...") to discover indices; otherwise
        fall back to legacy positions (1-based): focal=26, primary=32, secondary=36.
        """
        # Defaults (1-based)
        focal_idx_1b = 26
        primary_idx_1b = 32
        secondary_idx_1b = 36

        # Scan file for header mapping lines
        col_pat = re.compile(r"^\s*#\s*Column\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)
        with photons_file.open("r", encoding="utf-8") as fh:
            for raw in fh:
                s = raw.strip()
                if not s:
                    continue
                m = col_pat.match(s)
                if not m:
                    continue
                num = int(m.group(1))
                desc = m.group(2).strip().lower()
                if "angle of incidence at focal surface" in desc and "optical axis" in desc:
                    focal_idx_1b = num
                elif re.search(r"angle of incidence\s+on(to)?\s+primary mirror", desc):
                    primary_idx_1b = num
                elif re.search(r"angle of incidence\s+on(to)?\s+secondary mirror", desc):
                    secondary_idx_1b = num

        self.logger.info(
            "Imaging list columns (1-based): focal=%s primary=%s secondary=%s",
            focal_idx_1b,
            (primary_idx_1b if self.calculate_primary_secondary_angles else "n/a"),
            (secondary_idx_1b if self.calculate_primary_secondary_angles else "n/a"),
        )

        # Convert to 0-based
        focal_idx = focal_idx_1b - 1
        primary_idx = primary_idx_1b - 1 if self.calculate_primary_secondary_angles else None
        secondary_idx = secondary_idx_1b - 1 if self.calculate_primary_secondary_angles else None

        focal: list[float] = []
        primary: list[float] | None = [] if self.calculate_primary_secondary_angles else None
        secondary: list[float] | None = [] if self.calculate_primary_secondary_angles else None

        with photons_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) <= focal_idx:
                    continue
                try:
                    foc_val = float(parts[focal_idx])
                except ValueError:
                    continue
                focal.append(foc_val)

                if self.calculate_primary_secondary_angles:
                    if primary is not None:
                        if primary_idx is not None and len(parts) > primary_idx:
                            try:
                                primary.append(float(parts[primary_idx]))
                            except ValueError:
                                primary.append(float("nan"))
                        else:
                            primary.append(float("nan"))
                    if secondary is not None:
                        if secondary_idx is not None and len(parts) > secondary_idx:
                            try:
                                secondary.append(float(parts[secondary_idx]))
                            except ValueError:
                                secondary.append(float("nan"))
                        else:
                            secondary.append(float("nan"))

        result = {"angle_incidence_focal_deg": focal}
        if self.calculate_primary_secondary_angles:
            result["angle_incidence_primary_deg"] = primary
            result["angle_incidence_secondary_deg"] = secondary
        return result

    def _save_results(self) -> None:
        """Save the results to an ECSV file in the results directory if available."""
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to save")
            return
        output_file = self.results_dir / f"incident_angles_{self._label_suffix()}.ecsv"
        self.results.write(output_file, format="ascii.ecsv", overwrite=True)

    def export_results(self) -> None:
        """Export the results ECSV and a short text summary file."""
        if self.results is None or len(self.results) == 0:
            self.logger.error("Cannot export results because they do not exist")
            return
        table_file = self.results_dir / f"incident_angles_{self._label_suffix()}.ecsv"
        self.results.write(table_file, format="ascii.ecsv", overwrite=True)
        summary_file = self.results_dir / f"incident_angles_summary_{self._label_suffix()}.txt"
        with summary_file.open("w", encoding="utf-8") as f:
            f.write(f"Incident angle results for {self.telescope_model.name}\n")
            f.write(f"Site: {self.telescope_model.site}\n")
            f.write(f"Zenith angle: {self.rt_params['zenith_angle']}\n")
            f.write(f"Off-axis angle: {self.rt_params['off_axis_angle']}\n")
            f.write(f"Source distance: {self.rt_params['source_distance']}\n\n")
            f.write(f"Number of data points: {len(self.results)}\n")
