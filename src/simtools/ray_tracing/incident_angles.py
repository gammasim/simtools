"""Calculate photon incident angles on focal plane and primary/secondary mirrors.

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
    """Run a PSF-style sim_telarray job and compute incident angles at mirrors or focal surfaces.

    Parameters
    ----------
    simtel_path : str or pathlib.Path
        Path to the sim_telarray installation directory (containing ``sim_telarray/bin``).
    db_config : dict or str or None
        Database configuration passed to ``initialize_simulation_models``.
    config_data : dict
        Simulation configuration (e.g. ``site``, ``telescope``, ``model_version``,
        ``off_axis_angle``, ``source_distance``, ``number_of_photons``).
    output_dir : str or pathlib.Path
        Output directory where logs, scripts, photons files and results are written.
    label : str, optional
        Label used to name outputs; defaults to ``incident_angles_<telescope>`` when omitted.

    Notes
    -----
    Additional options are read from ``config_data`` when present:
    - ``perfect_mirror`` (bool, default False)
    - ``mirror_reflection_random_angle`` (float, optional)
    - ``mirror_alignment_random`` (float, optional)
    - ``calculate_primary_secondary_angles`` (bool, default True)
    """

    def __init__(
        self,
        simtel_path,
        db_config,
        config_data,
        output_dir,
        label=None,
    ):
        self.logger = logging.getLogger(__name__)

        self._simtel_path = Path(simtel_path)
        self.config_data = config_data
        self.output_dir = Path(output_dir)
        self.label = label or f"incident_angles_{config_data['telescope']}"
        cfg = config_data
        self.perfect_mirror = cfg.get("perfect_mirror", False)
        self.mirror_reflection_random_angle = cfg.get("mirror_reflection_random_angle")
        self.mirror_alignment_random = cfg.get("mirror_alignment_random")
        self.calculate_primary_secondary_angles = cfg.get(
            "calculate_primary_secondary_angles", True
        )
        self.results = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.scripts_dir = self.output_dir / "scripts"
        self.photons_dir = self.output_dir / "photons_files"
        self.results_dir = self.output_dir / "incident_angles"
        for d in (self.logs_dir, self.scripts_dir, self.photons_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

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

    def __repr__(self):
        """Return a concise representation helpful for logging/debugging."""
        return f"IncidentAnglesCalculator(label={self.label})"

    def _label_suffix(self):
        """Build a filename suffix including telescope and off-axis angle."""
        tel = str(self.config_data.get("telescope", "TEL"))
        off = float(self.config_data.get("off_axis_angle", 0.0 * u.deg).to_value(u.deg))
        return f"{self.label}_{tel}_off{off:g}"

    def run(self):
        """Run sim_telarray, parse imaging list, and return an angle table."""
        self.logger.info("Running sim_telarray PSF-style simulation for incident angles")

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
            if "primary_hit_radius_m" in data:
                self.results["primary_hit_radius"] = data["primary_hit_radius_m"] * u.m
            if "secondary_hit_radius_m" in data:
                self.results["secondary_hit_radius"] = data["secondary_hit_radius_m"] * u.m
            if "primary_hit_x_m" in data and "primary_hit_y_m" in data:
                self.results["primary_hit_x"] = data["primary_hit_x_m"] * u.m
                self.results["primary_hit_y"] = data["primary_hit_y_m"] * u.m
            if "secondary_hit_x_m" in data and "secondary_hit_y_m" in data:
                self.results["secondary_hit_x"] = data["secondary_hit_x_m"] * u.m
                self.results["secondary_hit_y"] = data["secondary_hit_y_m"] * u.m

        self._save_results()
        return self.results

    def run_for_offsets(self, offsets):
        """Run the simulation for multiple off-axis angles.

        For each off-axis angle provided, run a full simulation, labeling output files
        accordingly. Returns a mapping from off-axis angle (deg) to the resulting QTable.
        """
        results_by_offset = {}
        base_off = self.config_data.get("off_axis_angle", 0.0 * u.deg)

        for off in offsets:
            self.config_data["off_axis_angle"] = float(off) * u.deg
            self.logger.info(f"Running for off-axis angle {off:g} deg with label {self.label}")
            tbl = self.run()
            results_by_offset[float(off)] = tbl.copy()

        self.config_data["off_axis_angle"] = base_off
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
            # Always write zenith angle as 20 deg for incident angle calculations
            pf.write("# zenith_angle [deg] = 20\n")
            pf.write(
                f"# off_axis_angle [deg] = {self.config_data['off_axis_angle'].to_value(u.deg)}\n"
            )
            pf.write(f"# source_distance [km] = {self.config_data['source_distance'].value}\n")

        with stars_file.open("w", encoding="utf-8") as sf:
            zen = 20
            dist = float(self.config_data["source_distance"].to_value(u.km))
            sf.write(f"0. {90.0 - zen} 1.0 {dist}\n")

        return photons_file, stars_file, log_file

    def _write_run_script(self, photons_file, stars_file, log_file):
        """Generate a run script for sim_telarray with the provided configuration and inputs."""
        script_path = self.scripts_dir / f"run_incident_angles_{self._label_suffix()}.sh"
        simtel_bin = self._simtel_path / "sim_telarray/bin/sim_telarray_debug_trace"
        corsika_dummy = self._simtel_path / "sim_telarray/run9991.corsika.gz"

        theta = 20
        off = float(self.config_data["off_axis_angle"].to_value(u.deg))
        star_photons = self.config_data["number_of_photons"]

        def cfg(par, val):
            return f"-C {par}={val}"

        opts = [
            f"-c {self.telescope_model.config_file_path}",
            f"-I{self.telescope_model.config_file_directory}",
        ]
        if self.perfect_mirror:
            opts += [
                "-DPERFECT_DISH=1",
                "-C telescope_random_angle=0",
                "-C telescope_random_error=0",
                "-C random_focal_length=0",
                "-C mirror_reflection_random_angle=0",
                "-C mirror_align_random_distance=0",
                "-C mirror_align_random_horizontal=0,28,0,0",
                "-C mirror_align_random_vertical=0,28,0,0",
            ]

        if self.mirror_reflection_random_angle is not None:
            opts.append(cfg("mirror_reflection_random_angle", self.mirror_reflection_random_angle))
        if self.mirror_alignment_random is not None:
            opts.append(
                cfg("mirror_align_random_horizontal", f"{self.mirror_alignment_random},28.,0.0,0.0")
            )
            opts.append(
                cfg("mirror_align_random_vertical", f"{self.mirror_alignment_random},28.,0.0,0.0")
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

    def _run_script(self, script_path, log_file):
        """Execute the script and log output; raise an error if execution fails."""
        self.logger.info("Executing %s (logging to %s)", script_path, log_file)
        try:
            subprocess.check_call([str(script_path)])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Incident angles run failed, see log: {log_file}") from exc

    def _compute_incidence_angles_from_imaging_list(self, photons_file):
        """
        Compute incidence angles from imaging list.

        Column positions may differ between telescope types and sim_telarray builds.
        Parse header lines ("#   Column N: ...") to discover indices; otherwise
        fall back to legacy positions (1-based): focal=26, primary=32, secondary=36,
        primary X/Y = 29/30, secondary X/Y = 33/34.
        """
        (
            focal_idx,
            primary_idx,
            secondary_idx,
            primary_x_idx,
            primary_y_idx,
            secondary_x_idx,
            secondary_y_idx,
        ) = self._discover_column_indices(photons_file)

        col_idx = {
            "focal": focal_idx,
            "primary": primary_idx,
            "secondary": secondary_idx,
            "prim_x": primary_x_idx,
            "prim_y": primary_y_idx,
            "sec_x": secondary_x_idx,
            "sec_y": secondary_y_idx,
        }

        focal = []
        primary = [] if self.calculate_primary_secondary_angles else None
        secondary = [] if self.calculate_primary_secondary_angles else None
        # Primary-hit radius on M1 (computed from hit x/y in cm -> meters)
        radius_m = [] if self.calculate_primary_secondary_angles else None
        # Secondary-hit radius on M2 (computed from hit x/y in cm -> meters)
        secondary_radius_m = [] if self.calculate_primary_secondary_angles else None
        # Hit coordinates (meters), derived from cm columns
        primary_hit_x_m = [] if self.calculate_primary_secondary_angles else None
        primary_hit_y_m = [] if self.calculate_primary_secondary_angles else None
        secondary_hit_x_m = [] if self.calculate_primary_secondary_angles else None
        secondary_hit_y_m = [] if self.calculate_primary_secondary_angles else None

        for parts in self._iter_data_rows(photons_file):
            self._append_values(
                parts,
                col_idx,
                focal,
                primary,
                secondary,
                radius_m,
                secondary_radius_m,
                primary_hit_x_m,
                primary_hit_y_m,
                secondary_hit_x_m,
                secondary_hit_y_m,
            )

        result = {"angle_incidence_focal_deg": focal}
        if self.calculate_primary_secondary_angles:
            result["angle_incidence_primary_deg"] = primary
            result["angle_incidence_secondary_deg"] = secondary
            result["primary_hit_radius_m"] = radius_m
            result["secondary_hit_radius_m"] = secondary_radius_m
            result["primary_hit_x_m"] = primary_hit_x_m
            result["primary_hit_y_m"] = primary_hit_y_m
            result["secondary_hit_x_m"] = secondary_hit_x_m
            result["secondary_hit_y_m"] = secondary_hit_y_m
        return result

    def _discover_column_indices(self, photons_file):
        """Return 0-based indices discovered from headers.

        Returns a tuple: (focal_idx, primary_idx, secondary_idx,
        primary_x_idx, primary_y_idx, secondary_x_idx, secondary_y_idx)

        Defaults (1-based) are focal=26, primary=32, secondary=36,
        primary X/Y = 29/30, secondary X/Y = 33/34.
        """
        indices = self._default_column_indices()

        col_pat = re.compile(r"^\s*#\s*Column\s+(\d{1,4})\s*:(.*)$", re.IGNORECASE)
        with photons_file.open("r", encoding="utf-8") as fh:
            for raw in fh:
                s = raw.strip()
                if not s or not s.startswith("#"):
                    continue
                m = col_pat.match(s)
                if not m:
                    continue
                num = int(m.group(1))
                desc = m.group(2).strip().lower()
                self._update_indices_from_header_desc(desc, num, indices)

        return (
            indices["focal"],
            indices.get("primary"),
            indices.get("secondary"),
            indices.get("prim_x"),
            indices.get("prim_y"),
            indices.get("sec_x"),
            indices.get("sec_y"),
        )

    def _default_column_indices(self):
        """Return default 0-based indices matching legacy positions.

        Fallbacks (1-based): focal=26, primary=32, secondary=36,
        primary X/Y=29/30, secondary X/Y=33/34.
        """
        idx = {"focal": 25}
        if self.calculate_primary_secondary_angles:
            idx.update(
                {
                    "primary": 31,
                    "secondary": 35,
                    "prim_x": 28,
                    "prim_y": 29,
                    "sec_x": 32,
                    "sec_y": 33,
                }
            )
        return idx

    def _update_indices_from_header_desc(self, desc, num, indices):
        """Update indices dict in-place based on a header description and column number."""
        # Angles
        if "angle of incidence" in desc:
            if "focal surface" in desc:
                indices["focal"] = num - 1
                return
            if self.calculate_primary_secondary_angles:
                if "primary mirror" in desc:
                    indices["primary"] = num - 1
                    return
                if "secondary mirror" in desc:
                    indices["secondary"] = num - 1
                    return
        # Reflection points (X/Y)
        if not self.calculate_primary_secondary_angles or "reflection point" not in desc:
            return
        self._maybe_set_reflection_index(desc, num, indices)

    @staticmethod
    def _contains_axis(desc, axis):
        """Return True if desc contains stand-alone axis label ('x' or 'y')."""
        return bool(re.search(r"(^|\s)" + re.escape(axis) + r"(\s|$)", desc))

    def _maybe_set_reflection_index(self, desc, num, indices):
        """Set reflection point indices for primary/secondary mirrors if the header matches."""
        is_primary = "primary mirror" in desc
        is_secondary = "secondary mirror" in desc
        if not is_primary and not is_secondary:
            return
        is_x = self._contains_axis(desc, "x")
        is_y = self._contains_axis(desc, "y")
        if not (is_x or is_y):
            return
        key_prefix = "prim" if is_primary else "sec"
        key = f"{key_prefix}_{'x' if is_x else 'y'}"
        indices[key] = num - 1

    @staticmethod
    def _parse_float(parts, idx):
        """Try parse float from parts[idx]. Returns (ok, value)."""
        if idx is None or idx < 0 or idx >= len(parts):
            return False, 0.0
        try:
            return True, float(parts[idx])
        except ValueError:
            return False, 0.0

    @staticmethod
    def _parse_float_with_nan(parts, idx):
        """Parse float or return NaN when missing/invalid."""
        if idx is None or idx < 0 or idx >= len(parts):
            return float("nan")
        try:
            return float(parts[idx])
        except ValueError:
            return float("nan")

    @staticmethod
    def _iter_data_rows(photons_file):
        """Yield tokenized, non-empty, non-comment rows from imaging list."""
        with photons_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                yield line.split()

    def _append_values(
        self,
        parts,
        col_idx,
        focal,
        primary,
        secondary,
        radius_m,
        secondary_radius_m,
        primary_hit_x_m,
        primary_hit_y_m,
        secondary_hit_x_m,
        secondary_hit_y_m,
    ):
        """Append parsed values from parts into target arrays if valid."""
        foc_ok, foc_val = self._parse_float(parts, col_idx.get("focal"))
        if not foc_ok:
            return
        focal.append(foc_val)
        if not self.calculate_primary_secondary_angles:
            return

        self._append_primary_secondary_angles(parts, col_idx, primary, secondary)
        self._append_primary_hit_geometry(
            parts, col_idx, radius_m, primary_hit_x_m, primary_hit_y_m
        )
        self._append_secondary_hit_geometry(
            parts, col_idx, secondary_radius_m, secondary_hit_x_m, secondary_hit_y_m
        )

    def _append_primary_secondary_angles(self, parts, col_idx, primary, secondary):
        """Append primary/secondary angle values (or NaN) if arrays are provided."""
        if primary is not None:
            primary.append(self._parse_float_with_nan(parts, col_idx.get("primary")))
        if secondary is not None:
            secondary.append(self._parse_float_with_nan(parts, col_idx.get("secondary")))

    def _append_primary_hit_geometry(
        self, parts, col_idx, radius_m, primary_hit_x_m, primary_hit_y_m
    ):
        """Append primary-mirror hit geometry (radius and x/y in meters)."""
        x_ok, x_cm = self._parse_float(parts, col_idx.get("prim_x"))
        y_ok, y_cm = self._parse_float(parts, col_idx.get("prim_y"))
        if x_ok and y_ok:
            r_m = ((x_cm**2 + y_cm**2) ** 0.5) / 100.0
            if radius_m is not None:
                radius_m.append(r_m)
            if primary_hit_x_m is not None:
                primary_hit_x_m.append(x_cm / 100.0)
            if primary_hit_y_m is not None:
                primary_hit_y_m.append(y_cm / 100.0)
            return
        if radius_m is not None:
            radius_m.append(float("nan"))
        if primary_hit_x_m is not None:
            primary_hit_x_m.append(float("nan"))
        if primary_hit_y_m is not None:
            primary_hit_y_m.append(float("nan"))

    def _append_secondary_hit_geometry(
        self, parts, col_idx, secondary_radius_m, secondary_hit_x_m, secondary_hit_y_m
    ):
        """Append secondary-mirror hit geometry (radius and x/y in meters)."""
        sx_ok, sx_cm = self._parse_float(parts, col_idx.get("sec_x"))
        sy_ok, sy_cm = self._parse_float(parts, col_idx.get("sec_y"))
        if sx_ok and sy_ok:
            r2_m = ((sx_cm**2 + sy_cm**2) ** 0.5) / 100.0
            if secondary_radius_m is not None:
                secondary_radius_m.append(r2_m)
            if secondary_hit_x_m is not None:
                secondary_hit_x_m.append(sx_cm / 100.0)
            if secondary_hit_y_m is not None:
                secondary_hit_y_m.append(sy_cm / 100.0)
            return
        if secondary_radius_m is not None:
            secondary_radius_m.append(float("nan"))
        if secondary_hit_x_m is not None:
            secondary_hit_x_m.append(float("nan"))
        if secondary_hit_y_m is not None:
            secondary_hit_y_m.append(float("nan"))

    @staticmethod
    def _match_header_column(col_pat, raw):
        """Return (kind, column_number) if the header line defines a known angle column."""
        s = raw.strip()
        if s and ":" in s:
            prefix, desc = s.split(":", 1)
            m = col_pat.match(prefix)
            if m:
                num = int(m.group(1))
                desc = desc.strip().lower()
                if "angle of incidence at focal surface" in desc and "optical axis" in desc:
                    return "focal", num
                if re.search(r"angle of incidence\s+on(to)?\s+primary mirror", desc):
                    return "primary", num
                if re.search(r"angle of incidence\s+on(to)?\s+secondary mirror", desc):
                    return "secondary", num
        return None

    def _save_results(self):
        """Save the results to an ECSV file in the results directory if available."""
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to save")
            return
        output_file = self.results_dir / f"incident_angles_{self._label_suffix()}.ecsv"
        self.results.write(output_file, format="ascii.ecsv", overwrite=True)

    def export_results(self):
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
            f.write("Zenith angle: 20\n")
            f.write(f"Off-axis angle: {self.config_data['off_axis_angle']}\n")
            f.write(f"Source distance: {self.config_data['source_distance']}\n\n")
            f.write(f"Number of data points: {len(self.results)}\n")
