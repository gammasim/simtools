"""Calculate photon incident angles on focal plane and primary/secondary mirrors.

Parses the imaging list (``.lis``) produced by sim_telarray_debug_trace and uses
Angle of incidence at focal surface, with respect to the optical axis [deg],
Angle of incidence on to primary mirror [deg], and
Angle of incidence on to secondary mirror [deg] (if available).
"""

import logging
import math
import re
import subprocess
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

from simtools.data_model.metadata_collector import MetadataCollector
from simtools.model.model_utils import initialize_simulation_models


class IncidentAnglesCalculator:
    """Run a PSF-style sim_telarray job and compute incident angles at mirrors or focal surfaces.

    Parameters
    ----------
    simtel_path : str or pathlib.Path
        Path to the sim_telarray installation directory (containing ``sim_telarray/bin``).
    db_config : dict
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
    - ``calculate_primary_secondary_angles`` (bool, default True)
    """

    # Use fixed zenith angle (degrees) for incident-angle simulations.
    ZENITH_ANGLE_DEG = 0

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
        self.telescope_model, self.site_model, _ = initialize_simulation_models(
            label=self.label,
            db_config=db_config,
            site=config_data["site"],
            telescope_name=config_data["telescope"],
            model_version=config_data["model_version"],
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
        """Run sim_telarray, parse the imaging list, and return an angle table.

        Returns
        -------
        astropy.table.QTable
            Table containing at least the ``angle_incidence_focal`` column
            and, when configured, primary/secondary angles and hit geometry.
        """
        self.telescope_model.write_sim_telarray_config_file(additional_models=self.site_model)

        photons_file, stars_file, log_file = self._prepare_psf_io_files()
        run_script = self._write_run_script(photons_file, stars_file, log_file)
        self._run_script(run_script, log_file)

        data = self._compute_incidence_angles_from_imaging_list(photons_file)
        self.results = QTable()
        self.results["angle_incidence_focal"] = data["angle_incidence_focal_deg"] * u.deg
        if self.calculate_primary_secondary_angles:
            field_map = {
                "angle_incidence_primary_deg": ("angle_incidence_primary", u.deg),
                "angle_incidence_secondary_deg": ("angle_incidence_secondary", u.deg),
                "primary_hit_radius_m": ("primary_hit_radius", u.m),
                "secondary_hit_radius_m": ("secondary_hit_radius", u.m),
                "primary_hit_x_m": ("primary_hit_x", u.m),
                "primary_hit_y_m": ("primary_hit_y", u.m),
                "secondary_hit_x_m": ("secondary_hit_x", u.m),
                "secondary_hit_y_m": ("secondary_hit_y", u.m),
            }
            for key, (name, unit) in field_map.items():
                if key in data:
                    self.results[name] = data[key] * unit

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

    def _prepare_psf_io_files(self):
        """Prepare photons, stars, and log file paths for a PSF-style incident angle simulation.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path, pathlib.Path]
            Paths to the photons file, stars file, and log file.
        """
        suffix = self._label_suffix()
        photons_file = self.photons_dir / f"incident_angles_photons_{suffix}.lis"
        stars_file = self.photons_dir / f"incident_angles_stars_{suffix}.lis"
        log_file = self.logs_dir / f"incident_angles_{suffix}.log"

        if photons_file.exists():
            try:
                photons_file.unlink()
            except OSError as err:
                self.logger.error(f"Failed to remove existing photons file {photons_file}: {err}")

        with photons_file.open("w", encoding="utf-8") as pf:
            pf.write(f"#{'=' * 50}\n")
            pf.write("# Imaging list for Incident Angle simulations\n")
            pf.write(f"#{'=' * 50}\n")
            pf.write(f"# config_file = {self.telescope_model.config_file_path}\n")
            pf.write(f"# zenith_angle [deg] = {self.ZENITH_ANGLE_DEG}\n")
            pf.write(
                f"# off_axis_angle [deg] = {self.config_data['off_axis_angle'].to_value(u.deg)}\n"
            )
            pf.write(f"# source_distance [km] = {self.config_data['source_distance']}\n")

        with stars_file.open("w", encoding="utf-8") as sf:
            zen = self.ZENITH_ANGLE_DEG
            dist = float(self.config_data["source_distance"])
            sf.write(f"0. {90.0 - zen} 1.0 {dist}\n")

        return photons_file, stars_file, log_file

    def _write_run_script(self, photons_file, stars_file, log_file):
        """Generate a run script for sim_telarray with the provided configuration and inputs.

        Parameters
        ----------
        photons_file, stars_file, log_file : pathlib.Path
            Input/output files for the run.

        Returns
        -------
        pathlib.Path
            Path to the generated shell script.
        """
        script_path = self.scripts_dir / f"run_incident_angles_{self._label_suffix()}.sh"
        simtel_bin = self._simtel_path / "sim_telarray/bin/sim_telarray_debug_trace"
        corsika_dummy = self._simtel_path / "sim_telarray/run9991.corsika.gz"

        theta = self.ZENITH_ANGLE_DEG
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
                "-C random_focal_length=0",
                "-C mirror_reflection_random_angle=0",
                "-C mirror_align_random_distance=0",
                "-C mirror_align_random_horizontal=0,28,0,0",
                "-C mirror_align_random_vertical=0,28,0,0",
            ]

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
        """Execute the script and log output; raise an error if execution fails.

        Parameters
        ----------
        script_path : pathlib.Path
            Path to the script to execute.
        log_file : pathlib.Path
            Destination log file.
        """
        self.logger.info("Executing %s (logging to %s)", script_path, log_file)
        try:
            subprocess.check_call([str(script_path)])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Incident angles run failed, see log: {log_file}") from exc

    def _compute_incidence_angles_from_imaging_list(self, photons_file):
        """Compute incidence angles from an imaging list file.

        Column positions may differ between telescope types and sim_telarray versions.
        Header lines (``# Column N: ...``) are parsed to find indices; otherwise
        legacy positions (1-based) are used: focal=26, primary=32, secondary=36,
        primary X/Y = 29/30, secondary X/Y = 33/34.

        Parameters
        ----------
        photons_file : pathlib.Path
            Path to the imaging list file (``.lis``).

        Returns
        -------
        dict[str, list[float]]
            Parsed columns in degrees/meters as plain Python lists. Always
            contains ``angle_incidence_focal_deg``; additional keys are present
            when primary/secondary angles are enabled.
        """
        col_idx = self._find_column_indices(photons_file)

        focal = []
        # Initialize optional arrays once based on the configuration
        primary = secondary = radius_m = secondary_radius_m = None
        primary_hit_x_m = primary_hit_y_m = secondary_hit_x_m = secondary_hit_y_m = None
        if self.calculate_primary_secondary_angles:
            primary, secondary = [], []
            radius_m, secondary_radius_m = [], []
            primary_hit_x_m, primary_hit_y_m = [], []
            secondary_hit_x_m, secondary_hit_y_m = [], []

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

    def _find_column_indices(self, photons_file):
        """Return 0-based column indices found from headers as a dict.

        Returns a mapping with keys ``'focal'`` and, when applicable, ``'primary'``,
        ``'secondary'``, ``'prim_x'``, ``'prim_y'``, ``'sec_x'``, ``'sec_y'``.

        Parameters
        ----------
        photons_file : pathlib.Path
            Imaging list file whose headers may define column numbers.

        Returns
        -------
        dict[str, int]
            0-based indices for the required columns.
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

        return indices

    def _default_column_indices(self):
        """Return default 0-based indices matching SST-like photon files.

        Fallbacks (1-based): focal=26, primary=32, secondary=36,
        primary X/Y=29/30, secondary X/Y=33/34.

        Returns
        -------
        dict[str, int]
            Default index mapping. When primary/secondary angles are disabled,
            only ``'focal'`` is included.
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
        """Update indices dict in-place based on a header description and column number.

        Parameters
        ----------
        desc : str
            Header description text (lower-cased).
        num : int
            1-based column number from the header.
        indices : dict[str, int]
            Mapping to update in-place.
        """
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
        self._set_reflection_index_if_match(desc, num, indices)

    @staticmethod
    def _contains_axis(desc, axis):
        """Check whether a description contains a stand-alone axis label.

        Parameters
        ----------
        desc : str
            Header description string.
        axis : str
            Either ``"x"`` or ``"y"``.

        Returns
        -------
        bool
            True when the token is present as a separate word; False otherwise.
        """
        desc_l = desc.lower()
        axis_l = axis.lower()
        return bool(re.search(r"(^|\s)" + re.escape(axis_l) + r"(\s|$)", desc_l))

    def _set_reflection_index_if_match(self, desc, num, indices):
        """Set reflection point indices for primary/secondary mirrors if the header matches.

        Parameters
        ----------
        desc : str
            Header description string (lower-cased).
        num : int
            1-based column number from the header.
        indices : dict[str, int]
            Mapping to update in-place with keys ``prim_x``, ``prim_y``,
            ``sec_x``, or ``sec_y``.
        """
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
        """Try to parse a float from ``parts[idx]``.

        Parameters
        ----------
        parts : list[str]
            Tokenized row.
        idx : int | None
            Index into ``parts``.

        Returns
        -------
        tuple[bool, float]
            Tuple of ``(ok, value)`` where ``ok`` is False when parsing fails
            or the index is out of range; in that case ``value`` is 0.0.
        """
        if idx is None or idx < 0 or idx >= len(parts):
            return False, 0.0
        try:
            return True, float(parts[idx])
        except ValueError:
            return False, 0.0

    @staticmethod
    def _parse_float_with_nan(parts, idx):
        """Parse a float or return NaN when missing/invalid.

        Parameters
        ----------
        parts : list[str]
            Tokenized row.
        idx : int | None
            Index into ``parts``.

        Returns
        -------
        float
            Parsed float value, or ``nan`` when unavailable/invalid.
        """
        if idx is None or idx < 0 or idx >= len(parts):
            return float("nan")
        try:
            return float(parts[idx])
        except ValueError:
            return float("nan")

    @staticmethod
    def _iter_data_rows(photons_file):
        """Iterate over tokenized, non-empty, non-comment rows.

        Parameters
        ----------
        photons_file : pathlib.Path
            Imaging list file to read.

        Returns
        -------
        Iterator[list[str]]
            Iterator over tokenized rows.
        """
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
        """Append parsed values from parts into target arrays if valid.

        Parameters
        ----------
        parts : list[str]
            Tokenized input row.
        col_idx : dict[str, int]
            Column indices used to read values.
        focal, primary, secondary : list | None
            Output arrays to append into.
        radius_m, secondary_radius_m : list | None
            Output arrays for radii in meters.
        primary_hit_x_m, primary_hit_y_m, secondary_hit_x_m, secondary_hit_y_m : list | None
            Output arrays for hit coordinates in meters.
        """
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
        """Append primary/secondary angle values (or NaN) if arrays are provided.

        Parameters
        ----------
        parts : list[str]
            Tokenized input row.
        col_idx : dict[str, int]
            Indices for angle columns.
        primary, secondary : list | None
            Output arrays to append into.
        """
        if primary is not None:
            primary.append(self._parse_float_with_nan(parts, col_idx.get("primary")))
        if secondary is not None:
            secondary.append(self._parse_float_with_nan(parts, col_idx.get("secondary")))

    def _append_primary_hit_geometry(
        self, parts, col_idx, radius_m, primary_hit_x_m, primary_hit_y_m
    ):
        """Append primary-mirror hit geometry (radius and x/y in meters).

        Parameters
        ----------
        parts : list[str]
            Tokenized input row.
        col_idx : dict[str, int]
            Indices for hit coordinate columns.
        radius_m, primary_hit_x_m, primary_hit_y_m : list | None
            Output arrays to append into.
        """
        x_ok, x_cm = self._parse_float(parts, col_idx.get("prim_x"))
        y_ok, y_cm = self._parse_float(parts, col_idx.get("prim_y"))
        if x_ok and y_ok:
            x_m, y_m = x_cm / 100.0, y_cm / 100.0
            r_m = math.hypot(x_cm, y_cm) / 100.0
        else:
            x_m = y_m = r_m = math.nan

        if radius_m is not None:
            radius_m.append(r_m)
        if primary_hit_x_m is not None:
            primary_hit_x_m.append(x_m)
        if primary_hit_y_m is not None:
            primary_hit_y_m.append(y_m)

    def _append_secondary_hit_geometry(
        self, parts, col_idx, secondary_radius_m, secondary_hit_x_m, secondary_hit_y_m
    ):
        """Append secondary-mirror hit geometry (radius and x/y in meters).

        Parameters
        ----------
        parts : list[str]
            Tokenized input row.
        col_idx : dict[str, int]
            Indices for hit coordinate columns.
        secondary_radius_m, secondary_hit_x_m, secondary_hit_y_m : list | None
            Output arrays to append into.
        """
        sx_ok, sx_cm = self._parse_float(parts, col_idx.get("sec_x"))
        sy_ok, sy_cm = self._parse_float(parts, col_idx.get("sec_y"))
        if sx_ok and sy_ok:
            x_m, y_m = sx_cm / 100.0, sy_cm / 100.0
            r_m = math.hypot(sx_cm, sy_cm) / 100.0
        else:
            x_m = y_m = r_m = math.nan

        if secondary_radius_m is not None:
            secondary_radius_m.append(r_m)
        if secondary_hit_x_m is not None:
            secondary_hit_x_m.append(x_m)
        if secondary_hit_y_m is not None:
            secondary_hit_y_m.append(y_m)

    @staticmethod
    def _match_header_column(col_pat, raw):
        """Parse a header line for a known angle column.

        Parameters
        ----------
        col_pat : Pattern[str]
            Compiled regular expression matching ``# Column N`` prefix.
        raw : str
            Raw header line.

        Returns
        -------
        tuple[str, int] | None
            ``(kind, column_number)`` when recognized, otherwise ``None``.
        """
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
        """Save the results to an ECSV file with metadata."""
        if self.results is None or len(self.results) == 0:
            self.logger.warning("No results to save")
            return
        output_file = self.results_dir / f"incident_angles_{self._label_suffix()}.ecsv"
        self.results.write(output_file, format="ascii.ecsv", overwrite=True)

        MetadataCollector.dump(
            args_dict=self.config_data,
            output_file=output_file.with_suffix(".yml"),
        )
