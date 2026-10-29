"""Ray tracing with sim_telarray.

sim_telarray simulates the star light of each random instrument instance in one
run and writes photon lists; images are read from these lists (or analysed with
sim_telarray's rx). Incidence angles come from the imaging list of the
sim_telarray debug-trace build.
"""

import gzip
import logging
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from simtools.context import RunContext
from simtools.job_execution import job_manager
from simtools.ray_tracing import parameter_changes
from simtools.ray_tracing.backends.interface import image_file_name
from simtools.ray_tracing.psf_analysis import PSFImage
from simtools.simtel.simulator_ray_tracing import SimulatorRayTracing, telescope_pointing


class SimtelBackend:
    """
    Ray tracing with sim_telarray (see ``simtools.ray_tracing.backends.interface``).

    Parameters
    ----------
    telescope_model: TelescopeModel
        Telescope model.
    site_model: SiteModel
        Site model.
    label: str
        Label of the output files.
    zenith_angle: float
        Telescope zenith angle [deg].
    single_mirror_mode: bool
        Trace single mirrors.
    """

    # Calibration light: sim_telarray's LightEmission programs and optics.
    traces_light_sources = False

    def __init__(
        self, telescope_model, site_model, label, zenith_angle, single_mirror_mode, *, context=None
    ):
        self._logger = logging.getLogger(__name__)
        self.context = context or RunContext.from_environment()
        self.telescope_model = telescope_model
        self.site_model = site_model
        self.label = label
        self.zenith_angle = zenith_angle
        self.single_mirror_mode = single_mirror_mode
        self._output_directory = self.context.paths.get_output_directory()

    def simulate(self, requests, test=False, force=False, compress_photons=True, max_workers=8):
        """
        Run one sim_telarray simulation per image; photon lists are written to disk.

        Independent full-telescope simulations run concurrently. Single-mirror
        simulations run serially because their temporary camera files are shared.

        Parameters
        ----------
        requests: list of StarImageRequest
            Images to trace.
        test: bool
            Test mode (single short run).
        force: bool
            Simulate again even if photon lists exist.
        compress_photons: bool
            Compress the photon lists (gzip).
        max_workers: int
            Maximum number of concurrent full-telescope simulations.
        """
        if not self.single_mirror_mode:
            self.telescope_model.write_sim_telarray_config_file(
                additional_models=self.site_model, label=self.label
            )
        simulations = [(self._simulator(request, test, force), request) for request in requests]
        if not self.single_mirror_mode and len(simulations) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(simulations))) as executor:
                list(
                    executor.map(
                        lambda simulation: self._run(simulation, test, compress_photons),
                        simulations,
                    )
                )
            return
        for simulation in simulations:
            self._run(simulation, test, compress_photons)

    def _simulator(self, request, test, force):
        """Create the simulator of one image."""
        self._logger.info(
            f"Simulating RayTracing for off_axis=({request.off_axis_x:.3f}, "
            f"{request.off_axis_y:.3f}), mirror={request.mirror_number}"
        )
        simulator = SimulatorRayTracing(
            telescope_model=self.telescope_model,
            site_model=self.site_model,
            label=self.label,
            test=test,
            config_data={
                "zenith_angle": self.zenith_angle,
                "off_axis_x": request.off_axis_x,
                "off_axis_y": request.off_axis_y,
                "source_distance": request.source_distance_km,
                "single_mirror_mode": self.single_mirror_mode,
                "mirror_numbers": request.mirror_number,
                "photons_per_run": request.photons_per_instance,
                "parameter_changes": request.parameter_changes,
                "seeds": request.seeds,
            },
            force_simulate=force,
        )
        if not self.single_mirror_mode:
            simulator._write_config = False  # pylint: disable=protected-access
        return simulator

    def _run(self, simulation, test, compress_photons):
        """Run one simulator and optionally compress its photon list."""
        simulator, request = simulation
        simulator.run(test=test)
        if not compress_photons:
            return
        photons_file = self._files(request)[0]
        self._logger.debug(f"Using gzip to compress the photons file {photons_file}.")
        with open(photons_file, "rb") as f_in:
            with gzip.open(photons_file.with_suffix(photons_file.suffix + ".gz"), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        photons_file.unlink()

    def psf_image(
        self, request, focal_length, containment_fraction, use_rx=False, save_photons=False
    ):
        """
        Return the PSF image of one simulated image.

        Parameters
        ----------
        request: StarImageRequest
            Image.
        focal_length: float
            Focal length [cm].
        containment_fraction: float
            PSF containment fraction.
        use_rx: bool
            Analyse the photon list with sim_telarray's rx.
        save_photons: bool
            Keep the photon and star lists; they are removed otherwise.

        Returns
        -------
        tuple of PSFImage and bool
            The image, and False (sim_telarray does not trace the telescope structure).
        """
        photons_file_lis, photons_file_gz, stars_file = self._files(request)
        photons_file = photons_file_gz
        if not (use_rx and photons_file_gz.exists()) and photons_file_lis.exists():
            photons_file = photons_file_lis
        image = PSFImage(
            focal_length=focal_length,
            containment_fraction=containment_fraction,
            context=self.context,
        )
        image.process_photon_list(photons_file, use_rx)
        if not save_photons:
            for file in (photons_file_lis, photons_file_gz, stars_file):
                file.unlink(missing_ok=True)
        return image, False

    def _files(self, request):
        """Photon list (and its compressed name) and star list of one image."""
        photons_file = self._output_directory / image_file_name(
            self, request, "ray_tracing_photons", ".lis"
        )
        return (
            photons_file,
            photons_file.with_suffix(photons_file.suffix + ".gz"),
            self._output_directory / image_file_name(self, request, "ray_tracing_stars", ".lis"),
        )

    def incidence_angles(self, request, work_directory, name, with_mirrors, keep_files=False):
        """
        Per-photon incidence angles from a sim_telarray debug-trace run.

        Parameters
        ----------
        request: StarImageRequest
            Star image (one instance; off_axis_x is the field angle).
        work_directory: Path
            Directory for logs, scripts and photon files.
        name: str
            Name of the run's files.
        with_mirrors: bool
            Also angles and hit points on the primary and secondary mirror.
        keep_files: bool
            Keep the imaging list (not reported as intermediate file).

        Returns
        -------
        tuple of dict and list
            Angles [deg] and hit points [m] (``angle_incidence_focal``, and with
            mirrors ``angle_incidence_primary``, ``angle_incidence_secondary``,
            ``primary_hit_x``, ``primary_hit_y``, ``secondary_hit_x``,
            ``secondary_hit_y``), and the intermediate files written.
        """
        self.telescope_model.write_sim_telarray_config_file(additional_models=self.site_model)
        directories = {d: work_directory / d for d in ("logs", "scripts", "photons_files")}
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        photons_file = directories["photons_files"] / f"incident_angles_photons_{name}.lis"
        stars_file = directories["scripts"] / f"incident_angles_stars_{name}.lis"
        log_file = directories["logs"] / f"incident_angles_{name}.log"
        script = directories["scripts"] / f"run_incident_angles_{name}.sh"
        self._write_incidence_inputs(request, photons_file, stars_file)
        self._write_incidence_script(request, script, photons_file, stars_file, log_file)
        self._logger.info("Executing %s (logging to %s)", script, log_file)
        try:
            job_manager.submit(script)
        except job_manager.JobExecutionError as exc:
            raise RuntimeError(f"Incident angles run failed, see log: {log_file}") from exc
        data = _ImagingList(with_mirrors).read(photons_file)
        intermediate = [script, stars_file] + ([] if keep_files else [photons_file])
        return data, intermediate

    def _write_incidence_inputs(self, request, photons_file, stars_file):
        """Write the imaging-list header and the star list."""
        with photons_file.open("w", encoding="utf-8") as file:
            file.write(f"#{'=' * 50}\n")
            file.write("# Imaging list for Incident Angle simulations\n")
            file.write(f"#{'=' * 50}\n")
            file.write(f"# config_file = {self.telescope_model.config_file_path}\n")
            file.write(f"# zenith_angle [deg] = {self.zenith_angle}\n")
            file.write(f"# off_axis_angle [deg] = {request.off_axis_x}\n")
            file.write(f"# source_distance [km] = {request.source_distance_km}\n")
        with stars_file.open("w", encoding="utf-8") as file:
            file.write(f"0. {90.0 - self.zenith_angle} 1.0 {request.source_distance_km}\n")

    def _write_incidence_script(self, request, script, photons_file, stars_file, log_file):
        """Write the run script of the sim_telarray debug-trace build."""
        theta, phi = telescope_pointing(self.zenith_angle, request.off_axis_x, request.off_axis_y)
        instrument_seed, simulation_seed = request.seeds[0]
        options = {
            "random_seed": f"{instrument_seed},{simulation_seed}",
            "IMAGING_LIST": photons_file,
            "stars": stars_file,
            "altitude": self.site_model.get_parameter_value("corsika_observation_level"),
            "telescope_theta": theta,
            "star_photons": request.photons_per_instance,
            "telescope_phi": phi,
            "camera_transmission": 1.0,
            "nightsky_background": "all:0.",
            "trigger_current_limit": "1e10",
            "telescope_random_angle": 0,
            "telescope_random_error": 0,
            "convergent_depth": 0,
            "maximum_telescopes": 1,
            "show": "all",
            "camera_filter": "none",
        } | parameter_changes.sim_telarray_options(request.parameter_changes)
        command = " ".join(
            [
                str(self.context.software.sim_telarray_exe_debug_trace),
                f"-c {self.telescope_model.config_file_path}",
                f"-I{self.telescope_model.config_file_directory}",
                *(f"-C {key}={value}" for key, value in options.items()),
                str(self.context.software.corsika_dummy_file),
            ]
        )
        with script.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")
            file.write("set -e\nset -o pipefail\n\n")
            file.write(f"exec > '{log_file}' 2> '{log_file.with_suffix('.err')}'\n\n")
            file.write(f"{command}\n")
        script.chmod(script.stat().st_mode | 0o110)


class _ImagingList:
    """Parser of the imaging list of the sim_telarray debug-trace build."""

    def __init__(self, with_mirrors):
        self.with_mirrors = with_mirrors

    def read(self, photons_file):
        """Return angles [deg] and hit points [m] of the photons in an imaging list."""
        data = self._compute_incidence_angles_from_imaging_list(photons_file)
        renamed = {
            "angle_incidence_focal": data["angle_incidence_focal_deg"],
        }
        if self.with_mirrors:
            renamed |= {
                "angle_incidence_primary": data["angle_incidence_primary_deg"],
                "angle_incidence_secondary": data["angle_incidence_secondary_deg"],
                "primary_hit_x": data["primary_hit_x_m"],
                "primary_hit_y": data["primary_hit_y_m"],
                "secondary_hit_x": data["secondary_hit_x_m"],
                "secondary_hit_y": data["secondary_hit_y_m"],
            }
        return renamed

    def _compute_incidence_angles_from_imaging_list(self, photons_file):
        """Compute incidence angles from an imaging list file.

        Column positions may differ between telescope types and sim_telarray versions.
        Header lines (``# Column N: ...``) are required to identify the physical
        angle definitions. Missing required headers raise ValueError.

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
        primary = secondary = radius_m = secondary_radius_m = None
        primary_hit_x_m = primary_hit_y_m = secondary_hit_x_m = secondary_hit_y_m = None
        if self.with_mirrors:
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

        result = {
            "angle_incidence_focal_deg": focal,
        }
        if self.with_mirrors:
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
        indices = {}

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

        required = {"focal"}
        if self.with_mirrors:
            required |= {"primary", "secondary"}
        missing = required - indices.keys()
        if missing:
            raise ValueError(f"Missing imaging-list column headers: {sorted(missing)}")
        return indices

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
        if "angle of incidence" in desc:
            if "focal surface" in desc and "optical axis" in desc:
                indices["focal"] = num - 1
                return
        if not self.with_mirrors:
            return
        if "angle of incidence" in desc:
            for mirror in ("primary", "secondary"):
                if f"{mirror} mirror" in desc:
                    indices[mirror] = num - 1
                    return
        if "reflection point" not in desc:
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
        if not self.with_mirrors:
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
