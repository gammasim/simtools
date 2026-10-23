"""Ray tracing with obdeect, in process.

obdeect compiles the model parameters of a telescope into an optical model and
traces star light through the optics to the camera entrance. The light source,
the telescope pointing and the parameter changes of the ray-tracing modes are
those of the sim_telarray backend, so that both backends answer the same
question. Images stay in memory.

obdeect also makes the light of calibration light sources (point emitters) and
traces it to the camera entrance.
"""

import math
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy import constants
from astropy.table import QTable
from obdeect.model import DETECTED, OpticalModel

from simtools.io import io_handler
from simtools.ray_tracing.psf_analysis import PSFImage
from simtools.simtel.simulator_ray_tracing import telescope_pointing
from simtools.utils import names

_SPEED_OF_LIGHT_M_PER_NS = constants.c.to_value(u.m / u.ns)


def star_direction(zenith_angle, off_axis_x, off_axis_y):
    """Return the unit vector to the star in the telescope frame (+z: optical axis)."""
    theta, phi = (math.radians(a) for a in telescope_pointing(zenith_angle, off_axis_x, off_axis_y))
    zenith = math.radians(zenith_angle)
    return (
        math.sin(zenith) * math.cos(theta) * math.cos(phi) - math.cos(zenith) * math.sin(theta),
        math.sin(zenith) * math.sin(phi),
        math.sin(zenith) * math.sin(theta) * math.cos(phi) + math.cos(zenith) * math.cos(theta),
    )


class ObdeectRayTracer:
    """Trace star light through one telescope model with obdeect.

    Parameters
    ----------
    telescope_model: TelescopeModel
        Telescope model.
    """

    def __init__(self, telescope_model):
        self.telescope_model = telescope_model
        telescope_model.export_model_files(update_if_necessary=True)
        self._models = {}

    def optical_model(self, telescope_zenith, instrument_seed, changes=None):
        """Optical model at a telescope zenith angle [deg] (cached).

        ``instrument_seed`` fixes the random mirror errors of this instrument
        instance; ``changes`` are model-parameter values replacing those of the
        telescope model (see ``simtools.ray_tracing.parameter_changes``).
        """
        changes = changes or {}
        key = (round(telescope_zenith, 9), instrument_seed, repr(sorted(changes.items())))
        if key not in self._models:
            records, files = self._records(changes)
            self._models[key] = OpticalModel.from_records(
                records,
                lambda name: files[name].read_text(encoding="utf-8"),
                configuration_seed=instrument_seed,
                zenith_deg=telescope_zenith,
                provenance={
                    "telescope": self.telescope_model.name,
                    "model_version": self.telescope_model.model_version,
                },
            )
        return self._models[key]

    def _records(self, changes):
        """Parameter records and their files, with the changes of this run."""
        directory = self.telescope_model.config_file_directory
        records = {
            name: record
            for name, record in self.telescope_model.parameters.items()
            if isinstance(record, dict) and "value" in record
        }
        files = {
            name: directory / record["value"]
            for name, record in records.items()
            if record.get("file") and isinstance(record["value"], str)
        }
        for name, value in changes.items():
            if name not in records:
                continue
            if records[name].get("file"):
                files[name] = Path(value)
                value = Path(value).name
            records[name] = deepcopy(records[name]) | {"value": value}
        return records, files

    def trace_star(
        self,
        zenith_angle,
        off_axis_x,
        off_axis_y,
        source_distance_m,
        photons_per_instance,
        seeds,
        changes=None,
    ):
        """Trace star light, as in the sim_telarray ray-tracing runs.

        ``photons_per_instance`` photons for each random instrument instance,
        given by its (instrument seed, simulation seed) in ``seeds``, as
        sim_telarray runs one simulation per instance; ``changes`` as in
        ``optical_model``.

        Returns
        -------
        dict
            Arrival arrays (see ``obdeect.OpticalModel.trace``) of the photons
            reaching the camera entrance, the number of photons traced and the
            area they were sampled on [m^2], and whether the optical model
            shadows with its own structure.
        """
        telescope_zenith, _ = telescope_pointing(zenith_angle, off_axis_x, off_axis_y)
        towards_source = star_direction(zenith_angle, off_axis_x, off_axis_y)
        arrivals = []
        for instrument_seed, simulation_seed in seeds:
            model = self.optical_model(telescope_zenith, instrument_seed, changes)
            sample = model.star_photons(
                photons_per_instance,
                seed=simulation_seed,
                towards_source=towards_source,
                distance_m=source_distance_m,
            )
            traced = model.trace(sample, seed=simulation_seed, apply_responses=False)
            detected = traced["status"] == DETECTED
            arrivals.append({name: values[detected] for name, values in traced.items()})
        return {name: np.concatenate([a[name] for a in arrivals]) for name in arrivals[0]} | {
            "total_photons": photons_per_instance * len(seeds),
            "total_area_m2": model.sampling_area_m2,
            "has_structure": model.has_structure,
        }

    def trace_light_source(self, request):
        """
        Make and trace the light of a calibration light source, pulse by pulse.

        Parameters
        ----------
        request: LightSourceRequest
            Light source.

        Yields
        ------
        iterator of dict
            The photons of one pulse crossing the camera plane, block by block
            (see ``simtools.ray_tracing.backends``).
        """
        instrument_seed, simulation_seed = request.seeds
        model = self.optical_model(request.telescope_zenith, instrument_seed)
        plane_z = model.focal_surface_z_m
        disk = (
            ((0.0, 0.0, plane_z), (0.0, 0.0, 1.0), request.camera_radius_m)
            if request.inside_telescope
            else None  # The telescope entrance.
        )
        for pulse_number, photons in enumerate(request.photons_per_pulse):
            blocks = model.trace_emitter_blocks(
                photons,
                seed=simulation_seed,
                first_id=pulse_number << 40,
                position_m=request.position_m,
                intensity=request.angular_profile,
                axis=request.axis,
                pulse=request.pulse,
                wavelength_nm=request.wavelength_nm,
                disk=disk,
                direct=request.inside_telescope,
            )
            yield (_at_plane(block, plane_z, model.propagation_group_index) for block in blocks)


def _at_plane(photons, plane_z, group_index):
    """Move photons on the focal surface along their direction to the plane z = plane_z [m]."""
    position, direction = photons["position_m"], photons["direction"]
    path = (plane_z - position[:, 2]) / direction[:, 2]
    photons["position_m"] = position + path[:, None] * direction
    photons["time_ns"] = photons["time_ns"] + path * group_index / _SPEED_OF_LIGHT_M_PER_NS
    return photons


class ObdeectBackend:
    """
    Ray tracing with obdeect, in process (see ``simtools.ray_tracing.backends.interface``).

    Parameters
    ----------
    telescope_model: TelescopeModel
        Telescope model.
    site_model: SiteModel
        Site model (not used: obdeect traces from the telescope entrance).
    label: str
        Label of the output files.
    zenith_angle: float
        Telescope zenith angle [deg].
    single_mirror_mode: bool
        Trace single mirrors.
    """

    traces_light_sources = True

    def __init__(self, telescope_model, site_model, label, zenith_angle, single_mirror_mode):
        del site_model
        self.telescope_model = telescope_model
        self.label = label
        self.zenith_angle = zenith_angle
        self.single_mirror_mode = single_mirror_mode
        self._tracer = ObdeectRayTracer(telescope_model)
        self._images = {}

    def _trace(self, request):
        return self._tracer.trace_star(
            self.zenith_angle,
            request.off_axis_x,
            request.off_axis_y,
            request.source_distance_km * 1000.0,
            request.photons_per_instance,
            request.seeds,
            request.parameter_changes,
        )

    def simulate(self, requests, test=False, force=False, compress_photons=True, max_workers=8):
        """
        Trace the images; they are kept in memory.

        ``test``, ``force``, ``compress_photons`` and ``max_workers`` concern
        photon files and processes and have no effect here.
        """
        del test, force, compress_photons, max_workers
        for request in requests:
            arrivals = self._trace(request)
            # Only what the image needs: camera positions [cm] of the detected photons.
            self._images[request.key] = {
                "position_cm": 100.0 * arrivals["position_m"][:, :2],
                "total_photons": arrivals["total_photons"],
                "total_area_m2": arrivals["total_area_m2"],
                "has_structure": bool(arrivals["has_structure"]),
            }

    def psf_image(
        self, request, focal_length, containment_fraction, use_rx=False, save_photons=False
    ):
        """
        Return the PSF image of one traced image.

        Parameters as for the sim_telarray backend; ``save_photons`` writes the
        photons as an ECSV table.

        Returns
        -------
        tuple of PSFImage and bool
            The image, and whether the telescope structure was traced.
        """
        if use_rx:
            raise ValueError("rx analyses sim_telarray photon lists only")
        traced = self._images.get(request.key)
        if traced is None:
            raise RuntimeError(f"Image {request.key} was not traced; run simulate first")
        position_cm = traced["position_cm"]
        image = PSFImage(focal_length=focal_length, containment_fraction=containment_fraction)
        image.set_photon_positions(
            position_cm[:, 0], position_cm[:, 1], traced["total_photons"], traced["total_area_m2"]
        )
        if save_photons:
            self._write_photons(request, traced, position_cm)
        return image, bool(traced["has_structure"])

    def _write_photons(self, request, traced, position_cm):
        """Write the photons of an image as an ECSV table (camera-entrance positions)."""
        file_name = io_handler.IOHandler().get_output_directory() / names.generate_file_name(
            file_type="ray_tracing_photons",
            suffix=".ecsv",
            site=self.telescope_model.site,
            telescope_model_name=self.telescope_model.name,
            source_distance=None if self.single_mirror_mode else request.source_distance_km,
            zenith_angle=self.zenith_angle,
            off_axis_x=request.off_axis_x,
            off_axis_y=request.off_axis_y,
            mirror_number=request.mirror_number if self.single_mirror_mode else None,
            label=self.label,
        )
        table = QTable(
            [position_cm[:, 0] * u.cm, position_cm[:, 1] * u.cm],
            names=("x", "y"),
            meta={
                "total_photons": int(traced["total_photons"]),
                "total_area_m2": float(traced["total_area_m2"]),
                "traced_structure": bool(traced["has_structure"]),
            },
        )
        table.write(file_name, format="ascii.ecsv", overwrite=True)

    def incidence_angles(self, request, work_directory, name, with_mirrors, keep_files=False):
        """
        Per-photon incidence angles (see the sim_telarray backend); no files are written.

        Returns
        -------
        tuple of dict and list
            Angles [deg] and hit points [m], and no intermediate files.
        """
        del work_directory, name, keep_files
        arrivals = self._trace(request)
        if len(arrivals["direction"]) == 0:
            raise RuntimeError("No photon reached the focal surface")
        # Angle to the optical axis, as in the sim_telarray imaging list.
        data = {"angle_incidence_focal": np.degrees(np.arccos(np.abs(arrivals["direction"][:, 2])))}
        if with_mirrors:
            for mirror in ("primary", "secondary"):
                point = arrivals[f"{mirror}_point_m"]
                data[f"angle_incidence_{mirror}"] = arrivals[f"{mirror}_incidence_deg"]
                data[f"{mirror}_hit_x"] = point[:, 0]
                data[f"{mirror}_hit_y"] = point[:, 1]
        return data, []

    def camera_photons(self, request):
        """Photons of a calibration light source at the camera entrance, pulse by pulse, in blocks."""
        return self._tracer.trace_light_source(request)
