"""Optical ray tracing with obdeect, in process.

obdeect compiles the model parameters of a telescope into an optical model and
traces star light through the optics to the camera entrance. The light source,
the telescope pointing and the parameter changes of the single-mirror mode
follow the sim_telarray ray-tracing runs of simtools, so that both backends
answer the same question.
"""

import math
from copy import deepcopy

import numpy as np
from obdeect.model import DETECTED, OpticalModel

from simtools.simtel.simulator_ray_tracing import telescope_pointing

# Single-mirror mode: the mirror is placed alone at the dish centre and imaged
# from twice its focal length (as in SimulatorRayTracing).
_SINGLE_MIRROR_ZEROS = {
    "focus_offset": [0.0, 0.0, 0.0, 0.0],
    "camera_body_diameter": 0.0,
    "parabolic_dish": False,
    "mirror_align_random_distance": 0.0,
    "mirror_align_random_vertical": [0.0, 28.0, 0.0, 0.0],
}
_PERFECT_MIRROR_ZEROS = {
    "random_focal_length": [0.0, 0.0],
    "mirror_reflection_random_angle": [0.0, 0.0, 0.0],
    "mirror_align_random_distance": 0.0,
    "mirror_align_random_horizontal": [0.0, 28.0, 0.0, 0.0],
    "mirror_align_random_vertical": [0.0, 28.0, 0.0, 0.0],
}


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
    perfect_mirror: bool
        Remove random mirror errors (focal lengths, alignment, scattering).
    """

    def __init__(self, telescope_model, perfect_mirror=False):
        self.telescope_model = telescope_model
        self._changes = dict(_PERFECT_MIRROR_ZEROS) if perfect_mirror else {}
        telescope_model.export_model_files(update_if_necessary=True)
        self._models = {}

    def optical_model(self, telescope_zenith, instrument_seed, single_mirror=None):
        """Optical model at a telescope zenith angle [deg] (cached).

        ``instrument_seed`` fixes the random mirror errors of this instrument
        instance. ``single_mirror`` is a (mirror number, mirror list file,
        camera distance [cm]) tuple for the single-mirror mode.
        """
        key = (round(telescope_zenith, 9), instrument_seed, single_mirror)
        if key not in self._models:
            records, files = self._records(single_mirror)
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

    def _records(self, single_mirror):
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
        changes = dict(self._changes)
        if single_mirror is not None:
            _, mirror_list, camera_distance = single_mirror
            focal_length = self.telescope_model.get_parameter_value("mirror_focal_length")
            changes |= _SINGLE_MIRROR_ZEROS | {
                "focal_length": camera_distance,
                "dish_shape_length": focal_length,
                "mirror_focal_length": focal_length,
            }
            files["mirror_list"] = mirror_list
        for name, value in changes.items():
            if name in records:
                records[name] = deepcopy(records[name]) | {"value": value}
        return records, files

    def trace_star(
        self,
        zenith_angle,
        off_axis_x,
        off_axis_y,
        source_distance_m,
        photons,
        seeds,
        single_mirror=None,
    ):
        """Trace star light, as in the sim_telarray ray-tracing runs.

        Photons are split evenly over the random instrument instances, given by
        their (instrument seed, simulation seed) in ``seeds``, as sim_telarray
        runs one simulation per instance; ``single_mirror`` as in ``optical_model``.

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
        if photons < len(seeds):
            raise ValueError("number_of_photons must be at least one per instrument instance")
        per_instance = photons // len(seeds)
        arrivals = []
        for instrument_seed, simulation_seed in seeds:
            model = self.optical_model(telescope_zenith, instrument_seed, single_mirror)
            sample = model.star_photons(
                per_instance,
                seed=simulation_seed,
                towards_source=towards_source,
                distance_m=source_distance_m,
            )
            traced = model.trace(sample, seed=simulation_seed, apply_responses=False)
            detected = traced["status"] == DETECTED
            arrivals.append({name: values[detected] for name, values in traced.items()})
        return {name: np.concatenate([a[name] for a in arrivals]) for name in arrivals[0]} | {
            "total_photons": per_instance * len(seeds),
            "total_area_m2": model.sampling_area_m2,
            "has_structure": model.has_structure,
        }
