"""Ray-tracing request types, backend contracts, and helpers.

See :class:`RayTracingBackend` for the common backend contract.
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Protocol

from simtools.constants import RAY_TRACING_BACKEND_CLASSES
from simtools.context import RunContext
from simtools.utils import names


class RayTracingBackend(Protocol):
    """Contract implemented by ray-tracing backends.

    A backend traces star light through the optics of one telescope model. The
    ray-tracing applications describe what to trace (``StarImageRequest``) and
    analyse what comes back; how a backend traces, which files it writes and what
    it keeps in memory stays inside the backend.

    A backend class is constructed with the telescope model, the site model, a
    label, the telescope zenith angle [deg] and the single-mirror flag. Backends
    are selected by name through ``get_backend``.
    """

    def simulate(self, requests, test=False, force=False, compress_photons=True, max_workers=8):
        """Trace the requested star images."""

    def psf_image(
        self, request, focal_length, containment_fraction, use_rx=False, save_photons=False
    ):
        """Return the ``PSFImage`` of one traced image and whether the structure was traced.

        If the telescope structure was traced, ``telescope_transmission`` does not apply.
        """

    def incidence_angles(self, request, work_directory, name, with_mirrors, keep_files=False):
        """Return per-photon incidence angles on the focal surface (and mirrors).

        Also returns the intermediate files written.
        """


class LightSourceRayTracingBackend(RayTracingBackend, Protocol):
    """Ray-tracing backend that also traces calibration light sources.

    Such backends set ``traces_light_sources``; for other backends sim_telarray's
    LightEmission programs make the light and sim_telarray traces it.
    """

    def camera_photons(self, request):
        """Return, for each light pulse of a ``LightSourceRequest``, an iterator over blocks.

        Each block holds the photons crossing the camera plane (through the
        focal-surface vertex, normal to the optical axis; telescope frame, metres):
        ``position_m``, ``direction``, ``time_ns``, ``weight`` (photons) and
        ``wavelength_nm``.
        """


@dataclass(frozen=True)
class StarImageRequest:
    """
    One star image to trace.

    Parameters
    ----------
    off_axis_x, off_axis_y: float
        Star position in the camera [deg].
    source_distance_km: float
        Source distance [km].
    seeds: tuple of tuple of int
        (Instrument seed, simulation seed) of each random instrument instance.
    photons_per_instance: int
        Photons traced per instance.
    mirror_number: int or None
        Mirror in single-mirror mode.
    parameter_changes: dict
        Model-parameter values replacing those of the telescope model.
    """

    off_axis_x: float
    off_axis_y: float
    source_distance_km: float
    seeds: tuple
    photons_per_instance: int
    mirror_number: int | None = None
    parameter_changes: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def key(self):
        """Return the identifier of the image (offsets and mirror)."""
        return self.off_axis_x, self.off_axis_y, self.mirror_number


@dataclass(frozen=True)
class LightSourceRequest:
    """
    The light of a calibration light source (a point emitter), pulse by pulse.

    Parameters
    ----------
    position_m: tuple of float
        Emitter position in the telescope frame [m] (+z along the optical axis).
    axis: tuple of float
        Unit vector of the axis of the angular profile (telescope frame).
    angular_profile: tuple of numpy.ndarray
        Angles to the axis [deg] and relative intensities per solid angle.
    pulse: tuple of numpy.ndarray or None
        Emission times [ns] and relative rates; None: instantaneous.
    wavelength_nm: float
        Wavelength [nm].
    photons_per_pulse: tuple of float
        Photons emitted into 4 pi in each pulse.
    telescope_zenith: float
        Telescope zenith angle [deg].
    seeds: tuple of int
        (Instrument seed, simulation seed).
    inside_telescope: bool
        The light starts inside the telescope on its way to the camera
        (flasher): it reaches the camera directly, without the mirrors.
    camera_radius_m: float or None
        Radius of the camera [m]; the light of a source inside the telescope
        is made only on its way to the camera.
    """

    position_m: tuple
    axis: tuple
    angular_profile: tuple = field(hash=False, compare=False)
    pulse: tuple | None = field(hash=False, compare=False)
    wavelength_nm: float
    photons_per_pulse: tuple
    telescope_zenith: float
    seeds: tuple
    inside_telescope: bool = False
    camera_radius_m: float | None = None


def image_file_name(backend, request, file_type, suffix):
    """
    Return the name of a file of one traced image.

    Parameters
    ----------
    backend: object
        Backend (telescope model, zenith angle, label and single-mirror flag).
    request: StarImageRequest
        Star image.
    file_type: str
        File type (e.g. ``ray_tracing_photons``).
    suffix: str
        File suffix.

    Returns
    -------
    str
        File name.
    """
    return names.generate_file_name(
        file_type=file_type,
        suffix=suffix,
        site=backend.telescope_model.site,
        telescope_model_name=backend.telescope_model.name,
        source_distance=None if backend.single_mirror_mode else request.source_distance_km,
        zenith_angle=backend.zenith_angle,
        off_axis_x=request.off_axis_x,
        off_axis_y=request.off_axis_y,
        mirror_number=request.mirror_number if backend.single_mirror_mode else None,
        label=backend.label,
    )


def get_backend(name=None, *, context=None):
    """
    Return the ray-tracing backend class.

    Parameters
    ----------
    name: str or None
        Backend name; defaults to the context's configured backend.
    context : RunContext, optional
        Run configuration. The environment context is used when omitted.

    Returns
    -------
    type
        Backend class.
    """
    context = context or RunContext.from_environment()
    name = name or context.software.ray_tracing_backend
    try:
        module_name, class_name = RAY_TRACING_BACKEND_CLASSES[name].split(":")
    except KeyError as exc:
        valid_names = ", ".join(RAY_TRACING_BACKEND_CLASSES)
        raise ValueError(
            f"Unknown ray-tracing backend {name!r}; choose from: {valid_names}"
        ) from exc
    return getattr(import_module(module_name), class_name)
