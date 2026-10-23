"""Interface of the ray-tracing backends.

A backend traces star light through the optics of one telescope model. The
ray-tracing applications describe what to trace (``StarImageRequest``) and
analyse what comes back; how a backend traces, which files it writes and what it
keeps in memory stays inside the backend. Backends:

- ``sim_telarray``: sim_telarray runs writing photon lists
  (``backends.sim_telarray``);
- ``obdeect``: obdeect in process (``backends.obdeect``, optional dependency).

Each backend class is constructed with the telescope model, the site model, a
label, the telescope zenith angle [deg] and the single-mirror flag, and provides:

- ``simulate(requests, test, force, compress_photons, max_workers)``: trace the images;
- ``psf_image(request, focal_length, containment_fraction, use_rx, save_photons)``:
  the ``PSFImage`` of one traced image and whether the telescope structure was
  traced (then ``telescope_transmission`` does not apply);
- ``incidence_angles(request, work_directory, name, with_mirrors, keep_files)``:
  per-photon incidence angles on the focal surface (and mirrors) and the
  intermediate files written.

Backends with ``traces_light_sources`` also make and trace the light of
calibration light sources (flashers, illuminators; ``LightSourceRequest``):

- ``camera_photons(request)``: for each light pulse, an iterator over blocks of
  the photons crossing the camera plane (through the focal-surface vertex,
  normal to the optical axis; telescope frame, metres): ``position_m``,
  ``direction``, ``time_ns``, ``weight`` (photons) and ``wavelength_nm``.

Otherwise sim_telarray's LightEmission programs make the light and sim_telarray
traces it.
"""

from dataclasses import dataclass, field

from simtools import settings


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


def get_backend(name=None):
    """
    Return the ray-tracing backend class.

    Parameters
    ----------
    name: str or None
        Backend name; default: the configured ``ray_tracing_backend``.

    Returns
    -------
    type
        Backend class.
    """
    name = name or settings.config.ray_tracing_backend
    # pylint: disable=import-outside-toplevel
    if name == "obdeect":  # Optional dependency: imported only when selected.
        from simtools.ray_tracing.backends.obdeect import ObdeectBackend

        return ObdeectBackend
    from simtools.ray_tracing.backends.sim_telarray import SimtelBackend

    return SimtelBackend
