"""Telescope optics of the simulation chain: the request of one telescope.

An :class:`OpticsRequest` describes what is traced for one telescope of an array model: the model
it is built from, the pointing, the seeds and the telescope transmission mode. It does not depend
on a program. The optical model file and the command line of ``obdeect ray-tracing trace-stream``,
which traces a photon stream (ground or telescope frame) through the optics to camera-frame
photons, are made in ``simtools.simulation_software.obdeect.optics``.

Frames: the ground frame is the horizontal plane through the telescope sphere centre (x north,
y west, z up); the pointing is the telescope zenith and azimuth in that frame. The camera frame is
the position on the camera entrance, the direction and the time of the photons. The tracing moves
ground-frame photons to an input plane above the incoming obscurers by itself.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simtools.model import mirrors

#: Prefix of the seed string of a seed file, see ``SimtelSeeds``.
SEED_FILE_PREFIX = "file-by-run:"


def derive_seed(seed, *keys):
    """Return a reproducible 32-bit seed for the given non-negative integer keys."""
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1, np.uint32)[0])


@dataclass(frozen=True)
class OpticsRequest:
    """
    Optics job of one telescope of one array model.

    Parameters
    ----------
    telescope_type : str
        Telescope type (model name, e.g. ``LSTN-design``).
    telescope_name : str
        Array element name of the telescope.
    telescope_id : int
        Telescope id in the photon stream.
    model_version : str
        Model version.
    site : str
        Site (``North`` or ``South``).
    dual_mirror : bool
        Whether the telescope has two mirrors.
    zenith_deg, azimuth_deg : float
        Telescope pointing [deg] in the array frame (x north, y west).
    instrument_seed : int
        Seed of the random instrument (mirror errors) of this telescope.
    tracing_seed : int
        Seed of the random draws of the photon tracing.
    telescope_transmission : str
        Only ``full`` (the telescope transmission is applied completely).
    threads : int
        Tracing threads (0 uses all cores).
    """

    telescope_type: str
    telescope_name: str
    telescope_id: int
    model_version: str
    site: str
    dual_mirror: bool
    zenith_deg: float
    azimuth_deg: float
    instrument_seed: int
    tracing_seed: int
    telescope_transmission: str = "full"
    threads: int = 1

    def __post_init__(self):
        """Check the request."""
        if self.telescope_id < 0:
            raise ValueError("The telescope id of an optics request must not be negative.")
        if not 0.0 <= self.zenith_deg < 90.0:
            raise ValueError(f"Telescope zenith angle {self.zenith_deg} deg must lie in [0, 90).")
        if min(self.instrument_seed, self.tracing_seed) < 0:
            raise ValueError("Seeds of the optics must not be negative.")
        if self.threads < 0:
            raise ValueError("Threads of the optics must not be negative.")

    @property
    def model_file_name(self):
        """Name of the optical model file of this request (telescope, version, pointing, seed)."""
        return (
            f"optical_model_{self.telescope_name}_{self.model_version}"
            f"_zen{self.zenith_deg:g}_seed{self.instrument_seed}.json"
        )


def seeds_of(sim_telarray_seed):
    """
    Return the instrument seed and the tracing seed of the sim_telarray seeds.

    The seeds are read from the seed string ("instrument seed,simulation seed"). For a seed file
    ("file-by-run:..."), the instrument seed is taken from the seed object and the simulation seed
    is the tracing seed.

    Parameters
    ----------
    sim_telarray_seed : SimtelSeeds
        Seeds of the sim_telarray simulations.

    Returns
    -------
    tuple of int
        Instrument seed and tracing seed.
    """
    if sim_telarray_seed.seed_string.startswith(SEED_FILE_PREFIX):
        if sim_telarray_seed.instrument_seed is None:
            raise ValueError("Instrument seed is not set for the seed file of sim_telarray")
        return int(sim_telarray_seed.instrument_seed), int(sim_telarray_seed.simulation_seed)
    instrument, tracing = sim_telarray_seed.seed_string.split(",")
    return int(instrument), int(tracing)


def optics_requests(
    array_model,
    simulation_parameters,
    telescope_ids,
    *,
    stream_id_offset=0,
    telescope_transmission="full",
    threads=1,
):
    """
    Return the optics requests of an array model, one per telescope.

    Each telescope has its own random instrument: its instrument seed is derived from the
    instrument seed and the telescope id, as in the shower route.

    Parameters
    ----------
    array_model : ArrayModel
        Telescope and site models, and the seeds.
    simulation_parameters : SimulationParameters
        Provides the telescope pointing.
    telescope_ids : dict
        IACT telescope id (position in the CORSIKA telescope list) of each array element name.
    stream_id_offset : int
        Added to the IACT telescope id to give the telescope id in the photon stream.
    telescope_transmission : str
        Telescope transmission mode.
    threads : int
        Tracing threads of each request.

    Returns
    -------
    list of OpticsRequest
        One request per telescope, in the order of the array.
    """
    instrument_seed, tracing_seed = seeds_of(array_model.sim_telarray_seed)
    return [
        OpticsRequest(
            telescope_type=telescope.name,
            telescope_name=name,
            telescope_id=int(telescope_ids[name]) + stream_id_offset,
            model_version=array_model.model_version,
            site=array_model.site,
            dual_mirror=mirrors.is_dual_mirror(telescope),
            zenith_deg=float(simulation_parameters.zenith_angle),
            azimuth_deg=float(simulation_parameters.azimuth_angle),
            instrument_seed=derive_seed(instrument_seed, int(telescope_ids[name])),
            tracing_seed=tracing_seed,
            telescope_transmission=telescope_transmission,
            threads=threads,
        )
        for name, telescope in array_model.telescope_models.items()
    ]


def write_telescope_map(requests, path):
    """
    Write the telescopes of the optics requests as a JSON file (documentation of the job).

    Parameters
    ----------
    requests : iterable of OpticsRequest
        Optics requests of an array model.
    path : Path
        Output file.

    Returns
    -------
    Path
        The file written.
    """
    entries = {
        str(request.telescope_id): {
            "name": request.telescope_name,
            "model": request.telescope_type,
            "model_version": request.model_version,
            "site": request.site,
            "dual_mirror": request.dual_mirror,
            "theta_deg": request.zenith_deg,
            "phi_deg": request.azimuth_deg,
        }
        for request in requests
    }
    path = Path(path)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return path
