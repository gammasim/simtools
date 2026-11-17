"""Atmosphere of the simulation chain: request and its construction from the models.

The atmosphere applies the atmospheric transmission and the in-telescope path of each telescope
to a photon stream, moves photons from the sky to the ground frame, and thins or weights the
photons. This module holds the model inputs (:class:`AtmosphereRequest`) and does not depend on
a program. The command line of the obdeect program is in
``simtools.simulation_software.obdeect.atmosphere``; the simulation stage is in
``simtools.runners.atmosphere_stage``.
"""

from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u

from simtools.model.model_utils import export_model_files
from simtools.simulation.configuration import get_model_writer

PLANE_PARALLEL = "plane_parallel"
MODES = ("thinned", "weighted")
DEFAULT_SEED = 1
DEFAULT_MODE = "thinned"
#: Name of the transmission table exported from an ECSV model parameter.
TRANSMISSION_FILE_NAME = "atmosphere-transmission.dat"
#: Name of the transmission table with optical depth 0 (sim_telarray, bypass 1 or 2).
ZERO_TRANSMISSION_FILE_NAME = "atmosphere-zero-transmission.dat"
#: Name of the atmospheric profile exported from an ECSV model parameter (atmprof format).
PROFILE_FILE_NAME = "atmosphere-profile.dat"
#: obdeect reads transmission wavelengths that are integers below this value [nm].
OBDEECT_MAX_WAVELENGTH_NM = 1000
#: Stdin and stdout of a command in the shell.
STDIO = "-"


@dataclass(frozen=True)
class AtmosphereRequest:
    """
    Inputs of the atmosphere of one array.

    Parameters
    ----------
    site : str
        Site name.
    observation_level_m : float
        Observation level [m a.s.l.].
    profile_file : Path
        Atmospheric profile (atmprof format).
    transmission_file : Path
        Atmospheric transmission table (extinction-table format).
    seed : int
        Seed of the random numbers of the thinning.
    mode : str
        Output mode, ``thinned`` (one photon per surviving photon) or ``weighted``.
    wavelength_range_nm : tuple of float or None
        Wavelength range [nm] of photons without a wavelength; None uses the run keys.
    telescope_pathlen_m : dict
        Path length inside each telescope [m], keyed by the stream telescope id.
    geometry : str
        Atmosphere geometry. Only ``plane_parallel`` is available.
    """

    site: str
    observation_level_m: float
    profile_file: Path
    transmission_file: Path
    seed: int
    mode: str = DEFAULT_MODE
    wavelength_range_nm: tuple[float, float] | None = None
    telescope_pathlen_m: dict = field(default_factory=dict, hash=False)
    geometry: str = PLANE_PARALLEL

    def __post_init__(self):
        """Check the geometry and the output mode."""
        if self.geometry != PLANE_PARALLEL:
            raise ValueError(
                f"Unsupported atmosphere geometry {self.geometry!r}; use {PLANE_PARALLEL!r}."
            )
        if self.mode not in MODES:
            raise ValueError(f"Invalid atmosphere output mode {self.mode!r}; choose from {MODES}.")


def in_telescope_pathlength_m(telescope_model):
    """
    Return the path length inside a telescope [m] used by the extinction.

    Dual-mirror telescopes use 0 m (sim_telarray zeroes the constant term of the path for
    mirror class 2); single-mirror telescopes use the focal length.

    Parameters
    ----------
    telescope_model : TelescopeModel
        Telescope model.

    Returns
    -------
    float
        Path length [m].
    """
    if telescope_model.get_parameter_value("mirror_class") == 2:
        return 0.0
    return telescope_model.get_parameter_value_with_unit("focal_length").to_value(u.m)


def export_atmosphere_files(site_model, directory=None):
    """
    Export the atmospheric profile and transmission of a site model as files.

    Parameters
    ----------
    site_model : SiteModel
        Site model.
    directory : Path or None
        Directory of the transmission table exported from an ECSV model parameter; default is
        the configuration directory of the site model.

    Returns
    -------
    tuple of Path
        Profile file and transmission table file.
    """
    export_model_files(site_model, update_if_necessary=True)
    config_directory = Path(site_model.config_file_directory)
    export_directory = Path(directory or config_directory)
    profile_name = str(site_model.get_parameter_value("atmospheric_profile"))
    profile_file = config_directory / profile_name
    if profile_name.lower().endswith(".ecsv"):
        # obdeect reads the atmprof text format, not the ECSV model parameter
        profile_file = export_directory / get_model_writer(
            site_model
        ).export_model_parameter_as_simtel_file(
            "atmospheric_profile",
            export_directory,
            table_format="plain",
            output_name=PROFILE_FILE_NAME,
        )
    table_name = str(site_model.get_parameter_value("atmospheric_transmission"))
    if not table_name.lower().endswith(".ecsv"):
        return profile_file, config_directory / table_name
    table_name = get_model_writer(site_model).export_model_parameter_as_simtel_file(
        "atmospheric_transmission",
        export_directory,
        "atmospheric_transmission",
        TRANSMISSION_FILE_NAME,
    )
    return profile_file, export_directory / table_name


def build_atmosphere_request(
    site_model,
    telescope_models,
    *,
    seed=DEFAULT_SEED,
    mode=DEFAULT_MODE,
    wavelength_range_nm=None,
    directory=None,
):
    """
    Build the atmosphere request of an array from its site and telescope models.

    Telescope ids are the 1-based positions of the telescopes in ``telescope_models``.

    Parameters
    ----------
    site_model : SiteModel
        Site model of the array.
    telescope_models : dict
        Telescope models keyed by array element name, in the order of the array.
    seed : int
        Seed of the thinning.
    mode : str
        Output mode (``thinned`` or ``weighted``).
    wavelength_range_nm : tuple of float or None
        Wavelength range [nm] of photons without a wavelength.
    directory : Path or None
        Directory of the exported transmission table (see :func:`export_atmosphere_files`).

    Returns
    -------
    AtmosphereRequest
        Atmosphere request.
    """
    profile_file, transmission_file = export_atmosphere_files(site_model, directory)
    pathlengths = {
        telescope_id: in_telescope_pathlength_m(telescope_model)
        for telescope_id, telescope_model in enumerate(telescope_models.values(), start=1)
    }
    return AtmosphereRequest(
        site=site_model.site,
        observation_level_m=site_model.get_parameter_value_with_unit(
            "corsika_observation_level"
        ).to_value(u.m),
        profile_file=profile_file,
        transmission_file=transmission_file,
        seed=seed,
        mode=mode,
        wavelength_range_nm=None if wavelength_range_nm is None else tuple(wavelength_range_nm),
        telescope_pathlen_m=pathlengths,
    )


def _is_number(text):
    """Return True if the text is a number."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _is_integer_below(text, limit):
    """Return True if the text is an integer in [0, limit)."""
    try:
        value = int(text)
    except ValueError:
        return False
    return 0 <= value < limit


def write_obdeect_transmission(transmission_file, directory):
    """
    Write the transmission table as obdeect reads it and return its file.

    obdeect accepts wavelengths that are integers in [0, 1000) nm; other rows are dropped (the
    table ends at 1000 nm, where no photon is simulated). Lines that do not start with a number
    are copied.

    Parameters
    ----------
    transmission_file : Path
        Atmospheric transmission table (extinction-table format).
    directory : Path
        Directory of the written table (created if it does not exist).

    Returns
    -------
    Path
        Transmission table written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in Path(transmission_file).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens or not _is_number(tokens[0]):
            lines.append(line)
        elif _is_integer_below(tokens[0], OBDEECT_MAX_WAVELENGTH_NM):
            lines.append(" ".join(tokens))
    output = directory / TRANSMISSION_FILE_NAME
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")  # NOSONAR
    return output


def write_zero_transmission(transmission_file, directory):
    """
    Write the transmission table with optical depth 0 at all altitudes and return its file.

    Used by sim_telarray with Bypass_Optics 1 or 2 when an upstream stage applies the
    atmospheric extinction: sim_telarray then reads this table and applies no extinction. The
    wavelengths of the table are kept; lines that do not start with a number are copied.

    Parameters
    ----------
    transmission_file : Path
        Atmospheric transmission table (extinction-table format).
    directory : Path
        Directory of the written table (created if it does not exist).

    Returns
    -------
    Path
        Zero-depth transmission table written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in Path(transmission_file).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens and _is_number(tokens[0]):
            lines.append(" ".join([tokens[0], *["0"] * (len(tokens) - 1)]))
        else:
            lines.append(line)
    output = directory / ZERO_TRANSMISSION_FILE_NAME
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")  # NOSONAR
    return output
