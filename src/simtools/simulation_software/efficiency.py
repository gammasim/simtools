"""Quantum-efficiency envelope of the obdeect chain: photons are thinned before ray tracing.

The atmosphere stage thins photons with an efficiency bound E(lambda) (obdeect ``--envelope qe``).
The bound is at least QE_t / h_t for every telescope type t of the array, so that photons
which sim_telarray would reject are not traced. sim_telarray then applies the table QE_t / E
instead of QE_t; the stream converter declares this with ``--scaled-factor qe``.

sim_telarray's first cut is ``prob = p1 * p2 * p3 * max_qe_rel``, where ``max_qe_rel`` is the
largest pixel efficiency fluctuation of a camera (Gaussian with ``qe_variation``). The scaled
table must stay below the headroom h = 1 / (1 + QE_VARIATION_SIGMAS * qe_variation), so that the
cut does not saturate.

Tables are on integer nm bins, as sim_telarray bins photons by wavelength.
"""

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.model_repository.asset_names import get_simtel_table_file_name

#: Name of the efficiency factor in obdeect commands (``--envelope``, ``--scaled-factor``).
OBDEECT_QE_FACTOR = "qe"
#: Stage option that switches the envelope on (``qe``) or off (``none``).
ENVELOPE_OPTION = "efficiency_envelope"
ENVELOPE_ON = "qe"
ENVELOPE_OFF = "none"
QE_PARAMETER = "quantum_efficiency"
QE_VARIATION_PARAMETER = "qe_variation"
#: Standard deviations of the pixel QE variation left as margin by the headroom, so that the
#: largest pixel fluctuation of a camera does not saturate sim_telarray's first cut.
QE_VARIATION_SIGMAS = 6.0
#: Bound of bins where all telescope types have zero QE (a bound must be positive).
ENVELOPE_FLOOR = 1e-6
#: Second node of each nm bin: the bound is constant on [n, n + 1) nm.
NODE_WIDTH_NM = 0.999999
#: Integer nm bins of the Cherenkov band (CORSIKA EVTH words 96-97, as sim_telarray samples it).
WAVELENGTH_GRID_NM = np.arange(100, 2001)
ENVELOPE_FILE_NAME = "qe-envelope.dat"
SCALED_TABLE_DIRECTORY = "qe-table"
EFFICIENCY_DIRECTORY = "efficiency"


@dataclass(frozen=True)
class TelescopeEfficiency:
    """Quantum efficiency of one telescope type on the wavelength grid.

    Parameters
    ----------
    name : str
        Telescope model name.
    table_name : str
        File name of the sim_telarray QE table (as referenced by the telescope configuration).
    qe : numpy.ndarray
        Quantum efficiency on the wavelength grid.
    headroom : float
        Largest allowed value of the scaled table (see :func:`headroom`).
    """

    name: str
    table_name: str
    qe: np.ndarray
    headroom: float


@dataclass(frozen=True)
class ArrayEnvelope:
    """Efficiency bound of an array and the telescope types it bounds.

    Parameters
    ----------
    bound : numpy.ndarray
        Efficiency bound on the wavelength grid.
    telescopes : tuple of TelescopeEfficiency
        Quantum efficiency of each telescope type of the array.
    """

    bound: np.ndarray
    telescopes: tuple


def headroom(qe_variation):
    """Return the largest allowed value of the scaled QE table, 1 / (1 + 6 * qe_variation)."""
    return 1.0 / (1.0 + QE_VARIATION_SIGMAS * float(qe_variation))


def envelope_setting(options, default=ENVELOPE_OFF):
    """
    Return whether the efficiency envelope is on, from the options of the atmosphere stage.

    The values of the option are checked by the software catalog.

    Parameters
    ----------
    options : dict
        Stage options; ``efficiency_envelope`` is ``qe`` or ``none``.
    default : str
        Value used when the option is not given.

    Returns
    -------
    bool
        True if the envelope is on.
    """
    return options.get(ENVELOPE_OPTION, default) == ENVELOPE_ON


def sim_telarray_qe(table, grid=WAVELENGTH_GRID_NM):
    """
    Return the quantum efficiency of a model table on a wavelength grid.

    The table is interpolated linearly, as sim_telarray reads the table.

    Parameters
    ----------
    table : astropy.table.Table
        Table with columns ``wavelength`` (nm) and ``efficiency``.
    grid : numpy.ndarray
        Wavelengths [nm].

    Returns
    -------
    numpy.ndarray
        Quantum efficiency at the grid wavelengths.
    """
    wavelength = u.Quantity(table["wavelength"], u.nm).value
    efficiency = np.asarray(table["efficiency"], dtype=float)
    order = np.argsort(wavelength)
    return np.interp(grid, wavelength[order], efficiency[order])


def telescope_efficiencies(array_model):
    """
    Return the quantum efficiency of each telescope type of an array (one entry per type).

    Parameters
    ----------
    array_model : ArrayModel
        Array model; its telescope models are read.

    Returns
    -------
    tuple of TelescopeEfficiency
        One entry per telescope name, in the order of the array.

    Raises
    ------
    ValueError
        If the QE table of a telescope has no sim_telarray file name.
    """
    efficiencies = {}
    for telescope in array_model.telescope_models.values():
        if telescope.name in efficiencies:
            continue
        table_name = get_simtel_table_file_name(telescope.parameters[QE_PARAMETER])
        if table_name is None:
            raise ValueError(
                f"The {QE_PARAMETER} table of telescope {telescope.name} has no sim_telarray "
                "file name; the efficiency envelope needs an ECSV table with a shared name."
            )
        efficiencies[telescope.name] = TelescopeEfficiency(
            name=telescope.name,
            table_name=table_name,
            qe=sim_telarray_qe(telescope.get_parameter_table(QE_PARAMETER)),
            headroom=headroom(telescope.get_parameter_value(QE_VARIATION_PARAMETER)),
        )
    return tuple(efficiencies.values())


def envelope_bound(telescopes):
    """
    Return the efficiency bound of telescope types: the largest QE / headroom per wavelength.

    Bins where no telescope type has QE use ``ENVELOPE_FLOOR``.

    Parameters
    ----------
    telescopes : iterable of TelescopeEfficiency
        Telescope types of the array.

    Returns
    -------
    numpy.ndarray
        Bound on the wavelength grid.
    """
    bound = np.max([telescope.qe / telescope.headroom for telescope in telescopes], axis=0)
    return np.where(bound > 0, bound, ENVELOPE_FLOOR)


def array_envelope(array_model):
    """
    Return the efficiency bound of an array and its telescope types.

    One bound per array model bounds all telescope types of the array, so that one
    propagation run serves all of them.

    Parameters
    ----------
    array_model : ArrayModel
        Array model.

    Returns
    -------
    ArrayEnvelope
        Bound and telescope types.
    """
    telescopes = telescope_efficiencies(array_model)
    return ArrayEnvelope(bound=envelope_bound(telescopes), telescopes=telescopes)


def write_envelope(path, bound, grid=WAVELENGTH_GRID_NM):
    """
    Write the bound as two-column text (nm, value) for obdeect ``--envelope``.

    Each nm bin has nodes at n and n + NODE_WIDTH_NM, so the bound is constant on the bin.

    Parameters
    ----------
    path : Path
        Output file.
    bound : numpy.ndarray
        Bound on the wavelength grid.
    grid : numpy.ndarray
        Wavelengths [nm] of the bins.

    Returns
    -------
    Path
        Output file.
    """
    lines = []
    for wavelength, value in zip(grid, bound):
        lines.append(f"{wavelength:.6f} {value:.9g}")
        lines.append(f"{wavelength + NODE_WIDTH_NM:.6f} {value:.9g}")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_scaled_tables(directory, envelope, grid=WAVELENGTH_GRID_NM):
    """
    Write the sim_telarray QE table of each telescope type, QE / bound, into a directory.

    The files have the names of the telescope configurations' QE tables, so that sim_telarray
    reads them when the directory is on its include path before the model directory.

    Parameters
    ----------
    directory : Path
        Output directory (created).
    envelope : ArrayEnvelope
        Bound and telescope types of the array.
    grid : numpy.ndarray
        Wavelengths [nm] of the table rows.

    Returns
    -------
    Path
        Output directory.

    Raises
    ------
    ValueError
        If two telescope types use one table name with different efficiencies.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for telescope in envelope.telescopes:
        scaled = telescope.qe / envelope.bound
        if telescope.table_name in written and not np.array_equal(
            written[telescope.table_name], scaled
        ):
            raise ValueError(
                f"Telescope types share the QE table {telescope.table_name} with different "
                "efficiencies; the efficiency envelope cannot scale them."
            )
        written[telescope.table_name] = scaled
        lines = "".join(f"{wavelength} {value:.9g}\n" for wavelength, value in zip(grid, scaled))
        (directory / telescope.table_name).write_text(lines, encoding="utf-8")
    return directory


def envelope_directory(context, run_number, model_version):
    """Return the directory of the efficiency envelope files of a run and model version."""
    return context.paths.get_output_directory(
        [EFFICIENCY_DIRECTORY, f"run{run_number:06d}", str(model_version)]
    )
