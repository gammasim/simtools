"""Helper module for reading and writing in hd5 format."""

import logging
from pathlib import PosixPath

import astropy.units as u
import tables
from astropy.table import Table
from ctapipe.io import read_table

from simtools.utils.names import sanitize_name

__all__ = [
    "fill_hdf5_table",
    "read_hdf5",
]

_logger = logging.getLogger(__name__)


def fill_hdf5_table(hist, x_bin_edges, y_bin_edges, x_label, y_label, meta_data):
    """
    Create and fill an hdf5 table with the histogram information.

    It works for both 1D and 2D distributions.

    Parameters
    ----------
    hist: numpy.ndarray
        The counts of the histograms.
    x_bin_edges: numpy.array
        The x bin edges of the histograms.
    y_bin_edges: numpy.array
        The y bin edges of the histograms.
        Use None for 1D histograms.
    x_label: str
        X bin edges label.
    y_label: str
        Y bin edges label.
        Use None for 1D histograms.
    meta_data: dict
        Dictionary with the histogram metadata.
    """
    validate_histogram(hist, y_bin_edges)

    meta_data["x_bin_edges"] = x_bin_edges
    meta_data["x_bin_edges_unit"] = (
        x_bin_edges.unit if isinstance(x_bin_edges, u.Quantity) else u.dimensionless_unscaled
    )
    if y_bin_edges is not None:
        meta_data["y_bin_edges"] = y_bin_edges
        meta_data["y_bin_edges_unit"] = (
            y_bin_edges.unit if isinstance(y_bin_edges, u.Quantity) else u.dimensionless_unscaled
        )

    if hist.ndim == 1:
        if x_label is not None:
            names = sanitize_name(x_label)
        try:
            names = meta_data["Title"]
        except KeyError:
            _logger.warning("Title not found in metadata.")

        table = Table(
            [
                x_bin_edges[:-1],
                hist,
            ],
            names=(names, sanitize_name("Values")),
            meta=meta_data,
        )
    else:
        if y_label is not None:
            names = [
                f"{sanitize_name(y_label).split('__')[0]}_{i}" for i in range(len(y_bin_edges[:-1]))
            ]
        try:
            names = [
                f"{(meta_data['Title']).split('__')[0]}_{sanitize_name(y_label)}_{i}"
                for i in range(len(y_bin_edges[:-1]))
            ]
        except KeyError:
            _logger.warning("Title not found in metadata.")
            names = [
                f"{sanitize_name(y_label).split('__')[0]}_{i}" for i in range(len(y_bin_edges[:-1]))
            ]

        table = Table(
            [hist[i, :] for i in range(len(y_bin_edges[:-1]))],
            names=names,
            meta=meta_data,
        )

    return table


def validate_histogram(hist, y_bin_edges):
    """Validate histogram dimensions and y_bin_edges consistency.

    Parameters
    ----------
    hist (np.ndarray): The histogram array, expected to be 1D or 2D.
    y_bin_edges (array-like or None): Bin edges for the second dimension (if applicable).

    Raises
    ------
    ValueError: If histogram dimensions are invalid or inconsistent with y_bin_edges.
    """
    if hist.ndim not in (1, 2):
        raise ValueError("Histogram must be either 1D or 2D.")

    if hist.ndim == 1 and y_bin_edges is not None:
        raise ValueError("y_bin_edges should be None for 1D histograms.")

    if hist.ndim == 2 and y_bin_edges is None:
        raise ValueError("y_bin_edges should not be None for 2D histograms.")


def read_hdf5(hdf5_file_name):
    """
    Read a hdf5 output file.

    Parameters
    ----------
    hdf5_file_name: str or Path
        Name or Path of the hdf5 file to read from.

    Returns
    -------
    list
        The list with the astropy.Table instances for the various 1D and 2D histograms saved
        in the hdf5 file.
    """
    if isinstance(hdf5_file_name, PosixPath):
        hdf5_file_name = hdf5_file_name.absolute().as_posix()

    tables_list = []

    with tables.open_file(hdf5_file_name, mode="r") as file:
        for node in file.walk_nodes("/", "Table"):
            table_path = node._v_pathname  # pylint: disable=protected-access
            table = read_table(hdf5_file_name, table_path)
            tables_list.append(table)
    return tables_list
