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

    # Complement metadata
    if x_label is not None:
        meta_data["x_bin_edges"] = sanitize_name(x_label)
    meta_data["x_bin_edges_unit"] = (
        x_bin_edges.unit if isinstance(x_bin_edges, u.Quantity) else u.dimensionless_unscaled
    )

    if y_bin_edges is not None:
        if y_label is not None:
            meta_data["y_bin_edges"] = sanitize_name(y_label)
            names = [
                f"{meta_data['y_bin_edges'].split('__')[0]}_{i}"
                for i in range(len(y_bin_edges[:-1]))
            ]
        else:
            names = [
                f"{meta_data['Title'].split('__')[0]}_{i}" for i in range(len(y_bin_edges[:-1]))
            ]
        meta_data["y_bin_edges_unit"] = (
            y_bin_edges.unit if isinstance(y_bin_edges, u.Quantity) else u.dimensionless_unscaled
        )

        table = Table(
            [hist[i, :] for i in range(len(y_bin_edges[:-1]))],
            names=names,
            meta=meta_data,
        )

    else:
        if x_label is not None:
            meta_data["x_bin_edges"] = sanitize_name(x_label)
            names = meta_data["x_bin_edges"]
        else:
            names = meta_data["Title"]
        table = Table(
            [
                x_bin_edges[:-1],
                hist,
            ],
            names=(names, sanitize_name("Values")),
            meta=meta_data,
        )
    return table


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
