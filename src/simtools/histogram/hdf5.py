"""HDF5 storage for histogram payloads."""

import json

import numpy as np

from simtools.histogram.histogram import Histogram


def write_histograms(group, histograms):
    """Write named histograms to an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        Group in which to store histogram subgroups.
    histograms : dict[str, Histogram]
        Histograms keyed by their stored names.
    """
    for name, histogram in histograms.items():
        histogram_group = group.create_group(str(name))
        _create_dense_histogram_dataset(histogram_group, "values", histogram.values)
        for axis_index, edges in enumerate(histogram.edges):
            _create_dense_histogram_dataset(
                histogram_group, f"edges_{axis_index}", np.asarray(edges, dtype=float)
            )
        histogram_group.attrs["axis_names"] = json.dumps(histogram.axis_names)
        histogram_group.attrs["axis_units"] = json.dumps(histogram.axis_units)


def read_histograms(group) -> dict[str, Histogram]:
    """Read named histograms from an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        Group containing histogram subgroups.

    Returns
    -------
    dict[str, Histogram]
        Histograms keyed by their stored names.
    """
    histograms = {}
    for name, histogram_group in group.items():
        values = np.asarray(histogram_group["values"])
        edges = [
            np.asarray(histogram_group[f"edges_{axis_index}"], dtype=float)
            for axis_index in range(values.ndim)
        ]
        axis_names = json.loads(histogram_group.attrs.get("axis_names", "null"))
        axis_units = json.loads(histogram_group.attrs.get("axis_units", "null"))
        histograms[name] = Histogram.from_edges(
            edges, values=values, axis_names=axis_names, axis_units=axis_units
        )
    return histograms


def _create_dense_histogram_dataset(hdf5_group, dataset_name, data):
    """Create one compressed dataset for dense histogram payloads."""
    return hdf5_group.create_dataset(
        dataset_name,
        data=data,
        chunks=True,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
    )
