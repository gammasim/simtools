import h5py
import numpy as np

from simtools.histogram.hdf5 import read_histograms, write_histograms
from simtools.histogram.histogram import Histogram


def test_histograms_round_trip_1d_and_3d_with_compression(tmp_test_directory):
    expected = {
        "one_dimensional": Histogram.from_edges(
            [np.array([0.0, 1.0, 3.0])],
            values=np.array([2, 4]),
            axis_names=["energy"],
            axis_units=["TeV"],
        ),
        "three_dimensional": Histogram.from_edges(
            [np.array([0.0, 2.0]), np.array([1.0, 3.0, 4.0]), np.array([-1.0, 0.0])],
            values=np.arange(2).reshape(1, 2, 1),
            axis_names=["x", "y", "z"],
            axis_units=["m", None, "deg"],
        ),
    }
    file_path = tmp_test_directory / "histograms.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        write_histograms(hdf5_file, expected)

    with h5py.File(file_path, "r") as hdf5_file:
        actual = read_histograms(hdf5_file)
        for name, histogram in expected.items():
            for dataset in hdf5_file[name].values():
                assert dataset.compression == "gzip"
                assert dataset.compression_opts == 6
                assert dataset.shuffle
            np.testing.assert_array_equal(actual[name].values, histogram.values)
            assert actual[name].axis_names == histogram.axis_names
            assert actual[name].axis_units == histogram.axis_units
            for actual_edges, expected_edges in zip(
                actual[name].edges, histogram.edges, strict=True
            ):
                np.testing.assert_array_equal(actual_edges, expected_edges)


def test_read_histograms_from_empty_group(tmp_test_directory):
    file_path = tmp_test_directory / "empty.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        assert read_histograms(hdf5_file) == {}
