"""Tests for histogram array helpers."""

import numpy as np
import pytest

from simtools.histogram.histogram import compute_bin_edges, normalized_counts, rebin_counts


def test_compute_bin_edges_empty_and_constant_values():
    np.testing.assert_allclose(compute_bin_edges([], 2), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(compute_bin_edges([3.0, 3.0], 2), [3.0, 3.5, 4.0])


def test_compute_bin_edges_linear_and_logarithmic():
    np.testing.assert_allclose(compute_bin_edges([1.0, 5.0], 2), [1.0, 3.0, 5.0])
    np.testing.assert_allclose(compute_bin_edges([1.0, 100.0], 2, scale="log"), [1.0, 10.0, 100.0])


def test_rebin_counts_distributes_counts_by_overlap():
    rebinned = rebin_counts(np.array([4.0]), np.array([0.0, 4.0]), np.array([0.0, 1.0, 3.0, 4.0]))
    np.testing.assert_allclose(rebinned, [1.0, 2.0, 1.0])
    assert np.sum(rebinned) == pytest.approx(4.0)


@pytest.mark.parametrize("cumulative", [False, True])
def test_normalized_counts(cumulative):
    values, errors = normalized_counts(np.array([1.0, 3.0]), cumulative=cumulative)
    assert values.shape == (2,)
    assert errors.shape == (2,)
    if cumulative:
        np.testing.assert_allclose(values, np.array([0.25, 1.0]))
        np.testing.assert_array_equal(errors, np.array([0.0, 0.0]))
    else:
        np.testing.assert_allclose(values, np.array([0.25, 0.75]))
        np.testing.assert_allclose(errors, np.sqrt(np.array([1.0, 3.0])) / 4.0)


def test_normalized_counts_zero_total():
    values, errors = normalized_counts(np.array([0.0, 0.0]))
    np.testing.assert_array_equal(values, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(errors, np.array([0.0, 0.0]))
