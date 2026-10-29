"""Tests for histogram objects and array helpers."""

import numpy as np
import pytest

from simtools.histogram.histogram import (
    Histogram,
    compute_bin_edges,
    normalized_counts,
    rebin_counts,
)


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


def test_histogram_fill_and_add():
    edges = [np.array([0, 1, 2]), np.array([0, 2, 4])]
    histogram = Histogram.from_edges(edges, metadata={"quantity": "events"})
    assert histogram.ndim == 2
    assert histogram.shape == (2, 2)
    histogram.fill([0, 1, 2, -1], [0, 2, 4, 1])
    histogram.fill([0], [0])
    np.testing.assert_array_equal(histogram.values, [[2, 0], [0, 2]])
    summed = histogram + histogram
    np.testing.assert_array_equal(summed.values, [[4, 0], [0, 4]])
    np.testing.assert_array_equal(histogram.values, [[2, 0], [0, 2]])
    assert summed.metadata == {"quantity": "events"}
    assert summed.metadata is not histogram.metadata
    with pytest.raises(ValueError, match="same edges"):
        histogram + Histogram.from_edges([np.array([0, 1, 3]), edges[1]])
    with pytest.raises(ValueError, match="same edges"):
        histogram + Histogram.from_edges([edges[0]])


def test_histogram_values_shape():
    with pytest.raises(ValueError, match="shape"):
        Histogram.from_edges([[0, 1, 2]], values=[1])


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_histogram_fill_matches_numpy(ndim):
    edges = [np.array([0.0, 1.0, 3.0])] * ndim
    columns = [np.array([-1.0, 0.0, 0.5, 1.0, 3.0, 4.0])] * ndim
    histogram = Histogram.from_edges(edges)
    histogram.fill(*columns)
    expected, _ = np.histogramdd(np.column_stack(columns), bins=edges)
    expected_dtype = np.dtype(np.int64) if ndim == 1 else np.dtype(float)
    assert histogram.values.dtype == expected_dtype
    np.testing.assert_array_equal(histogram.values, expected)
    histogram.fill(*[[] for _ in edges])
    np.testing.assert_array_equal(histogram.values, expected)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_histogram_cumulative_1d(reverse, normalize):
    histogram = Histogram.from_edges([[0, 1, 2, 3, 4]], values=[1, 2, 3, 4])
    expected = np.array([10, 9, 7, 4] if reverse else [1, 3, 6, 10])
    if normalize:
        expected = expected / 10
    result = histogram.cumulative(reverse=reverse, normalize=normalize)
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(histogram.values, [1, 2, 3, 4])
    np.testing.assert_array_equal(result.edges[0], histogram.edges[0])


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_histogram_cumulative_2d(axis, reverse, normalize):
    values = np.array([[1, 0, 3], [4, 0, 6], [0, 0, 0]])
    histogram = Histogram.from_edges([np.arange(4), np.arange(4)], values=values)
    expected = np.cumsum(np.flip(values, axis) if reverse else values, axis=axis)
    if reverse:
        expected = np.flip(expected, axis)
    if normalize:
        totals = values.sum(axis=axis, keepdims=True)
        expected = np.divide(
            expected, totals, out=np.zeros_like(expected, dtype=float), where=totals > 0
        )
    np.testing.assert_array_equal(
        histogram.cumulative(axis=axis, reverse=reverse, normalize=normalize).values, expected
    )


def test_histogram_efficiency_and_empty_cumulative():
    triggered = Histogram.from_edges([[0, 1, 2, 3]], values=[0, 2, 4])
    simulated = Histogram.from_edges(triggered.edges, values=[0, 4, 8])
    np.testing.assert_array_equal(triggered.efficiency(simulated).values, [0, 0.5, 0.5])
    empty = Histogram.from_edges(triggered.edges)
    np.testing.assert_array_equal(empty.cumulative(normalize=True).values, [0, 0, 0])


def _histogram_from_values(values):
    return Histogram.from_edges([np.arange(size + 1) for size in values.shape], values=values)


def test_calculate_cumulative_histogram():
    # Test 1D histogram
    test_hist_1d = np.array([1, 2, 3, 4])

    # Accumulate one-dimensional counts.
    result_1d_direct = _histogram_from_values(test_hist_1d).cumulative(reverse=False).values
    expected_1d = np.array([1, 3, 6, 10])
    np.testing.assert_array_equal(result_1d_direct, expected_1d)

    # Normal cumulative (left to right)
    result_1d = _histogram_from_values(test_hist_1d).cumulative().values
    expected_1d = np.array([1, 3, 6, 10])
    np.testing.assert_array_equal(result_1d, expected_1d)

    # Reverse cumulative (right to left)
    result_1d_reverse = _histogram_from_values(test_hist_1d).cumulative(reverse=True).values
    expected_1d_reverse = np.array([10, 9, 7, 4])
    np.testing.assert_array_equal(result_1d_reverse, expected_1d_reverse)

    # Test 1D histogram with normalization
    result_1d_normalized = _histogram_from_values(test_hist_1d).cumulative(normalize=True).values
    expected_1d_normalized = np.array([0.1, 0.3, 0.6, 1.0])
    np.testing.assert_allclose(result_1d_normalized, expected_1d_normalized)

    # Test 2D histogram - use float dtype to avoid casting issues
    test_hist_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    # Accumulate along each two-dimensional axis.
    # Test axis=1, reverse=False
    result_axis1_no_reverse = (
        _histogram_from_values(test_hist_2d.copy()).cumulative(axis=1, reverse=False).values
    )
    expected_axis1_no_reverse = np.array([[1, 3, 6], [4, 9, 15]])
    np.testing.assert_array_equal(result_axis1_no_reverse, expected_axis1_no_reverse)

    # Test axis=1, reverse=True
    result_axis1_reverse = (
        _histogram_from_values(test_hist_2d.copy()).cumulative(axis=1, reverse=True).values
    )
    expected_axis1_reverse = np.array([[6, 5, 3], [15, 11, 6]])
    np.testing.assert_array_equal(result_axis1_reverse, expected_axis1_reverse)

    # Test axis=0, reverse=False
    result_axis0_no_reverse = (
        _histogram_from_values(test_hist_2d.copy()).cumulative(axis=0, reverse=False).values
    )
    expected_axis0_no_reverse = np.array([[1, 2, 3], [5, 7, 9]])
    np.testing.assert_array_equal(result_axis0_no_reverse, expected_axis0_no_reverse)

    # Test axis=0, reverse=True
    result_axis0_reverse = (
        _histogram_from_values(test_hist_2d.copy()).cumulative(axis=0, reverse=True).values
    )
    expected_axis0_reverse = np.array([[5, 7, 9], [4, 5, 6]])
    np.testing.assert_array_equal(result_axis0_reverse, expected_axis0_reverse)

    # Accumulate along rows.
    result_2d = _histogram_from_values(test_hist_2d).cumulative(axis=1).values
    expected_2d = np.array([[1, 3, 6], [4, 9, 15]])
    np.testing.assert_array_equal(result_2d, expected_2d)

    # Test 2D histogram with normalization - row normalization
    result_2d_normalized = (
        _histogram_from_values(test_hist_2d).cumulative(normalize=True, axis=1).values
    )
    expected_2d_normalized = np.array([[1 / 6, 3 / 6, 1.0], [4 / 15, 9 / 15, 1.0]])
    np.testing.assert_allclose(result_2d_normalized, expected_2d_normalized)

    # Along axis 0
    result_2d_axis0 = _histogram_from_values(test_hist_2d).cumulative(axis=0).values
    expected_2d_axis0 = np.array([[1, 2, 3], [5, 7, 9]])
    np.testing.assert_array_equal(result_2d_axis0, expected_2d_axis0)

    # With reverse=True
    result_2d_reverse = _histogram_from_values(test_hist_2d).cumulative(reverse=True, axis=1).values
    np.testing.assert_array_equal(result_2d_reverse, np.array([[6, 5, 3], [15, 11, 6]]))


def test_normalized_cumulative_histogram():
    test_hist_2d = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0],
        ],
        dtype=float,
    )

    # Test with explicit axis=1
    result_axis1 = _histogram_from_values(test_hist_2d).cumulative(normalize=True, axis=1).values

    expected_axis1 = np.array([[1 / 6, 3 / 6, 1.0], [4 / 15, 9 / 15, 1.0], [0, 0, 0]])

    np.testing.assert_allclose(result_axis1, expected_axis1, rtol=1e-4)

    # Test with axis=0
    result_axis0 = _histogram_from_values(test_hist_2d).cumulative(axis=0, normalize=True).values

    expected_axis0 = np.array([[1 / 5, 2 / 7, 3 / 9], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    np.testing.assert_allclose(result_axis0, expected_axis0, rtol=1e-4)

    # Test 1D histogram normalization
    test_hist_1d = np.array([10, 20, 30, 40], dtype=float)
    result_1d = _histogram_from_values(test_hist_1d).cumulative(normalize=True).values
    expected_1d = np.array([0.1, 0.3, 0.6, 1.0])
    np.testing.assert_allclose(result_1d, expected_1d)

    # Test normalization with reverse=True
    result_reverse = (
        _histogram_from_values(test_hist_1d).cumulative(reverse=True, normalize=True).values
    )
    expected_reverse = np.array([1.0, 0.9, 0.7, 0.4])
    np.testing.assert_allclose(result_reverse, expected_reverse)
