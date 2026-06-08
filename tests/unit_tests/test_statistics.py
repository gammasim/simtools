import numpy as np

from simtools.statistics import compare_samples_with_statistics


def test_compare_samples_with_statistics_returns_statistics():
    """Test KS statistics for sample-based comparisons."""
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([1.0, 2.0, 2.5, 4.0])
    bin_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = compare_samples_with_statistics(baseline, candidate, bin_edges)

    assert result["ks_statistic"] is not None
    assert result["ks_pvalue"] is not None
    assert result["valid"]
    assert result["reason"] == "ok"
    assert result["bin_edges"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_compare_samples_with_statistics_empty_samples():
    """Test KS test returns None for empty samples."""
    result = compare_samples_with_statistics([], [], [0.0, 1.0])
    assert result["ks_statistic"] is None
    assert result["ks_pvalue"] is None
