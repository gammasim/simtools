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

    results_none = compare_samples_with_statistics([], [], bin_edges)
    assert results_none["ks_statistic"] is None


def test_ks_test_samples_handles_empty_samples():
    """Test KS test returns None for empty samples."""
    ks_stat, ks_pval = (
        compare_samples_with_statistics([], [], [0.0, 1.0])["ks_statistic"],
        compare_samples_with_statistics([], [], [0.0, 1.0])["ks_pvalue"],
    )
    assert ks_stat is None
    assert ks_pval is None
