import numpy as np

from simtools.statistics import (
    chi2_test_histograms,
    compare_samples_with_statistics,
)


def test_compare_samples_with_statistics_returns_statistics():
    """Test KS/Chi2 statistics for sample-based comparisons."""
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([1.0, 2.0, 2.5, 4.0])
    bin_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = compare_samples_with_statistics(baseline, candidate, bin_edges)

    assert result["ks_statistic"] is not None
    assert result["ks_pvalue"] is not None
    assert result["chi2_statistic"] is not None
    assert result["chi2_pvalue"] is not None
    assert result["valid"]
    assert result["reason"] == "ok"
    assert result["bin_edges"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_chi2_test_histograms_handles_empty_counts():
    """Test aligned histogram statistics return invalid for empty counts."""
    baseline = np.array([0.0, 0.0, 0.0])
    candidate = np.array([1.0, 0.0, 0.0])

    chi2_stat, chi2_pval, valid, reason = chi2_test_histograms(baseline, candidate)

    assert chi2_stat is None
    assert chi2_pval is None
    assert valid is False
    assert reason == "insufficient_data"


def test_chi2_test_histograms_handles_outside_support_counts():
    """Test Chi2 invalid_expected_counts when candidate counts are outside baseline support."""
    baseline = np.array([1.0, 0.0, 0.0])
    candidate = np.array([0.0, 1.0, 0.0])

    chi2_stat, chi2_pval, valid, reason = chi2_test_histograms(baseline, candidate)

    assert chi2_stat is None
    assert chi2_pval is None
    assert valid is False
    assert reason == "invalid_expected_counts"
