"""Generic statistical comparison helpers."""

import numpy as np
from scipy import stats


def ks_test_samples(samples1, samples2):
    """Compute the Kolmogorov-Smirnov (KS) statistic and p-value for two 1D sample arrays.

    Parameters
    ----------
    samples1 : np.ndarray
        First sample array.
    samples2 : np.ndarray
        Second sample array.

    Returns
    -------
    ks_statistic : float
        KS statistic value.
    ks_pvalue : float
        KS test p-value.
    """
    samples1 = np.asarray(samples1)
    samples2 = np.asarray(samples2)
    if samples1.size == 0 or samples2.size == 0:
        return None, None
    ks_test = stats.ks_2samp(samples1, samples2)
    return float(ks_test.statistic), float(ks_test.pvalue)


def compare_samples_with_statistics(baseline_samples, candidate_samples, bin_edges):
    """Compute KS statistics for two sample arrays using provided bin edges.

    Parameters
    ----------
    baseline_samples : np.ndarray
        Baseline sample values.
    candidate_samples : np.ndarray
        Candidate sample values.
    bin_edges : np.ndarray
        Histogram bin edges for Chi2 calculation.

    Returns
    -------
    dict
        KS statistics, p-values, and histogram metadata.
    """
    baseline_samples = np.asarray(baseline_samples)
    candidate_samples = np.asarray(candidate_samples)
    result = {
        "ks_statistic": None,
        "ks_pvalue": None,
        "valid": False,
        "reason": "insufficient_data",
    }
    if baseline_samples.size == 0 or candidate_samples.size == 0:
        return result
    ks_stat, ks_pval = ks_test_samples(baseline_samples, candidate_samples)
    result["ks_statistic"] = ks_stat
    result["ks_pvalue"] = ks_pval
    baseline_counts, _ = np.histogram(baseline_samples, bins=bin_edges)
    candidate_counts, _ = np.histogram(candidate_samples, bins=bin_edges)
    result["valid"] = True
    result["reason"] = "ok"
    result["baseline_counts"] = baseline_counts.astype(int).tolist()
    result["candidate_counts"] = candidate_counts.astype(int).tolist()
    result["bin_edges"] = np.asarray(bin_edges, dtype=float).tolist()
    return result
