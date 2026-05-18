"""Generic statistical comparison helpers for simtools and other applications."""

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


def chi2_test_histograms(counts1, counts2):
    """Compute the Chi2 statistic and p-value for two aligned histogram count arrays.

    Parameters
    ----------
    counts1 : np.ndarray
        First histogram counts (expected/baseline).
    counts2 : np.ndarray
        Second histogram counts (observed/candidate).

    Returns
    -------
    chi2_statistic : float
        Chi2 statistic value.
    chi2_pvalue : float
        Chi2 test p-value.
    valid : bool
        Whether the test is valid (enough data, no zero division).
    reason : str
        Reason for invalidity if not valid.
    """
    counts1 = np.asarray(counts1, dtype=float)
    counts2 = np.asarray(counts2, dtype=float)
    total1 = float(np.sum(counts1))
    total2 = float(np.sum(counts2))
    if total1 <= 0 or total2 <= 0:
        return None, None, False, "insufficient_data"
    support_mask = counts1 > 0
    if not np.any(support_mask):
        return None, None, False, "zero_baseline_counts"
    observed = counts2[support_mask]
    observed_total = float(np.sum(observed))
    baseline_supported_total = float(np.sum(counts1[support_mask]))
    if observed_total <= 0 or baseline_supported_total <= 0:
        return None, None, False, "invalid_expected_counts"
    expected = counts1[support_mask] * (observed_total / baseline_supported_total)
    if np.sum(expected) <= 0:
        return None, None, False, "invalid_expected_counts"
    chi2 = stats.chisquare(f_obs=observed, f_exp=expected)
    return float(chi2.statistic), float(chi2.pvalue), True, "ok"


def compare_samples_with_statistics(baseline_samples, candidate_samples, bin_edges):
    """Compute KS and Chi2 statistics for two sample arrays using provided bin edges.

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
        KS and Chi2 statistics, p-values, and histogram metadata.
    """
    baseline_samples = np.asarray(baseline_samples)
    candidate_samples = np.asarray(candidate_samples)
    result = {
        "ks_statistic": None,
        "ks_pvalue": None,
        "chi2_statistic": None,
        "chi2_pvalue": None,
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
    chi2_stat, chi2_pval, valid, reason = chi2_test_histograms(baseline_counts, candidate_counts)
    result["chi2_statistic"] = chi2_stat
    result["chi2_pvalue"] = chi2_pval
    result["valid"] = valid
    result["reason"] = reason
    result["baseline_counts"] = baseline_counts.astype(int).tolist()
    result["candidate_counts"] = candidate_counts.astype(int).tolist()
    result["bin_edges"] = np.asarray(bin_edges, dtype=float).tolist()
    return result
