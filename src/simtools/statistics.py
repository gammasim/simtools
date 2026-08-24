"""Generic statistical comparison helpers."""

import numpy as np
from scipy import stats
from scipy.spatial import distance


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


def compare_histogram_counts(counts1, counts2, metric="ks", bin_edges=None):
    """Compare two aligned count distributions with a selected metric.

    Parameters
    ----------
    counts1 : array-like
        Baseline distribution counts.
    counts2 : array-like
        Candidate distribution counts.
    metric : {"ks", "jensen_shannon", "wasserstein"}, optional
        Metric used for the comparison. ``"ks"`` compares cumulative counts,
        ``"jensen_shannon"`` compares unordered categories, and
        ``"wasserstein"`` compares ordered support values.
    bin_edges : array-like, optional
        Bin edges for the Wasserstein comparison. Required for that metric.

    Returns
    -------
    dict
        Metric result with ``valid`` and ``reason`` fields. Empty distributions
        return an invalid result; malformed distributions raise ``ValueError``.

    Raises
    ------
    ValueError
        If the count arrays, metric, or bin edges are invalid.
    """
    supported_metrics = {"ks", "jensen_shannon", "wasserstein"}
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported histogram comparison metric: {metric}")

    counts1, counts2 = _validate_count_arrays(counts1, counts2)
    support = _validate_bin_edges(bin_edges, counts1.size) if metric == "wasserstein" else None
    if np.sum(counts1) <= 0 or np.sum(counts2) <= 0:
        return _comparison_result(metric)

    normalized1 = counts1 / np.sum(counts1)
    normalized2 = counts2 / np.sum(counts2)
    if metric == "ks":
        value = np.max(np.abs(np.cumsum(normalized1) - np.cumsum(normalized2)))
    elif metric == "jensen_shannon":
        value = distance.jensenshannon(normalized1, normalized2, base=2.0)
    elif metric == "wasserstein":
        centers = 0.5 * (support[:-1] + support[1:])
        value = stats.wasserstein_distance(
            centers,
            centers,
            u_weights=counts1,
            v_weights=counts2,
        )
    return _comparison_result(metric, value=value)


def _comparison_result(metric, value=None, pvalue=None):
    """Build a serializable comparison result with one metric value."""
    result = {
        "metric": metric,
        "value": None if value is None else float(value),
        "valid": value is not None,
        "reason": "ok" if value is not None else "insufficient_data",
    }
    if pvalue is not None:
        result["pvalue"] = float(pvalue)
    return result


def _validate_count_arrays(counts1, counts2):
    """Return validated one-dimensional count arrays."""
    counts1 = np.asarray(counts1, dtype=float)
    counts2 = np.asarray(counts2, dtype=float)
    if counts1.ndim != 1 or counts2.ndim != 1 or counts1.shape != counts2.shape:
        raise ValueError("Histogram count arrays must be one-dimensional and have equal shape.")
    if not np.all(np.isfinite(counts1)) or not np.all(np.isfinite(counts2)):
        raise ValueError("Histogram count arrays must contain only finite values.")
    if np.any(counts1 < 0) or np.any(counts2 < 0):
        raise ValueError("Histogram count arrays must not contain negative values.")
    return counts1, counts2


def _validate_bin_edges(bin_edges, count_size):
    """Return validated bin edges for a histogram with ``count_size`` bins."""
    if bin_edges is None:
        raise ValueError("Bin edges are required for a Wasserstein histogram comparison.")
    bin_edges = np.asarray(bin_edges, dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size != count_size + 1:
        raise ValueError("Bin edges must contain exactly one more value than histogram counts.")
    if not np.all(np.isfinite(bin_edges)) or np.any(np.diff(bin_edges) <= 0):
        raise ValueError("Bin edges must be finite and strictly increasing.")
    return bin_edges


def compare_samples_with_statistics(baseline_samples, candidate_samples):
    """Compute the KS statistic and p-value for two sample arrays.

    Parameters
    ----------
    baseline_samples : np.ndarray
        Baseline sample values.
    candidate_samples : np.ndarray
        Candidate sample values.
    Returns
    -------
    dict
        KS statistic and p-value in the standard comparison-result format.
    """
    baseline_samples = np.asarray(baseline_samples)
    candidate_samples = np.asarray(candidate_samples)
    if baseline_samples.size == 0 or candidate_samples.size == 0:
        return _comparison_result("ks")
    ks_stat, ks_pval = ks_test_samples(baseline_samples, candidate_samples)
    return _comparison_result("ks", value=ks_stat, pvalue=ks_pval)
