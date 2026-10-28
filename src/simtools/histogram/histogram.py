"""Array helpers for histogram construction and comparison."""

import numpy as np


def compute_bin_edges(values, bins, scale="linear"):
    """Return robust bin edges for the given values."""
    if len(values) == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        max_value = min_value + 1.0

    if scale == "log":
        min_value = max(min_value, np.finfo(float).tiny)
        return np.logspace(np.log10(min_value), np.log10(max_value), bins + 1)
    return np.linspace(min_value, max_value, bins + 1)


def rebin_counts(counts, source_edges, target_edges):
    """Rebin counts by distributing each source bin over overlapping target bins.

    Counts are assumed to be uniformly distributed within each source bin. This
    preserves the total count and avoids assigning a complete source bin to one
    target bin when the target edges split it.
    """
    rebinned = np.zeros(len(target_edges) - 1, dtype=float)
    for count, source_low, source_high in zip(counts, source_edges[:-1], source_edges[1:]):
        source_width = source_high - source_low
        target_start = np.searchsorted(target_edges, source_low, side="right") - 1
        target_stop = np.searchsorted(target_edges, source_high, side="left")
        for target_index in range(max(0, target_start), min(len(rebinned), target_stop + 1)):
            overlap = min(source_high, target_edges[target_index + 1]) - max(
                source_low, target_edges[target_index]
            )
            if overlap > 0:
                rebinned[target_index] += count * overlap / source_width
    return rebinned


def normalized_counts(counts, cumulative=False):
    """Return normalized bin values and Poisson errors for histogram counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        zeros = np.zeros_like(counts, dtype=float)
        return zeros, zeros
    if cumulative:
        values = np.cumsum(counts) / total
        errors = np.zeros_like(values, dtype=float)
        return values, errors
    values = counts / total
    errors = np.sqrt(counts) / total
    return values, errors
