"""Histogram construction, accumulation, and comparison."""

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


class Histogram:
    """N-dimensional numpy histogram with bin edges and optional metadata.

    Parameters
    ----------
    edges : list of numpy.ndarray
        One bin-edge array per axis.
    values : array-like, optional
        Bin values. Defaults to zeros of the shape defined by edges.
    metadata : dict, optional
        Metadata associated with the histogram.
    axis_names : list[str], optional
        One name per histogram axis.
    axis_units : list[str | None], optional
        One unit per histogram axis, or ``None`` for unitless axes.
    """

    def __init__(self, edges, values=None, metadata=None, axis_names=None, axis_units=None):
        self.edges = [np.asarray(axis_edges) for axis_edges in edges]
        self.axis_names = self._axis_metadata(axis_names, "axis_names", str)
        self.axis_units = self._axis_metadata(axis_units, "axis_units", (str, type(None)))
        shape = tuple(len(axis_edges) - 1 for axis_edges in self.edges)
        dtype = np.int64 if len(shape) == 1 else float
        self.values = np.zeros(shape, dtype=dtype) if values is None else np.asarray(values)
        if self.values.shape != shape:
            raise ValueError(f"Histogram values have shape {self.values.shape}; expected {shape}.")
        self.metadata = dict(metadata) if metadata is not None else {}

    def _axis_metadata(self, values, name, value_type):
        """Validate and copy optional metadata associated with each axis."""
        if values is None:
            return None
        if not isinstance(values, list):
            raise ValueError(f"{name} must be a list with one value per histogram axis.")
        if len(values) != len(self.edges):
            raise ValueError(f"{name} must contain one value per histogram axis.")
        if not all(isinstance(value, value_type) for value in values):
            raise ValueError(f"Every {name} entry must be {value_type}.")
        return list(values)

    @classmethod
    def from_edges(cls, edges, values=None, metadata=None, axis_names=None, axis_units=None):
        """Create a histogram from one bin-edge array per axis.

        Parameters
        ----------
        edges : list of numpy.ndarray
            Bin edges for each axis.
        values : array-like, optional
            Initial bin values, defaulting to zeros.
        metadata : dict, optional
            Histogram metadata.
        axis_names : list[str], optional
            One name per histogram axis.
        axis_units : list[str | None], optional
            One unit per histogram axis.

        Returns
        -------
        Histogram
            Histogram with the supplied binning and values.
        """
        return cls(
            edges,
            values=values,
            metadata=metadata,
            axis_names=axis_names,
            axis_units=axis_units,
        )

    @property
    def ndim(self):
        """Return the number of histogram axes."""
        return self.values.ndim

    @property
    def shape(self):
        """Return the number of bins along each axis."""
        return self.values.shape

    def fill(self, *columns):
        """Add event counts from one array of coordinates per axis.

        Parameters
        ----------
        *columns : array-like
            Coordinate arrays of equal length, one for each axis.
        """
        counts, _ = np.histogramdd(np.column_stack(columns), bins=self.edges)
        if self.ndim == 1:
            counts = counts.astype(np.int64)
        self.values = self.values + counts

    def __add__(self, other):
        """Return the sum of histograms with identical edges."""
        if len(self.edges) != len(other.edges) or not all(
            np.array_equal(first, second) for first, second in zip(self.edges, other.edges)
        ):
            raise ValueError("Histograms must have the same edges to be added.")
        return self.from_edges(
            self.edges,
            self.values + other.values,
            self.metadata,
            self.axis_names,
            self.axis_units,
        )

    def cumulative(self, axis=0, reverse=False, normalize=False):
        """Return cumulative bin values along an axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to accumulate counts.
        reverse : bool, optional
            Accumulate from high to low coordinates.
        normalize : bool, optional
            Divide by the original total of each slice along the axis.

        Returns
        -------
        Histogram
            Cumulative values with the same edges and metadata.
        """
        hist = self.values
        if hist.ndim == 1:
            result = self._calculate_cumulative_1d(hist, reverse)
            if normalize and np.sum(hist) > 0:
                result = result / np.sum(hist)
        else:
            result = self._apply_cumsum_along_axis(hist.copy(), axis, reverse)
            if normalize:
                if not np.issubdtype(result.dtype, np.floating):
                    result = result.astype(float)
                self._normalize_along_axis(result, hist, axis)
        return self.from_edges(self.edges, result, self.metadata, self.axis_names, self.axis_units)

    def efficiency(self, simulated):
        """Return triggered counts divided by simulated counts.

        Parameters
        ----------
        simulated : Histogram
            Simulated event counts with the same binning.

        Returns
        -------
        Histogram
            Efficiency, with zero where the simulated count is zero.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.divide(
                self.values,
                simulated.values,
                out=np.zeros_like(self.values, dtype=float),
                where=simulated.values > 0,
            )
        return self.from_edges(self.edges, values, self.metadata, self.axis_names, self.axis_units)

    @staticmethod
    def _normalize_along_axis(result, hist, axis):
        """
        Normalize a cumulative histogram by the total of each slice along an axis.

        Parameters
        ----------
        result : np.ndarray
            Cumulative histogram to normalize (modified in-place)
        hist : np.ndarray
            Original histogram (for calculating totals)
        axis : int
            Axis along which the cumulative sum was computed
        """
        totals = np.sum(hist, axis=axis, keepdims=True)
        normalized = np.divide(
            result,
            totals,
            out=np.zeros_like(result, dtype=float),
            where=np.broadcast_to(totals > 0, result.shape),
        )
        np.copyto(result, normalized)

    @staticmethod
    def _calculate_cumulative_1d(hist, reverse):
        """Calculate cumulative distribution for 1D histogram."""
        if reverse:
            return np.cumsum(hist[::-1])[::-1]
        return np.cumsum(hist)

    @staticmethod
    def _apply_cumsum_along_axis(hist, axis, reverse):
        """Apply cumulative sum along the specified axis of a 2D histogram."""

        def cumsum_func(arr):
            return np.cumsum(arr[::-1])[::-1] if reverse else np.cumsum(arr)

        return np.apply_along_axis(cumsum_func, axis, hist)
