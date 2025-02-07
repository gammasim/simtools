"""Calculate the thresholds for energy, radial distance, and viewcone."""

import numpy as np


class ThresholdCalculator:
    """
    Class to compute thresholds/limits for energy, radial distance, and viewcone.

    Parameters
    ----------
    hdf5_file : list of astropy.table.Table
        The list of tables containing the event data.
    """

    def __init__(self, hdf5_file):
        """
        Initialize the ThresholdCalculator with the given HDF5 file.

        Parameters
        ----------
        hdf5_file : list of astropy.table.Table
            The list of tables containing the event data.
        """
        self.angle_to_observing_position__triggered_showers_ = None
        self.event_weight__ra3d__log10_e__ = None

        for table in hdf5_file:
            if (
                "Title" in table.meta
                and table.meta["Title"] == "angle_to_observing_position__triggered_showers_"
            ):
                self.angle_to_observing_position__triggered_showers_ = table
            elif "Title" in table.meta and table.meta["Title"] == "event_weight__ra3d__log10_e__":
                self.event_weight__ra3d__log10_e__ = table

    def compute_threshold(
        self, event_weight_array, bin_edges, loss_fraction, axis=0, limit_type="lower"
    ):
        """
        Compute the threshold based on the loss fraction.

        Parameters
        ----------
        event_weight_array : np.ndarray
            Array of event weights.
        bin_edges : np.ndarray
            Array of bin edges.
        loss_fraction : float
            Fraction of events to be lost.
        axis : int, optional
            Axis along which to sum the event weights. Default is 0.
        limit_type : str, optional
            Type of limit ('lower' or 'upper'). Default is 'lower'.

        Returns
        -------
        int
            Bin index where the threshold is reached.
        float
            Bin edge value corresponding to the threshold.
        """
        projection = np.sum(event_weight_array, axis=axis)
        bin_edge_value = None
        cumulative_sum = None
        if limit_type == "upper":
            cumulative_sum = np.cumsum(projection)

        elif limit_type == "lower":
            cumulative_sum = np.cumsum(projection[::-1])

        total_events = np.sum(projection)
        threshold = (1 - loss_fraction) * total_events
        bin_index = np.searchsorted(cumulative_sum, threshold)
        if limit_type == "upper":
            bin_edge_value = bin_edges[bin_index]
        elif limit_type == "lower":
            bin_edge_value = bin_edges[-bin_index]
        return bin_index, bin_edge_value

    def compute_lower_energy_limit(self, loss_fraction):
        """
        Compute the lower energy limit based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        float
            Lower energy limit.
        """
        event_weight_array = np.column_stack(
            [
                self.event_weight__ra3d__log10_e__[name]
                for name in self.event_weight__ra3d__log10_e__.dtype.names
            ]
        )
        bin_edges = self.event_weight__ra3d__log10_e__.meta["y_bin_edges"]
        _, lower_bin_edge_value = self.compute_threshold(
            event_weight_array, bin_edges, loss_fraction, axis=0, limit_type="lower"
        )
        return 10**lower_bin_edge_value

    def compute_upper_radial_distance(self, loss_fraction):
        """
        Compute the upper radial distance based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        float
            Upper radial distance.
        """
        event_weight_array = np.column_stack(
            [
                self.event_weight__ra3d__log10_e__[name]
                for name in self.event_weight__ra3d__log10_e__.dtype.names
            ]
        )
        bin_edges = self.event_weight__ra3d__log10_e__.meta["y_bin_edges"]
        _, upper_bin_edge_value = self.compute_threshold(
            event_weight_array, bin_edges, loss_fraction, axis=1, limit_type="upper"
        )
        return upper_bin_edge_value

    def compute_viewcone(self, loss_fraction):
        """
        Compute the viewcone based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        float
            Viewcone value.
        """
        angle_to_observing_position__triggered_showers = np.column_stack(
            [
                self.angle_to_observing_position__triggered_showers_[name]
                for name in self.angle_to_observing_position__triggered_showers_.dtype.names
            ]
        )
        bin_edges = self.angle_to_observing_position__triggered_showers_.meta["x_bin_edges"]
        _, upper_bin_edge_value = self.compute_threshold(
            angle_to_observing_position__triggered_showers,
            bin_edges,
            loss_fraction,
            axis=0,
            limit_type="upper",
        )
        return upper_bin_edge_value
