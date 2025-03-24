"""Calculate CORSIKA thresholds for energy, radial distance, and viewcone."""

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader


class LimitCalculator:
    """
    Compute thresholds for CORSIKA configuration for energy, radial distance, and viewcone.

    Event data is read from the reduced MC event data file.

    Parameters
    ----------
    event_data_file : str
        Path to the HDF5 file containing the event data.
    telescope_list : list, optional
        List of telescope IDs to filter the events (default is None).
    """

    def __init__(self, event_data_file, telescope_list=None):
        """Initialize the LimitCalculator with the given event data file."""
        self._logger = logging.getLogger(__name__)
        self.event_data_file = event_data_file
        self.telescope_list = telescope_list

        self.reader = SimtelIOEventDataReader(event_data_file, telescope_list=[7, 12])
        self.event_data = self.reader.triggered_shower_data
        self.triggered_data = self.reader.triggered_data

    def _compute_limits(self, hist, bin_edges, loss_fraction, limit_type="lower"):
        """
        Compute the limits based on the loss fraction.

        Parameters
        ----------
        hist : np.ndarray
            1D histogram array.
        bin_edges : np.ndarray
            Array of bin edges.
        loss_fraction : float
            Fraction of events to be lost.
        limit_type : str, optional
            Type of limit ('lower' or 'upper'). Default is 'lower'.

        Returns
        -------
        float
            Bin edge value corresponding to the threshold.
        """
        cumulative_sum = np.cumsum(hist) if limit_type == "upper" else np.cumsum(hist[::-1])
        total_events = np.sum(hist)
        threshold = (1 - loss_fraction) * total_events
        bin_index = np.searchsorted(cumulative_sum, threshold)

        return bin_edges[bin_index] if limit_type == "upper" else bin_edges[-bin_index]

    @property
    def energy_bins(self):
        """Return bins for the energy histogram."""
        return np.logspace(
            np.log10(self.event_data.simulated_energy.min()),
            np.log10(self.event_data.simulated_energy.max()),
            1000,
        )

    def compute_lower_energy_limit(self, loss_fraction):
        """
        Compute the lower energy limit in TeV based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Lower energy limit.
        """
        hist, _ = np.histogram(self.event_data.simulated_energy, bins=self.energy_bins)
        return (
            self._compute_limits(hist, self.energy_bins, loss_fraction, limit_type="lower") * u.TeV
        )

    @property
    def core_distance_bins(self):
        """Return bins for the core distance histogram."""
        return np.linspace(
            self.event_data.core_distance_shower.min(),
            self.event_data.core_distance_shower.max(),
            1000,
        )

    def compute_upper_radial_distance(self, loss_fraction):
        """
        Compute the upper radial distance based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Upper radial distance in m.
        """
        hist, _ = np.histogram(self.event_data.core_distance_shower, bins=self.core_distance_bins)
        return (
            self._compute_limits(hist, self.core_distance_bins, loss_fraction, limit_type="upper")
            * u.m
        )

    @property
    def view_cone_bins(self):
        """Return bins for the viewcone histogram."""
        return np.linspace(
            self.triggered_data.angular_distance.min(),
            self.triggered_data.angular_distance.max(),
            1000,
        )

    def compute_viewcone(self, loss_fraction):
        """
        Compute the viewcone based on the event loss fraction.

        The shower IDs of triggered events are used to create a mask for the
        azimuth and altitude of the triggered events. A mapping is created
        between the triggered events and the simulated events using the shower IDs.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Viewcone radius in degrees.
        """
        hist, _ = np.histogram(self.triggered_data.angular_distance, bins=self.view_cone_bins)
        return (
            self._compute_limits(hist, self.view_cone_bins, loss_fraction, limit_type="upper")
            * u.deg
        )

    def plot_data(self, lower_energy_limit, upper_radial_distance, viewcone, output_path=None):
        """
        Plot the core distances and energies of triggered events.

        Parameters
        ----------
        lower_energy_limit: astropy.units.Quantity
            Lower energy limit to display on plots.
        upper_radial_distance: astropy.units.Quantity
            Upper radial distance limit to display on plots.
        viewcone: astropy.units.Quantity
            Viewcone radius to display on plots.
        output_path: Path or str, optional
            Directory to save plots. If None, plots will be displayed.
        """
        plots = {
            "core_vs_energy": {
                "x_data": self.event_data.core_distance_shower,
                "y_data": self.event_data.simulated_energy,
                "bins": [self.core_distance_bins, self.energy_bins],
                "plot_type": "histogram2d",
                "plot_params": {"norm": "log", "cmap": "viridis"},
                "x_label": "Core Distance [m]",
                "y_label": "Energy [TeV]",
                "title": "Triggered events: core distance vs energy",
                "y_scale": "log",
                "colorbar_label": "Event Count",
                "filename": "core_vs_energy_distribution.png",
            },
            "energy_distribution": {
                "x_data": self.event_data.simulated_energy,
                "bins": np.logspace(-3, 0.0, 100),
                "plot_type": "histogram",
                "plot_params": {"histtype": "step", "color": "k", "lw": 2},
                "x_label": "Energy [TeV]",
                "y_label": "Event Count",
                "title": "Triggered events: energy distribution",
                "x_scale": "log",
                "y_scale": "log",
                "x_line": lower_energy_limit.value,
                "filename": "energy_distribution.png",
            },
            "core_distance": {
                "x_data": self.event_data.core_distance_shower,
                "bins": self.core_distance_bins,
                "plot_type": "histogram",
                "plot_params": {"histtype": "step", "color": "k", "lw": 2},
                "x_label": "Core Distance [m]",
                "y_label": "Event Count",
                "title": "Triggered events: core distance distribution",
                "x_line": upper_radial_distance.value,
                "filename": "core_distance_distribution.png",
            },
            "core_xy": {
                "x_data": self.event_data.x_core_shower,
                "y_data": self.event_data.y_core_shower,
                "bins": 100,
                "plot_type": "histogram2d",
                "plot_params": {"norm": "log", "cmap": "viridis"},
                "x_label": "Core X [m]",
                "y_label": "Core Y [m]",
                "title": "Triggered events: core x vs core y",
                "colorbar_label": "Event Count",
                "x_line": upper_radial_distance.value,
                "y_line": upper_radial_distance.value,
                "filename": "core_xy_distribution.png",
            },
            "view-cone": {
                "x_data": self.triggered_data.angular_distance,
                "bins": self.view_cone_bins,
                "plot_type": "histogram",
                "plot_params": {"histtype": "step", "color": "k", "lw": 2},
                "x_label": "Distance to pointing direction [deg]",
                "y_label": "Event Count",
                "title": "Triggered events: viewcone distribution",
                "x_line": viewcone.value,
                "filename": "viewcone_distribution.png",
            },
        }

        for _, plot_args in plots.items():
            filename = plot_args.pop("filename")
            output_file = output_path / filename if output_path else None
            self._create_plot(**plot_args, output_file=output_file)

    def _create_plot(
        self,
        x_data,
        y_data=None,
        bins=None,
        plot_type="histogram",
        plot_params=None,
        x_label="",
        y_label="",
        title="",
        x_scale=None,
        y_scale=None,
        colorbar_label=None,
        output_file=None,
        x_line=None,
        y_line=None,
    ):
        """
        Create and save a plot with the given parameters.

        Parameters
        ----------
        x_data : array-like
            Data for the x-axis or primary data for histograms.
        y_data : array-like, optional
            Data for the y-axis in scatter or 2D histograms.
        bins : int, array-like, or list, optional
            Bins specification for histograms.
        plot_type : str, optional
            Type of plot: 'histogram', 'histogram2d', or 'scatter'.
        plot_params : dict, optional
            Additional parameters to pass to the plotting function.
        x_label : str, optional
            Label for the x-axis.
        y_label : str, optional
            Label for the y-axis.
        title : str, optional
            Title for the plot.
        x_scale : str, optional
            Scale for x-axis ('log' or 'linear').
        y_scale : str, optional
            Scale for y-axis ('log' or 'linear').
        colorbar_label : str, optional
            Label for the colorbar in 2D histograms.
        output_file : Path, optional
            File path to save the plot. If not provided, the plot will be displayed.
        x_line : float, optional
            Value for vertical line on the plot.
        y_line : float, optional
            Value for horizontal line on the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.
        """
        fig = plt.figure(figsize=(8, 6))
        plot_params = plot_params or {}

        if plot_type == "histogram":
            plt.hist(x_data, bins=bins, **plot_params)
        elif plot_type == "histogram2d":
            plt.hist2d(x_data, y_data, bins=bins, **plot_params)
            if colorbar_label:
                plt.colorbar(label=colorbar_label)
        elif plot_type == "scatter":
            plt.scatter(x_data, y_data, **plot_params)

        if x_line:
            plt.axvline(x_line, color="r", linestyle="--")
        if y_line:
            plt.axhline(y_line, color="r", linestyle="--")

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        if x_scale:
            plt.xscale(x_scale)
        if y_scale:
            plt.yscale(y_scale)

        if output_file:
            self._logger.info(f"Saving plot to {output_file}")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

        return fig
