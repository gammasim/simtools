"""Calculate CORSIKA thresholds for energy, radial distance, and viewcone."""

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader


class LimitCalculator:
    """
    Compute limits for CORSIKA configuration for energy, radial distance, and viewcone.

    Event data is read from the reduced MC event data file.

    Parameters
    ----------
    event_data_file : str
        Path to the event-data file.
    array_name : str, optional
        Name of the telescope array configuration (default is None).
    telescope_list : list, optional
        List of telescope IDs to filter the events (default is None).
    """

    def __init__(self, event_data_file, array_name=None, telescope_list=None):
        """Initialize the LimitCalculator with the given event data file."""
        self._logger = logging.getLogger(__name__)
        self.event_data_file = event_data_file
        self.array_name = array_name
        self.telescope_list = telescope_list

        self.limits = None
        self.histograms = {}
        self.file_info = {}

        self.reader = SimtelIOEventDataReader(event_data_file, telescope_list=telescope_list)

    def _prepare_limit_data(self, file_info_table):
        """
        Prepare result data structure for limit calculation.

        Contains both the point in parameter space and the limits derived for that point.

        Parameters
        ----------
        file_info_table : astropy.table.Table
            Table containing file information.

        Returns
        -------
        dict
            Dictionary containing limits (not yet calculated) and parameter space information.
        """
        self.file_info = self.reader.get_reduced_simulation_file_info(file_info_table)
        return {
            "primary_particle": self.file_info["primary_particle"],
            "zenith": self.file_info["zenith"],
            "azimuth": self.file_info["azimuth"],
            "nsb_level": self.file_info["nsb_level"],
            "array_name": self.array_name,
            "telescope_ids": self.telescope_list,
            "lower_energy_limit": None,
            "upper_radius_limit": None,
            "viewcone_radius": None,
        }

    def compute_limits(self, loss_fraction):
        """
        Compute the limits for energy, radial distance, and viewcone.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        dict
            Dictionary containing the computed limits.
        """
        self._fill_histograms()

        self.limits["lower_energy_limit"] = self.compute_lower_energy_limit(loss_fraction)
        self.limits["upper_radius_limit"] = self.compute_upper_radius_limit(loss_fraction)
        self.limits["viewcone_radius"] = self.compute_viewcone(loss_fraction)

        return self.limits

    def _fill_histogram_and_bin_edges(self, name, data, bins, hist1d=True):
        """
        Fill histogram and bin edges and it both to histogram dictionary.

        Adds histogram to existing histogram if it exists, otherwise initializes it.

        """
        if name in self.histograms:
            if hist1d:
                bins = self.histograms[f"{name}_bin_edges"]
                hist, _ = np.histogram(data, bins=bins)
                self.histograms[name] += hist
            else:
                x_bins = self.histograms[f"{name}_bin_x_edges"]
                y_bins = self.histograms[f"{name}_bin_y_edges"]
                hist, _, _ = np.histogram2d(data[0], data[1], bins=[x_bins, y_bins])
                self.histograms[name] += hist
        else:
            if hist1d:
                hist, bin_edges = np.histogram(data, bins=bins)
                self.histograms[name] = hist
                self.histograms[f"{name}_bin_edges"] = bin_edges
            else:
                hist, x_edges, y_edges = np.histogram2d(data[0], data[1], bins=bins)
                self.histograms[name] = hist
                self.histograms[f"{name}_bin_x_edges"] = x_edges
                self.histograms[f"{name}_bin_y_edges"] = y_edges

    def _fill_histograms(self):
        """
        Fill histograms with event data.

        Involves looping over all event data, and therefore is the slowest part of the
        limit calculation. Adds the histograms to the histogram dictionary.
        """
        for data_set in self.reader.data_sets:
            self._logger.info(f"Reading event data from {self.event_data_file} for {data_set}")
            file_info, _, event_data, triggered_data = self.reader.read_event_data(
                self.event_data_file, table_name_map=data_set
            )
            self.limits = self.limits if self.limits else self._prepare_limit_data(file_info)

            self._fill_histogram_and_bin_edges(
                "energy", event_data.simulated_energy, self.energy_bins
            )
            self._fill_histogram_and_bin_edges(
                "core_distance", event_data.core_distance_shower, self.core_distance_bins
            )
            self._fill_histogram_and_bin_edges(
                "angular_distance", triggered_data.angular_distance, self.view_cone_bins
            )

            xy_bins = np.linspace(
                -1.0 * self.core_distance_bins.max(),
                self.core_distance_bins.max(),
                len(self.core_distance_bins),
            )
            self._fill_histogram_and_bin_edges(
                "shower_cores",
                (event_data.x_core_shower, event_data.y_core_shower),
                [xy_bins, xy_bins],
                hist1d=False,
            )
            self._fill_histogram_and_bin_edges(
                "core_vs_energy",
                (event_data.core_distance_shower, event_data.simulated_energy),
                [self.core_distance_bins, self.energy_bins],
                hist1d=False,
            )

    def _compute_limits(self, hist, bin_edges, loss_fraction, limit_type="lower"):
        """
        Compute the limits based on the loss fraction.

        Add or subtract one bin to be on the safe side of the limit.

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
        total_events = np.sum(hist)
        threshold = (1 - loss_fraction) * total_events
        if limit_type == "upper":
            cum = np.cumsum(hist)
            idx = np.searchsorted(cum, threshold) + 1
            return bin_edges[min(idx, len(bin_edges) - 1)]
        if limit_type == "lower":
            cum = np.cumsum(hist[::-1])
            idx = np.searchsorted(cum, threshold) + 1
            return bin_edges[max(len(bin_edges) - 1 - idx, 0)]
        raise ValueError("limit_type must be 'lower' or 'upper'")

    @property
    def energy_bins(self):
        """Return bins for the energy histogram."""
        return np.logspace(
            np.log10(self.file_info.get("energy_min", 1.0e-3 * u.TeV).to("TeV").value),
            np.log10(self.file_info.get("energy_max", 1.0e3 * u.TeV).to("TeV").value),
            100,
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
        return (
            self._compute_limits(
                self.histograms.get("energy"), self.energy_bins, loss_fraction, limit_type="lower"
            )
            * u.TeV
        )

    @property
    def core_distance_bins(self):
        """Return bins for the core distance histogram."""
        return np.linspace(
            self.file_info.get("core_scatter_min", 0.0 * u.m).to("m").value,
            self.file_info.get("core_scatter_max", 1.0e5 * u.m).to("m").value,
            100,
        )

    def compute_upper_radius_limit(self, loss_fraction):
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
        return (
            self._compute_limits(
                self.histograms.get("core_distance"),
                self.core_distance_bins,
                loss_fraction,
                limit_type="upper",
            )
            * u.m
        )

    @property
    def view_cone_bins(self):
        """Return bins for the viewcone histogram."""
        return np.linspace(
            self.file_info.get("viewcone_min", 0.0 * u.deg).to("deg").value,
            self.file_info.get("viewcone_max", 20.0 * u.deg).to("deg").value,
            100,
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
        return (
            self._compute_limits(
                self.histograms.get("angular_distance"),
                self.view_cone_bins,
                loss_fraction,
                limit_type="upper",
            )
            * u.deg
        )

    def plot_data(self, output_path=None):
        """
        Histogram plotting.

        Parameters
        ----------
        output_path: Path or str, optional
            Directory to save plots. If None, plots will be displayed.
        """
        self._logger.info(f"Plotting histograms written to {output_path}")
        event_counts = "Event Count"
        plots = {
            "core_vs_energy": {
                "data": self.histograms.get("core_vs_energy"),
                "bins": [
                    self.histograms.get("core_vs_energy_bin_x_edges"),
                    self.histograms.get("core_vs_energy_bin_y_edges"),
                ],
                "plot_type": "histogram2d",
                "plot_params": {"norm": "log", "cmap": "viridis"},
                "labels": {
                    "x": "Core Distance [m]",
                    "y": "Energy [TeV]",
                    "title": "Triggered events: core distance vs energy",
                },
                "lines": {
                    "x": self.limits["upper_radius_limit"].value,
                    "y": self.limits["lower_energy_limit"].value,
                },
                "scales": {"y": "log"},
                "colorbar_label": event_counts,
                "filename": "core_vs_energy_distribution",
            },
            "energy_distribution": {
                "data": self.histograms.get("energy"),
                "bins": self.histograms.get("energy_bin_edges"),
                "plot_type": "histogram",
                "plot_params": {"color": "g", "edgecolor": "g", "lw": 1},
                "labels": {
                    "x": "Energy [TeV]",
                    "y": event_counts,
                    "title": "Triggered events: energy distribution",
                },
                "scales": {"x": "log", "y": "log"},
                "lines": {"x": self.limits["lower_energy_limit"].value},
                "filename": "energy_distribution",
            },
            "core_distance": {
                "data": self.histograms.get("core_distance"),
                "bins": self.histograms.get("core_distance_bin_edges"),
                "plot_type": "histogram",
                "plot_params": {"color": "g", "edgecolor": "g", "lw": 1},
                "labels": {
                    "x": "Core Distance [m]",
                    "y": event_counts,
                    "title": "Triggered events: core distance distribution",
                },
                "lines": {"x": self.limits["upper_radius_limit"].value},
                "filename": "core_distance_distribution",
            },
            "core_xy": {
                "data": self.histograms.get("shower_cores"),
                "bins": [
                    self.histograms.get("shower_cores_bin_x_edges"),
                    self.histograms.get("shower_cores_bin_y_edges"),
                ],
                "plot_type": "histogram2d",
                "plot_params": {"norm": "log", "cmap": "viridis", "aspect": "equal"},
                "labels": {
                    "x": "Core X [m]",
                    "y": "Core Y [m]",
                    "title": "Triggered events: core x vs core y",
                },
                "colorbar_label": event_counts,
                "lines": {
                    "r": self.limits["upper_radius_limit"].value,
                },
                "filename": "core_xy_distribution",
            },
            "angular_distance": {
                "data": self.histograms.get("angular_distance"),
                "bins": self.histograms.get("angular_distance_bin_edges"),
                "plot_type": "histogram",
                "plot_params": {"color": "g", "edgecolor": "g", "lw": 1},
                "labels": {
                    "x": "Distance to pointing direction [deg]",
                    "y": event_counts,
                    "title": "Triggered events: angular distance distribution",
                },
                "lines": {"x": self.limits["viewcone_radius"].value},
                "filename": "angular_distance_distribution",
            },
        }

        for _, plot_args in plots.items():
            filename = plot_args.pop("filename")
            if self.array_name:
                if plot_args.get("labels", {}).get("title"):
                    plot_args["labels"]["title"] += f" ({self.array_name} array)"
                filename = f"{filename}_{self.array_name}.png"
            else:
                filename = f"{filename}.png"
            output_file = output_path / filename if output_path else None
            self._create_plot(**plot_args, output_file=output_file)

    def _create_plot(
        self,
        data,
        bins=None,
        plot_type="histogram",
        plot_params=None,
        labels=None,
        scales=None,
        colorbar_label=None,
        output_file=None,
        lines=None,
    ):
        """Create and save a plot with the given parameters."""
        plot_params = plot_params or {}
        labels = labels or {}
        scales = scales or {}
        lines = lines or {}

        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == "histogram":
            plt.bar(bins[:-1], data, width=np.diff(bins), **plot_params)
        elif plot_type == "histogram2d":
            pcm = plt.pcolormesh(
                bins[0], bins[1], data.T, norm=LogNorm(vmin=1, vmax=data.max()), cmap="viridis"
            )
            plt.colorbar(pcm, label=colorbar_label)

        if "x" in lines:
            plt.axvline(lines["x"], color="r", linestyle="--", linewidth=0.5)
        if "y" in lines:
            plt.axhline(lines["y"], color="r", linestyle="--", linewidth=0.5)
        if "r" in lines:
            circle = plt.Circle(
                (0, 0), lines["r"], color="r", fill=False, linestyle="--", linewidth=0.5
            )
            plt.gca().add_artist(circle)

        ax.set(
            xlabel=labels.get("x", ""),
            ylabel=labels.get("y", ""),
            title=labels.get("title", ""),
            xscale=scales.get("x", "linear"),
            yscale=scales.get("y", "linear"),
        )

        if output_file:
            self._logger.info(f"Saving plot to {output_file}")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

        return fig
