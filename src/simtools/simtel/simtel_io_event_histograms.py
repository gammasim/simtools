"""Histograms for shower and triggered events."""

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader


class SimtelIOEventHistograms:
    """
    Generate and fill histograms for shower and triggered events.

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
        """Initialize."""
        self._logger = logging.getLogger(__name__)
        self.event_data_file = event_data_file
        self.array_name = array_name
        self.telescope_list = telescope_list

        self.histograms = {}
        self.file_info = {}

        self.reader = SimtelIOEventDataReader(event_data_file, telescope_list=telescope_list)

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

    def fill(self):
        """
        Fill histograms with event data.

        Involves looping over all event data, and therefore is the slowest part of the
        limit calculation. Adds the histograms to the histogram dictionary.

        Assume that all event data files are generated with similar configurations
        (self.file_info contains the latest file info).
        """
        _file_info_table = None
        for data_set in self.reader.data_sets:
            self._logger.info(f"Reading event data from {self.event_data_file} for {data_set}")
            _file_info_table, _, event_data, triggered_data = self.reader.read_event_data(
                self.event_data_file, table_name_map=data_set
            )

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
            self._fill_histogram_and_bin_edges(
                "angular_distance_vs_energy",
                (triggered_data.angular_distance, event_data.simulated_energy),
                [self.view_cone_bins, self.energy_bins],
                hist1d=False,
            )

        _file_info_table = self.reader.get_reduced_simulation_file_info(_file_info_table)
        self.file_info = {
            "energy_min": _file_info_table["energy_min"].to("TeV"),
            "core_scatter_max": _file_info_table["core_scatter_max"].to("m"),
            "viewcone_max": _file_info_table["viewcone_max"].to("deg"),
        }

    @property
    def energy_bins(self):
        """Return bins for the energy histogram."""
        if "energy_bin_edges" in self.histograms:
            return self.histograms["energy_bin_edges"]
        return np.logspace(
            np.log10(self.file_info.get("energy_min", 1.0e-3 * u.TeV).to("TeV").value),
            np.log10(self.file_info.get("energy_max", 1.0e3 * u.TeV).to("TeV").value),
            100,
        )

    @property
    def core_distance_bins(self):
        """Return bins for the core distance histogram."""
        if "core_distance_bin_edges" in self.histograms:
            return self.histograms["core_distance_bin_edges"]
        return np.linspace(
            self.file_info.get("core_scatter_min", 0.0 * u.m).to("m").value,
            self.file_info.get("core_scatter_max", 1.0e5 * u.m).to("m").value,
            100,
        )

    @property
    def view_cone_bins(self):
        """Return bins for the viewcone histogram."""
        if "viewcone_bin_edges" in self.histograms:
            return self.histograms["viewcone_bin_edges"]
        return np.linspace(
            self.file_info.get("viewcone_min", 0.0 * u.deg).to("deg").value,
            self.file_info.get("viewcone_max", 20.0 * u.deg).to("deg").value,
            100,
        )

    def plot_data(self, output_path=None, limits=None, rebin_factor=2):
        """
        Histogram plotting.

        Parameters
        ----------
        output_path: Path or str, optional
            Directory to save plots. If None, plots will be displayed.
        limits: dict, optional
            Dictionary containing limits for plotting. Keys can include:
            - "upper_radius_limit": Upper limit for core distance
            - "lower_energy_limit": Lower limit for energy
            - "viewcone_radius": Radius for the viewcone
        rebin_factor: int, optional
            Factor by which to reduce the number of bins in 2D histograms for rebinned plots.
            Default is 2 (merge every 2 bins). Set to 0 or 1 to disable rebinning.
        """
        # Plot label constants
        core_distance_label = "Core Distance [m]"
        energy_label = "Energy [TeV]"
        pointing_direction_label = "Distance to pointing direction [deg]"
        cumulative_prefix = "Cumulative "
        event_count_label = "Event Count"
        core_x_label = "Core X [m]"
        core_y_label = "Core Y [m]"

        # Plot parameter constants
        hist_1d_params = {"color": "tab:green", "edgecolor": "tab:green", "lw": 1}
        hist_1d_cumulative_params = {"color": "tab:blue", "edgecolor": "tab:blue", "lw": 1}
        hist_2d_params = {"norm": "log", "cmap": "viridis", "show_contour": False}
        hist_2d_equal_params = {
            "norm": "log",
            "cmap": "viridis",
            "aspect": "equal",
            "show_contour": False,
        }
        hist_2d_normalized_params = {"norm": "linear", "cmap": "viridis", "show_contour": True}

        self._logger.info(f"Plotting histograms written to {output_path}")

        angular_dist_vs_energy = self.histograms.get("angular_distance_vs_energy")
        normalized_cumulative_angular_vs_energy = self._calculate_cumulative_histogram(
            angular_dist_vs_energy, axis=0, normalize=True
        )

        core_vs_energy = self.histograms.get("core_vs_energy")
        normalized_cumulative_core_vs_energy = self._calculate_cumulative_histogram(
            core_vs_energy, axis=0, normalize=True
        )

        energy_hist = self.histograms.get("energy")
        cumulative_energy = self._calculate_cumulative_histogram(energy_hist, reverse=True)

        core_distance_hist = self.histograms.get("core_distance")
        cumulative_core_distance = self._calculate_cumulative_histogram(core_distance_hist)

        angular_distance_hist = self.histograms.get("angular_distance")
        cumulative_angular_distance = self._calculate_cumulative_histogram(angular_distance_hist)

        upper_radius_limit, lower_energy_limit, viewcone_radius = self._get_limits(limits)

        plots = {
            "core_vs_energy": {
                "data": self.histograms.get("core_vs_energy"),
                "bins": [
                    self.histograms.get("core_vs_energy_bin_x_edges"),
                    self.histograms.get("core_vs_energy_bin_y_edges"),
                ],
                "plot_type": "histogram2d",
                "plot_params": hist_2d_params,
                "labels": {
                    "x": core_distance_label,
                    "y": energy_label,
                    "title": "Triggered events: core distance vs energy",
                },
                "lines": {"x": upper_radius_limit, "y": lower_energy_limit},
                "scales": {"y": "log"},
                "colorbar_label": event_count_label,
                "filename": "core_vs_energy_distribution",
            },
            "energy_distribution": {
                "data": self.histograms.get("energy"),
                "bins": self.histograms.get("energy_bin_edges"),
                "plot_type": "histogram",
                "plot_params": hist_1d_params,
                "labels": {
                    "x": energy_label,
                    "y": event_count_label,
                    "title": "Triggered events: energy distribution",
                },
                "scales": {"x": "log", "y": "log"},
                "lines": {"x": lower_energy_limit},
                "filename": "energy_distribution",
            },
            "energy_distribution_cumulative": {
                "data": cumulative_energy,
                "bins": self.histograms.get("energy_bin_edges"),
                "plot_type": "histogram",
                "plot_params": hist_1d_cumulative_params,
                "labels": {
                    "x": energy_label,
                    "y": cumulative_prefix + event_count_label,
                    "title": "Triggered events: cumulative energy distribution",
                },
                "scales": {"x": "log", "y": "log"},
                "lines": {"x": lower_energy_limit},
                "filename": "energy_distribution_cumulative",
            },
            "core_distance": {
                "data": self.histograms.get("core_distance"),
                "bins": self.histograms.get("core_distance_bin_edges"),
                "plot_type": "histogram",
                "plot_params": hist_1d_params,
                "labels": {
                    "x": core_distance_label,
                    "y": event_count_label,
                    "title": "Triggered events: core distance distribution",
                },
                "lines": {"x": upper_radius_limit},
                "filename": "core_distance_distribution",
            },
            "core_distance_cumulative": {
                "data": cumulative_core_distance,
                "bins": self.histograms.get("core_distance_bin_edges"),
                "plot_type": "histogram",
                "plot_params": hist_1d_cumulative_params,
                "labels": {
                    "x": core_distance_label,
                    "y": cumulative_prefix + event_count_label,
                    "title": "Triggered events: cumulative core distance distribution",
                },
                "lines": {"x": upper_radius_limit},
                "filename": "core_distance_cumulative_distribution",
            },
            "core_xy": {
                "data": self.histograms.get("shower_cores"),
                "bins": [
                    self.histograms.get("shower_cores_bin_x_edges"),
                    self.histograms.get("shower_cores_bin_y_edges"),
                ],
                "plot_type": "histogram2d",
                "plot_params": hist_2d_equal_params,
                "labels": {
                    "x": core_x_label,
                    "y": core_y_label,
                    "title": "Triggered events: core x vs core y",
                },
                "colorbar_label": event_count_label,
                "lines": {
                    "r": upper_radius_limit,
                },
                "filename": "core_xy_distribution",
            },
            "angular_distance": {
                "data": self.histograms.get("angular_distance"),
                "bins": self.histograms.get("angular_distance_bin_edges"),
                "plot_type": "histogram",
                "plot_params": hist_1d_params,
                "labels": {
                    "x": pointing_direction_label,
                    "y": event_count_label,
                    "title": "Triggered events: angular distance distribution",
                },
                "lines": {"x": viewcone_radius},
                "filename": "angular_distance_distribution",
            },
            "angular_distance_cumulative": {
                "data": cumulative_angular_distance,
                "bins": self.histograms.get("angular_distance_bin_edges"),
                "plot_type": "histogram",
                "plot_params": hist_1d_cumulative_params,
                "labels": {
                    "x": pointing_direction_label,
                    "y": cumulative_prefix + event_count_label,
                    "title": "Triggered events: cumulative angular distance distribution",
                },
                "lines": {"x": viewcone_radius},
                "filename": "angular_distance_cumulative_distribution",
            },
            "angular_distance_vs_energy": {
                "data": self.histograms.get("angular_distance_vs_energy"),
                "bins": [
                    self.histograms.get("angular_distance_vs_energy_bin_x_edges"),
                    self.histograms.get("angular_distance_vs_energy_bin_y_edges"),
                ],
                "plot_type": "histogram2d",
                "plot_params": hist_2d_params,
                "labels": {
                    "x": pointing_direction_label,
                    "y": energy_label,
                    "title": "Triggered events: angular distance distance vs energy",
                },
                "lines": {
                    "x": viewcone_radius,
                    "y": lower_energy_limit,
                },
                "scales": {"y": "log"},
                "colorbar_label": event_count_label,
                "filename": "angular_distance_vs_energy_distribution",
            },
            "angular_distance_vs_energy_cumulative": {
                "data": normalized_cumulative_angular_vs_energy,
                "bins": [
                    self.histograms.get("angular_distance_vs_energy_bin_x_edges"),
                    self.histograms.get("angular_distance_vs_energy_bin_y_edges"),
                ],
                "plot_type": "histogram2d",
                "plot_params": hist_2d_normalized_params,  # Includes contour line at value=1
                "labels": {
                    "x": pointing_direction_label,
                    "y": energy_label,
                    "title": "Triggered events: fraction of events by angular distance vs energy",
                },
                "lines": {
                    "x": viewcone_radius,
                    "y": lower_energy_limit,
                },
                "scales": {"y": "log"},
                "colorbar_label": "Fraction of events",
                "filename": "angular_distance_vs_energy_cumulative_distribution",
            },
            "core_vs_energy_cumulative": {
                "data": normalized_cumulative_core_vs_energy,
                "bins": [
                    self.histograms.get("core_vs_energy_bin_x_edges"),
                    self.histograms.get("core_vs_energy_bin_y_edges"),
                ],
                "plot_type": "histogram2d",
                "plot_params": hist_2d_normalized_params,
                "labels": {
                    "x": core_distance_label,
                    "y": energy_label,
                    "title": "Triggered events: fraction of events by core distance vs energy",
                },
                "lines": {
                    "x": upper_radius_limit,
                    "y": lower_energy_limit,
                },
                "scales": {"y": "log"},
                "colorbar_label": "Fraction of events",
                "filename": "core_vs_energy_cumulative_distribution",
            },
        }

        for plot_key, plot_args in plots.items():
            plot_filename = plot_args.pop("filename")
            if self.array_name and plot_args.get("labels", {}).get("title"):
                plot_args["labels"]["title"] += f" ({self.array_name} array)"

            filename = self._build_plot_filename(plot_filename, self.array_name)
            output_file = output_path / filename if output_path else None
            self._create_plot(**plot_args, output_file=output_file)

            if self._should_create_rebinned_plot(rebin_factor, plot_args, plot_key):
                self._create_rebinned_plot(plot_args, filename, output_path, rebin_factor)

    def _get_limits(self, limits):
        """Extract limits from the provided dictionary for plotting."""
        upper_radius_limit = None
        lower_energy_limit = None
        viewcone_radius = None
        if limits:
            upper_radius_limit = (
                limits["upper_radius_limit"].value if "upper_radius_limit" in limits else None
            )
            lower_energy_limit = (
                limits["lower_energy_limit"].value if "lower_energy_limit" in limits else None
            )
            viewcone_radius = (
                limits["viewcone_radius"].value if "viewcone_radius" in limits else None
            )
        return upper_radius_limit, lower_energy_limit, viewcone_radius

    def _build_plot_filename(self, base_filename, array_name=None):
        """
        Build the full plot filename with appropriate extensions.

        Parameters
        ----------
        base_filename : str
            The base filename without extension
        array_name : str, optional
            Name of the array to append to filename

        Returns
        -------
        str
            Complete filename with extension
        """
        if array_name:
            return f"{base_filename}_{array_name}.png"
        return f"{base_filename}.png"

    def _should_create_rebinned_plot(self, rebin_factor, plot_args, plot_key):
        """
        Check if a rebinned version of the plot should be created.

        Parameters
        ----------
        rebin_factor : int
            Factor by which to rebin the energy axis
        plot_args : dict
            Plot arguments
        plot_key : str
            Key identifying the plot type

        Returns
        -------
        bool
            True if a rebinned plot should be created, False otherwise
        """
        return (
            rebin_factor > 1
            and plot_args["plot_type"] == "histogram2d"
            and plot_key.endswith("_cumulative")
            and plot_args.get("plot_params", {}).get("norm") == "linear"
        )

    def _create_rebinned_plot(self, plot_args, filename, output_path, rebin_factor):
        """
        Create a rebinned version of a 2D histogram plot.

        Parameters
        ----------
        plot_args : dict
            Plot arguments for the original plot
        filename : str
            Filename of the original plot
        output_path : Path or None
            Path to save the plot to, or None
        rebin_factor : int
            Factor by which to rebin the energy axis
        """
        data = plot_args["data"]
        bins = plot_args["bins"]

        rebinned_data, rebinned_x_bins, rebinned_y_bins = self._rebin_2d_histogram(
            data, bins[0], bins[1], rebin_factor
        )

        rebinned_plot_args = plot_args.copy()
        rebinned_plot_args["data"] = rebinned_data
        rebinned_plot_args["bins"] = [rebinned_x_bins, rebinned_y_bins]

        if rebinned_plot_args.get("labels", {}).get("title"):
            rebinned_plot_args["labels"]["title"] += f" (Energy rebinned {rebin_factor}x)"

        rebinned_filename = f"{filename.replace('.png', '')}_rebinned.png"
        rebinned_output_file = output_path / rebinned_filename if output_path else None
        self._create_plot(**rebinned_plot_args, output_file=rebinned_output_file)

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
        """
        Create and save a plot with the given parameters.

        For normalized 2D histograms, a contour line is drawn at the value of 1.0
        to indicate the boundary where each energy bin reaches complete containment.
        This can be controlled with the 'show_contour' parameter in plot_params.
        """
        plot_params = plot_params or {}
        labels = labels or {}
        scales = scales or {}
        lines = lines or {}

        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == "histogram":
            plt.bar(bins[:-1], data, width=np.diff(bins), **plot_params)
        elif plot_type == "histogram2d":
            pcm = self._create_2d_histogram_plot(data, bins, plot_params)
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

    def _create_2d_histogram_plot(self, data, bins, plot_params):
        """
        Create a 2D histogram plot with the given parameters.

        Parameters
        ----------
        data : np.ndarray
            2D histogram data
        bins : tuple of np.ndarray
            Bin edges for x and y axes
        plot_params : dict
            Plot parameters including norm, cmap, and show_contour

        Returns
        -------
        matplotlib.collections.QuadMesh
            The created pcolormesh object for colorbar attachment
        """
        if plot_params.get("norm") == "linear":
            pcm = plt.pcolormesh(
                bins[0],
                bins[1],
                data.T,
                vmin=0,
                vmax=1,
                cmap=plot_params.get("cmap", "viridis"),
            )
            # Add contour line at value=1.0 for normalized histograms
            if plot_params.get("show_contour", True):
                x_centers = (bins[0][1:] + bins[0][:-1]) / 2
                y_centers = (bins[1][1:] + bins[1][:-1]) / 2
                x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
                plt.contour(
                    x_mesh,
                    y_mesh,
                    data.T,
                    levels=[0.999999],  # very close to 1 for floating point precision
                    colors=["tab:red"],
                    linestyles=["--"],
                    linewidths=[0.5],
                )
        else:
            pcm = plt.pcolormesh(
                bins[0], bins[1], data.T, norm=LogNorm(vmin=1, vmax=data.max()), cmap="viridis"
            )

        return pcm

    def _calculate_cumulative_histogram(self, hist, reverse=False, axis=None, normalize=False):
        """
        Calculate cumulative distribution of a histogram.

        Works with both 1D and 2D histograms.

        Parameters
        ----------
        hist : np.ndarray
            Histogram (1D or 2D)
        reverse : bool, optional
            If True, sum from high to low values
        axis : int, optional
            For 2D histograms, axis along which to compute cumulative sum
            None means default behavior: for 1D just cumsum, for 2D along rows
        normalize : bool, optional
            If True, normalize by the total sum for each slice along the specified axis
            For 1D histograms, normalizes by the total sum

        Returns
        -------
        np.ndarray
            Histogram with cumulative counts, optionally normalized
        """
        if hist is None:
            return None

        if hist.ndim == 1:
            result = self._calculate_cumulative_1d(hist, reverse)
            if normalize and np.sum(hist) > 0:
                result = result / np.sum(hist)
            return result

        if axis is None:
            axis = 1

        result = self._apply_cumsum_along_axis(hist.copy(), axis, reverse)

        if normalize:
            self._normalize_along_axis(result, hist, axis)

        return result

    def _normalize_along_axis(self, result, hist, axis):
        """
        Normalize cumulative histogram along the specified axis.

        Parameters
        ----------
        result : np.ndarray
            Cumulative histogram to normalize (modified in-place)
        hist : np.ndarray
            Original histogram (for calculating totals)
        axis : int
            Axis along which normalization should be applied
        """
        normalized = np.zeros_like(result, dtype=float)

        if axis == 0:
            for i in range(result.shape[1]):
                col_total = np.sum(hist[:, i])
                if col_total > 0:
                    normalized[:, i] = result[:, i] / col_total
        else:  # axis == 1
            for i in range(result.shape[0]):
                row_total = np.sum(hist[i, :])
                if row_total > 0:
                    normalized[i, :] = result[i, :] / row_total

        np.copyto(result, normalized)

    def _calculate_cumulative_1d(self, hist, reverse):
        """Calculate cumulative distribution for 1D histogram."""
        if reverse:
            return np.cumsum(hist[::-1])[::-1]
        return np.cumsum(hist)

    def _calculate_cumulative_2d(self, hist, reverse, axis=None):
        """Calculate cumulative distribution for 2D histogram."""
        if axis is None:
            axis = 1

        return self._apply_cumsum_along_axis(hist, axis, reverse)

    def _apply_cumsum_along_axis(self, hist, axis, reverse):
        """Apply cumulative sum along the specified axis of a 2D histogram."""

        def cumsum_func(arr):
            return np.cumsum(arr[::-1])[::-1] if reverse else np.cumsum(arr)

        return np.apply_along_axis(cumsum_func, axis, hist)

    def _rebin_2d_histogram(self, hist, x_bins, y_bins, rebin_factor=2):
        """
        Rebin a 2D histogram by merging neighboring bins along the energy dimension (y-axis) only.

        Parameters
        ----------
        hist : np.ndarray
            Original 2D histogram data
        x_bins : np.ndarray
            Original x-axis bin edges (preserved)
        y_bins : np.ndarray
            Original y-axis (energy) bin edges
        rebin_factor : int, optional
            Factor by which to reduce the number of bins in the energy dimension
            Default is 2 (merge every 2 bins)

        Returns
        -------
        tuple
            (rebinned_hist, x_bins, rebinned_y_bins)
        """
        if rebin_factor <= 1:
            return hist, x_bins, y_bins

        x_size = hist.shape[0]
        new_y_size = hist.shape[1] // rebin_factor

        new_hist = np.zeros((x_size, new_y_size), dtype=float)

        for i in range(x_size):
            for j in range(new_y_size):
                y_start = j * rebin_factor
                y_end = (j + 1) * rebin_factor
                new_hist[i, j] = np.sum(hist[i, y_start:y_end])

        new_y_bins = y_bins[::rebin_factor]

        return new_hist, x_bins, new_y_bins
