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
        for data_set in self.reader.data_sets:
            self._logger.info(f"Reading event data from {self.event_data_file} for {data_set}")
            _file_info_table, shower_data, event_data, triggered_data = self.reader.read_event_data(
                self.event_data_file, table_name_map=data_set
            )
            _file_info_table = self.reader.get_reduced_simulation_file_info(_file_info_table)
            self.file_info = {
                "energy_min": _file_info_table["energy_min"].to("TeV"),
                "core_scatter_max": _file_info_table["core_scatter_max"].to("m"),
                "viewcone_max": _file_info_table["viewcone_max"].to("deg"),
            }

            xy_bins = np.linspace(
                -1.0 * self.core_distance_bins.max(),
                self.core_distance_bins.max(),
                len(self.core_distance_bins),
            )

            hist_defs = [
                ("energy", event_data.simulated_energy, self.energy_bins, True),
                ("energy_mc", shower_data.simulated_energy, self.energy_bins, True),
                ("core_distance", event_data.core_distance_shower, self.core_distance_bins, True),
                (
                    "core_distance_mc",
                    shower_data.core_distance_shower,
                    self.core_distance_bins,
                    True,
                ),
                ("angular_distance", triggered_data.angular_distance, self.view_cone_bins, True),
                ("angular_distance_mc", shower_data.angular_distance, self.view_cone_bins, True),
                (
                    "x_core_shower_vs_y_core_shower",
                    (event_data.x_core_shower, event_data.y_core_shower),
                    [xy_bins, xy_bins],
                    False,
                ),
                (
                    "x_core_shower_vs_y_core_shower_mc",
                    (shower_data.x_core_shower, shower_data.y_core_shower),
                    [xy_bins, xy_bins],
                    False,
                ),
                (
                    "core_vs_energy",
                    (event_data.core_distance_shower, event_data.simulated_energy),
                    [self.core_distance_bins, self.energy_bins],
                    False,
                ),
                (
                    "core_vs_energy_mc",
                    (shower_data.core_distance_shower, shower_data.simulated_energy),
                    [self.core_distance_bins, self.energy_bins],
                    False,
                ),
                (
                    "angular_distance_vs_energy",
                    (triggered_data.angular_distance, event_data.simulated_energy),
                    [self.view_cone_bins, self.energy_bins],
                    False,
                ),
                (
                    "angular_distance_vs_energy_mc",
                    (shower_data.angular_distance, shower_data.simulated_energy),
                    [self.view_cone_bins, self.energy_bins],
                    False,
                ),
            ]

            for name, data, bins, hist1d in hist_defs:
                self._fill_histogram_and_bin_edges(name, data, bins, hist1d=hist1d)

            break

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

    def plot(self, output_path=None, limits=None, rebin_factor=2):
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
        self._logger.info(f"Plotting histograms written to {output_path}")

        cumulative_data = self._calculate_cumulative_data()

        upper_radius_limit, lower_energy_limit, viewcone_radius = self._get_limits(limits)

        plots = self._generate_plot_configurations(
            cumulative_data, upper_radius_limit, lower_energy_limit, viewcone_radius
        )

        self._execute_plotting_loop(plots, output_path, rebin_factor)

    def _calculate_cumulative_data(self):
        """Calculate all cumulative histograms needed for plotting."""
        cumulative_data = {}

        # Calculate normalized cumulative for 2D vs_energy histograms
        for hist_key in self.histograms:
            if hist_key.endswith("_vs_energy") and not hist_key.endswith("_mc"):
                output_key = f"normalized_cumulative_{hist_key}"
                hist = self.histograms.get(hist_key)
                cumulative_data[output_key] = self._calculate_cumulative_histogram(
                    hist, axis=0, normalize=True
                )

        # Calculate cumulative for 1D histograms (triggered events only)
        for hist_key in ["energy", "core_distance", "angular_distance"]:
            if hist_key in self.histograms:
                output_key = f"cumulative_{hist_key}"
                hist = self.histograms.get(hist_key)
                reverse = hist_key == "energy"  # Only energy uses reverse cumulative
                cumulative_data[output_key] = self._calculate_cumulative_histogram(
                    hist, reverse=reverse
                )

        return cumulative_data

    def _generate_plot_configurations(
        self, cumulative_data, upper_radius_limit, lower_energy_limit, viewcone_radius
    ):
        """Generate plot configurations for all histogram types."""
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

        plots = {}

        plots.update(
            self._generate_1d_plots(
                hist_1d_params,
                hist_1d_cumulative_params,
                cumulative_data,
                energy_label,
                core_distance_label,
                pointing_direction_label,
                event_count_label,
                cumulative_prefix,
                upper_radius_limit,
                lower_energy_limit,
                viewcone_radius,
            )
        )

        plots.update(
            self._generate_2d_plots(
                hist_2d_params,
                hist_2d_equal_params,
                hist_2d_normalized_params,
                cumulative_data,
                energy_label,
                core_distance_label,
                pointing_direction_label,
                event_count_label,
                core_x_label,
                core_y_label,
                upper_radius_limit,
                lower_energy_limit,
                viewcone_radius,
            )
        )

        return plots

    def _generate_1d_plots(
        self,
        hist_1d_params,
        hist_1d_cumulative_params,
        cumulative_data,
        energy_label,
        core_distance_label,
        pointing_direction_label,
        event_count_label,
        cumulative_prefix,
        upper_radius_limit,
        lower_energy_limit,
        viewcone_radius,
    ):
        """Generate 1D histogram plot configurations."""
        plots = {}

        plot_configs = [
            {
                "base_key": "energy_distribution",
                "histogram_key": "energy",
                "bin_key": "energy_bin_edges",
                "cumulative_key": "cumulative_energy",
                "x_label": energy_label,
                "title_base": "energy distribution",
                "scales": {"x": "log", "y": "log"},
                "line_key": "x",
                "line_value": lower_energy_limit,
            },
            {
                "base_key": "core_distance",
                "histogram_key": "core_distance",
                "bin_key": "core_distance_bin_edges",
                "cumulative_key": "cumulative_core_distance",
                "x_label": core_distance_label,
                "title_base": "core distance distribution",
                "scales": {},
                "line_key": "x",
                "line_value": upper_radius_limit,
            },
            {
                "base_key": "angular_distance",
                "histogram_key": "angular_distance",
                "bin_key": "angular_distance_bin_edges",
                "cumulative_key": "cumulative_angular_distance",
                "x_label": pointing_direction_label,
                "title_base": "angular distance distribution",
                "scales": {},
                "line_key": "x",
                "line_value": viewcone_radius,
            },
        ]

        for config in plot_configs:
            plots[config["base_key"]] = self._create_1d_plot_config(
                config, hist_1d_params, event_count_label, False
            )

            mc_key = f"{config['base_key']}_mc"
            plots[mc_key] = self._create_1d_plot_config(
                config, hist_1d_params, event_count_label, True
            )

            # Cumulative version
            cumulative_key = f"{config['base_key']}_cumulative"
            plots[cumulative_key] = self._create_1d_cumulative_plot_config(
                config,
                hist_1d_cumulative_params,
                cumulative_data,
                event_count_label,
                cumulative_prefix,
            )

        return plots

    def _create_1d_plot_config(self, config, plot_params, event_count_label, is_mc=False):
        """Create a 1D plot configuration."""
        suffix = "_mc" if is_mc else ""
        histogram_key = f"{config['histogram_key']}{suffix}"
        event_type = "Simulated events" if is_mc else "Triggered events"

        lines = {}
        if config["line_value"] is not None:
            lines[config["line_key"]] = config["line_value"]

        return {
            "data": self.histograms.get(histogram_key),
            "bins": self.histograms.get(config["bin_key"]),
            "plot_type": "histogram",
            "plot_params": plot_params,
            "labels": {
                "x": config["x_label"],
                "y": event_count_label,
                "title": f"{event_type}: {config['title_base']}",
            },
            "scales": config["scales"],
            "lines": lines,
            "filename": f"{config['base_key']}{suffix}",
        }

    def _create_1d_cumulative_plot_config(
        self, config, plot_params, cumulative_data, event_count_label, cumulative_prefix
    ):
        """Create a 1D cumulative plot configuration."""
        lines = {}
        if config["line_value"] is not None:
            lines[config["line_key"]] = config["line_value"]

        return {
            "data": cumulative_data[config["cumulative_key"]],
            "bins": self.histograms.get(config["bin_key"]),
            "plot_type": "histogram",
            "plot_params": plot_params,
            "labels": {
                "x": config["x_label"],
                "y": cumulative_prefix + event_count_label,
                "title": f"Triggered events: cumulative {config['title_base']}",
            },
            "scales": config["scales"],
            "lines": lines,
            "filename": f"{config['base_key']}_cumulative",
        }

    def _generate_2d_plots(
        self,
        hist_2d_params,
        hist_2d_equal_params,
        hist_2d_normalized_params,
        cumulative_data,
        energy_label,
        core_distance_label,
        pointing_direction_label,
        event_count_label,
        core_x_label,
        core_y_label,
        upper_radius_limit,
        lower_energy_limit,
        viewcone_radius,
    ):
        """Generate 2D histogram plot configurations."""
        plots = {}

        # Define base configurations for different 2D plot types
        plot_configs = [
            {
                "base_key": "core_vs_energy",
                "event_type": "Triggered events",
                "x_label": core_distance_label,
                "y_label": energy_label,
                "plot_params": hist_2d_params,
                "lines": {"x": upper_radius_limit, "y": lower_energy_limit},
                "scales": {"y": "log"},
                "colorbar_label": event_count_label,
            },
            {
                "base_key": "angular_distance_vs_energy",
                "event_type": "Triggered events",
                "x_label": pointing_direction_label,
                "y_label": energy_label,
                "plot_params": hist_2d_params,
                "lines": {"x": viewcone_radius, "y": lower_energy_limit},
                "scales": {"y": "log"},
                "colorbar_label": event_count_label,
            },
            {
                "base_key": "x_core_shower_vs_y_core_shower",
                "event_type": "Triggered events",
                "x_label": core_x_label,
                "y_label": core_y_label,
                "plot_params": hist_2d_equal_params,
                "lines": {"r": upper_radius_limit},
                "scales": {},
                "colorbar_label": event_count_label,
            },
        ]

        for config in plot_configs:
            plots[config["base_key"]] = self._create_2d_plot_config(config, False)
            mc_config = config.copy()
            mc_config["event_type"] = "Simulated events"
            plots[f"{config['base_key']}_mc"] = self._create_2d_plot_config(mc_config, True)

            # Cumulative version (only for plots that make sense to have cumulative)
            if config["base_key"] != "x_core_shower_vs_y_core_shower":
                cumulative_config = config.copy()
                cumulative_config["plot_params"] = hist_2d_normalized_params
                cumulative_config["colorbar_label"] = "Fraction of events"
                cumulative_config["is_cumulative"] = True
                plots[f"{config['base_key']}_cumulative"] = self._create_2d_cumulative_plot_config(
                    cumulative_config, cumulative_data
                )

        return plots

    def _create_2d_plot_config(self, config, is_mc=False):
        """Create a 2D plot configuration."""
        suffix = "_mc" if is_mc else ""
        base_key = config["base_key"]
        data_key = f"{base_key}{suffix}"

        # Handle special case for title formatting
        if "angular_distance" in base_key:
            distance_type = "angular distance distance"
        elif base_key == "core_xy":
            distance_type = "core x vs core y"
        else:
            distance_type = base_key.replace("_", " ")

        return {
            "data": self.histograms.get(data_key),
            "bins": [
                self.histograms.get(f"{data_key}_bin_x_edges"),
                self.histograms.get(f"{data_key}_bin_y_edges"),
            ],
            "plot_type": "histogram2d",
            "plot_params": config["plot_params"],
            "labels": {
                "x": config["x_label"],
                "y": config["y_label"],
                "title": f"{config['event_type']}: {distance_type}",
            },
            "lines": config["lines"],
            "scales": config.get("scales", {}),
            "colorbar_label": config["colorbar_label"],
            "filename": f"{base_key}{suffix}_distribution",
        }

    def _create_2d_cumulative_plot_config(self, config, cumulative_data):
        """Create a 2D cumulative plot configuration."""
        base_key = config["base_key"]

        # Handle special case for title formatting
        if "angular_distance" in base_key:
            distance_type = "angular distance"
        else:
            distance_type = base_key.replace("_vs_energy", "").replace("_", " ")

        # Generate cumulative data key dynamically to match _calculate_cumulative_data
        cumulative_data_key = f"normalized_cumulative_{base_key}"

        return {
            "data": cumulative_data[cumulative_data_key],
            "bins": [
                self.histograms.get(f"{base_key}_bin_x_edges"),
                self.histograms.get(f"{base_key}_bin_y_edges"),
            ],
            "plot_type": "histogram2d",
            "plot_params": config["plot_params"],
            "labels": {
                "x": config["x_label"],
                "y": config["y_label"],
                "title": f"{config['event_type']}: fraction of events by {distance_type} vs energy",
            },
            "lines": config["lines"],
            "scales": config.get("scales", {}),
            "colorbar_label": config["colorbar_label"],
            "filename": f"{base_key}_cumulative_distribution",
        }

    def _execute_plotting_loop(self, plots, output_path, rebin_factor):
        """Execute the main plotting loop for all plot configurations."""
        for plot_key, plot_args in plots.items():
            plot_filename = plot_args.pop("filename")

            # Skip plots with no data
            if plot_args.get("data") is None:
                self._logger.warning(f"Skipping plot {plot_key} - no data available")
                continue

            if self.array_name and plot_args.get("labels", {}).get("title"):
                plot_args["labels"]["title"] += f" ({self.array_name} array)"

            filename = self._build_plot_filename(plot_filename, self.array_name)
            output_file = output_path / filename if output_path else None
            result = self._create_plot(**plot_args, output_file=output_file)

            # Skip rebinned plot if main plot failed
            if result is None:
                continue

            if self._should_create_rebinned_plot(rebin_factor, plot_args, plot_key):
                self._create_rebinned_plot(plot_args, filename, output_path, rebin_factor)

    def _get_limits(self, limits):
        """Extract limits from the provided dictionary for plotting."""
        if not limits:
            return None, None, None
        return tuple(
            limits.get(key).value if key in limits else None
            for key in ("upper_radius_limit", "lower_energy_limit", "viewcone_radius")
        )

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
        return f"{base_filename}_{array_name}.png" if array_name else f"{base_filename}.png"

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

        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            self._logger.warning("No data available for plotting")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == "histogram":
            plt.bar(bins[:-1], data, width=np.diff(bins), **plot_params)
        elif plot_type == "histogram2d":
            pcm = self._create_2d_histogram_plot(data, bins, plot_params)
            plt.colorbar(pcm, label=colorbar_label)

        for xy in ["x", "y"]:
            if xy in lines and lines[xy] is not None:
                ax.axvline(lines[xy], color="r", linestyle="--", linewidth=0.5)
        if "r" in lines and lines["r"] is not None:
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
            # Handle empty or invalid data for logarithmic scaling
            data_max = data.max()
            if data_max <= 0:
                self._logger.warning(
                    "No positive data found for logarithmic scaling, using linear scale"
                )
                pcm = plt.pcolormesh(
                    bins[0], bins[1], data.T, vmin=0, vmax=max(1, data_max), cmap="viridis"
                )
            else:
                # Ensure vmin is less than vmax for LogNorm
                vmin = max(1, data[data > 0].min()) if np.any(data > 0) else 1
                vmax = max(vmin + 1, data_max)
                pcm = plt.pcolormesh(
                    bins[0], bins[1], data.T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis"
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
