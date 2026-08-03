"""Plot production-grid points on sky coordinate projections."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.production_configuration.job_grid_io import JOB_GRID_SCHEMA

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_FILE_STEM = "production_grid_sky_projection"
PLOT_VALUE_KEYS = tuple(
    key
    for key in JOB_GRID_SCHEMA.column_units
    if key
    not in (
        "azimuth_angle",
        "zenith_angle",
        "ha",
        "dec",
        "view_cone_min",
        "nsb_rate",
        "corsika_hadronic_transition_energy",
    )
)
DEFAULT_OUTPUT_FILE_EXTENSION = "png"
DEFAULT_MARKER_SIZE = 8


def _azimuth_zenith_output_file_stem(value_key):
    """Return output stem for azimuth/zenith color-scale plots."""
    return f"production_grid_altaz_{value_key}"


def _zenith_profile_output_file_stem(value_key):
    """Return output stem for zenith profile plots."""
    return f"production_grid_zenith_profile_{value_key}"


class ProductionGridPlotter:
    """
    Plot production grid points on sky coordinate projections.

    Parameters
    ----------
    grid_points_file : str or Path
        Path to the ECSV file containing grid points.
    output_path : str or Path
        Path to save output plots.
    """

    def __init__(
        self,
        grid_points_file,
        output_path,
    ):
        """Initialize the ProductionGridPlotter."""
        self.grid_points_file = Path(grid_points_file)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.grid_metadata = {}
        self.grid_columns = []
        self.has_hadec_columns = False
        self.grid_points = self._load_grid_points()

        logger.info(f"Loaded {len(self.grid_points)} grid points from {self.grid_points_file}")

    def _load_grid_points(self):
        """
        Load grid points from ECSV file.

        Returns
        -------
        list
            List of grid point dictionaries.

        Raises
        ------
        FileNotFoundError
            If the grid points file does not exist.
        ValueError
            If the grid points file is not ECSV.
        """
        if not self.grid_points_file.exists():
            msg = f"Grid points file not found: {self.grid_points_file}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if self.grid_points_file.suffix.lower() != ".ecsv":
            msg = f"Grid points file must be ECSV: {self.grid_points_file}"
            logger.error(msg)
            raise ValueError(msg)

        grid_table = Table.read(self.grid_points_file, format="ascii.ecsv")
        self.grid_metadata = dict(grid_table.meta)
        self.grid_columns = list(grid_table.colnames)
        self.has_hadec_columns = {"ha", "dec"}.issubset(self.grid_columns)
        return self._convert_ecsv_table_to_grid_points(grid_table)

    @classmethod
    def _convert_ecsv_table_to_grid_points(cls, grid_table):
        """Convert an ECSV grid-point table to plain dictionaries."""
        return [cls._convert_ecsv_row_to_grid_point(row, grid_table) for row in grid_table]

    @staticmethod
    def _plain_table_value(value):
        """Return a plain scalar for non-masked table values."""
        if np.ma.is_masked(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _convert_ecsv_row_to_grid_point(cls, row, grid_table):
        """Convert one ECSV row and merge split value/unit quantity columns."""
        point = {}
        for column_name in grid_table.colnames:
            value = cls._plain_table_value(row[column_name])
            if value is not None:
                unit = grid_table[column_name].unit
                point[column_name] = value * unit if unit is not None else value

        return point

    @staticmethod
    def _extract_quantity_value(point, key):
        """
        Extract numeric value from a grid-point entry.

        Parameters
        ----------
        point : dict
            Grid-point dictionary.
        key : str
            Key to extract.

        Returns
        -------
        float or None
            Extracted numeric value.
        """
        value = point.get(key)
        if isinstance(value, u.Quantity):
            return float(value.value)
        if isinstance(value, dict):
            if "value" in value:
                return float(value["value"])
            if "lower" in value and isinstance(value["lower"], dict) and "value" in value["lower"]:
                return float(value["lower"]["value"])
            return None
        if value is None:
            return None
        return float(value)

    @classmethod
    def _extract_first_available_quantity_value(cls, point, keys):
        """Extract first available quantity value from a list of candidate keys."""
        for key in keys:
            value = cls._extract_quantity_value(point, key)
            if value is not None:
                return value
        return None

    def _normalize_grid_point(self, point):
        """
        Normalize a grid point from available columns.

        Parameters
        ----------
        point : dict
            Grid-point dictionary.

        Returns
        -------
        dict or None
            Normalized point with native frame metadata.
        """
        azimuth = self._extract_first_available_quantity_value(
            point,
            ("azimuth_angle",),
        )
        zenith = self._extract_first_available_quantity_value(point, ("zenith_angle",))
        hour_angle = self._extract_quantity_value(point, "ha")
        declination = self._extract_quantity_value(point, "dec")
        value_data = {value_key: point.get(value_key) for value_key in PLOT_VALUE_KEYS}
        point_ha = float(hour_angle) if hour_angle is not None else None
        point_dec = float(declination) if declination is not None else None

        if azimuth is not None and zenith is not None:
            return {
                "native_frame": "altaz",
                "azimuth": azimuth % 360.0,
                "zenith": zenith,
                "ha": point_ha,
                "dec": point_dec,
                **value_data,
                "visible_in_altaz": True,
            }

        if hour_angle is not None and declination is not None:
            return {
                "native_frame": "hadec",
                "azimuth": None,
                "zenith": None,
                "ha": float(hour_angle),
                "dec": float(declination),
                **value_data,
                "visible_in_altaz": None,
            }

        logger.warning(f"Skipping point without supported coordinates: {point}")
        return None

    def _normalize_grid_points(self):
        """Normalize all grid points for plotting."""
        normalized_points = []
        for point in self.grid_points:
            normalized_point = self._normalize_grid_point(point)
            if normalized_point is not None:
                normalized_points.append(normalized_point)
        return normalized_points

    def normalize_grid_points(self):
        """Return normalized grid points for plotting and validation."""
        return self._normalize_grid_points()

    @staticmethod
    def _split_points_by_frame(plot_points, frame_name, require_altaz_visibility=False):
        """Split plot points by native coordinate frame."""
        selected_points = []
        for point in plot_points:
            if point["native_frame"] != frame_name:
                continue
            if require_altaz_visibility and not point["visible_in_altaz"]:
                continue
            selected_points.append(point)
        return selected_points

    @staticmethod
    def _resolve_value_unit(plot_points, value_key):
        """Resolve a representative unit string for one plotted value key."""
        for point in plot_points:
            value = point.get(value_key)
            if isinstance(value, u.Quantity):
                return str(value.unit)
        return None

    @staticmethod
    def _plot_value(value):
        """Return a plain numeric value for plotting."""
        if isinstance(value, u.Quantity):
            return float(value.value)
        return float(value)

    @classmethod
    def _format_value_label_with_unit(cls, plot_points, value_key, value_label):
        """Format a plot label with units when available in normalized points."""
        unit_value = cls._resolve_value_unit(plot_points, value_key)
        if unit_value:
            return f"{value_label} [{unit_value}]"
        return value_label

    @staticmethod
    def _has_plottable_hadec_points(plot_points):
        """Return whether normalized points include plottable HA/Dec coordinates."""
        return any(point["ha"] is not None and point["dec"] is not None for point in plot_points)

    @staticmethod
    def _create_projection_axes(show_hadec_panel):
        """Create figure and projection axes based on available coordinate panels."""
        figure = plt.figure(figsize=(15, 7) if show_hadec_panel else (8, 7))
        if show_hadec_panel:
            altaz_axis = figure.add_subplot(1, 2, 1, projection="polar")
            hadec_axis = figure.add_subplot(1, 2, 2)
            return figure, altaz_axis, hadec_axis

        altaz_axis = figure.add_subplot(1, 1, 1, projection="polar")
        return figure, altaz_axis, None

    @staticmethod
    def _add_axis_legend_if_present(axis, location_kwargs):
        """Add a legend to an axis only when visible labels are present."""
        _, labels = axis.get_legend_handles_labels()
        if any(label and not label.startswith("_") for label in labels):
            axis.legend(**location_kwargs)

    def _add_panel_legends(self, altaz_axis, hadec_axis, altaz_count, hadec_count):
        """Add legends to Alt/Az and HA/Dec panels when needed."""
        if altaz_count > 0:
            self._add_axis_legend_if_present(
                altaz_axis,
                {"loc": "upper left", "bbox_to_anchor": (1.0, 1.15)},
            )

        if hadec_axis and hadec_count > 0:
            self._add_axis_legend_if_present(hadec_axis, {"loc": "upper right"})

    def _build_subtitle_lines(self):
        """Build subtitle lines from available grid metadata."""
        subtitle_lines = []
        if self.grid_metadata.get("site"):
            subtitle_lines.append(f"Site: {self.grid_metadata['site']}")
        if self.grid_metadata.get("direction_grid_density") is not None:
            density_unit = self.grid_metadata.get("direction_grid_density_unit") or "1/deg^2"
            subtitle_lines.append(
                f"Grid density: {self.grid_metadata['direction_grid_density']} {density_unit}"
            )
        if self.grid_metadata.get("time_of_observation_utc"):
            subtitle_lines.append(
                f"Observation time (UTC): {self.grid_metadata['time_of_observation_utc']}"
            )
        return subtitle_lines

    @staticmethod
    def _render_subtitle_lines(figure, subtitle_lines):
        """Render subtitle lines below the figure title."""
        for index, subtitle_line in enumerate(subtitle_lines):
            figure.text(
                0.5,
                0.95 - index * 0.025,
                subtitle_line,
                ha="center",
                va="top",
                fontsize=10,
            )

    def plot_sky_projection(self):
        """Create sky projection plots with Alt/Az and HA/Dec grid points."""
        plot_points = self.normalize_grid_points()
        show_hadec_panel = self.has_hadec_columns and self._has_plottable_hadec_points(plot_points)
        figure, altaz_axis, hadec_axis = self._create_projection_axes(show_hadec_panel)

        altaz_count = self._plot_altaz_points(altaz_axis, plot_points)
        hadec_count = self._plot_hadec_points(hadec_axis, plot_points) if hadec_axis else 0
        self._add_panel_legends(altaz_axis, hadec_axis, altaz_count, hadec_count)

        figure.suptitle("Production Grid Points", fontsize=14, y=0.98)
        self._render_subtitle_lines(figure, self._build_subtitle_lines())

        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

        output_file = self.output_path / (
            f"{DEFAULT_OUTPUT_FILE_STEM}.{DEFAULT_OUTPUT_FILE_EXTENSION}"
        )
        figure.savefig(output_file, bbox_inches="tight", dpi=300)
        logger.info(f"Saved sky projection plot to {output_file}")

        plt.close(figure)

    def _configure_altaz_axis(self, axis):
        """Configure the azimuth/zenith polar projection axis."""
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rmax(90)
        axis.set_rticks(np.arange(10, 91, 10))
        axis.grid(True, color="gray", alpha=0.5, linestyle="--")
        axis.set_title("Local Azimuth / Zenith", pad=18)
        axis.set_xlabel("Azimuth [deg]")
        axis.set_ylabel("")
        axis.set_axisbelow(False)

    def _configure_hadec_axis(self, axis, plot_points):
        """Configure the HA/Dec axis."""
        hour_angles = (
            np.array([point["ha"] for point in plot_points]) if plot_points else np.array([])
        )
        declinations = (
            np.array([point["dec"] for point in plot_points]) if plot_points else np.array([])
        )

        axis.grid(True, color="gray", alpha=0.5, linestyle="--")
        axis.set_xlabel("Hour angle [deg]")
        axis.set_ylabel("Declination [deg]")
        axis.set_title("Hour angle / Declination")

        if hour_angles.size > 0:
            ha_min = np.floor(np.min(hour_angles) / 10.0) * 10.0
            ha_max = np.ceil(np.max(hour_angles) / 10.0) * 10.0
            if np.isclose(ha_min, ha_max):
                ha_min -= 5.0
                ha_max += 5.0
            axis.set_xlim(ha_min, ha_max)

        if declinations.size > 0:
            dec_min = np.floor(np.min(declinations) / 5.0) * 5.0
            dec_max = np.ceil(np.max(declinations) / 5.0) * 5.0
            if np.isclose(dec_min, dec_max):
                dec_min -= 5.0
                dec_max += 5.0
            axis.set_ylim(dec_min, dec_max)

    @staticmethod
    def _scatter_group(axis, plot_points, color, x_key, y_key, x_transform=None):
        """Scatter a group of points with configurable coordinates."""
        if not plot_points:
            return 0

        x_values = [point[x_key] for point in plot_points]
        y_values = [point[y_key] for point in plot_points]
        if x_transform is not None:
            x_values = x_transform(x_values)

        axis.scatter(
            x_values,
            y_values,
            s=DEFAULT_MARKER_SIZE,
            linewidths=1.0,
            edgecolors=color,
            facecolors="none",
            zorder=2.2,
        )
        return len(plot_points)

    def _plot_frame_points(
        self,
        axis,
        plot_points,
        primary_frame,
        secondary_frame,
        primary_color,
        secondary_color,
        x_key,
        y_key,
        panel_name,
        require_altaz_visibility=False,
        x_transform=None,
    ):
        """Plot primary and transformed point groups for a given panel."""
        primary_points = self._split_points_by_frame(
            plot_points,
            primary_frame,
            require_altaz_visibility=require_altaz_visibility,
        )
        secondary_points = self._split_points_by_frame(
            plot_points,
            secondary_frame,
            require_altaz_visibility=require_altaz_visibility,
        )

        plotted_points = 0
        plotted_points += self._scatter_group(
            axis,
            primary_points,
            color=primary_color,
            x_key=x_key,
            y_key=y_key,
            x_transform=x_transform,
        )
        plotted_points += self._scatter_group(
            axis,
            secondary_points,
            color=secondary_color,
            x_key=x_key,
            y_key=y_key,
            x_transform=x_transform,
        )

        if plotted_points == 0:
            logger.warning(f"No valid grid points found for {panel_name} plotting")
            return 0

        logger.info(f"Plotted {plotted_points} grid points in the {panel_name} panel")
        return plotted_points

    def _plot_altaz_points(self, axis, plot_points):
        """
        Plot grid points on the Alt/Az projection.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Polar projection axes.
        plot_points : list of dict
            Normalized plot points.
        """
        self._configure_altaz_axis(axis)
        plotted_points = self._plot_frame_points(
            axis=axis,
            plot_points=plot_points,
            primary_frame="altaz",
            secondary_frame="hadec",
            primary_color="tab:blue",
            secondary_color="tab:orange",
            x_key="azimuth",
            y_key="zenith",
            panel_name="Azimuth/Zenith",
            require_altaz_visibility=True,
            x_transform=np.radians,
        )

        hidden_points = sum(point["visible_in_altaz"] is False for point in plot_points)
        if hidden_points > 0:
            logger.info(
                f"Skipping {hidden_points} HA/Dec points below the horizon in Azimuth/Zenith panel"
            )
        return plotted_points

    def _plot_hadec_points(self, axis, plot_points):
        """
        Plot grid points on the HA/Dec projection.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Cartesian axes.
        plot_points : list of dict
            Normalized plot points.
        """
        self._configure_hadec_axis(axis, plot_points)
        return self._plot_frame_points(
            axis=axis,
            plot_points=plot_points,
            primary_frame="hadec",
            secondary_frame="altaz",
            primary_color="tab:orange",
            secondary_color="tab:blue",
            x_key="ha",
            y_key="dec",
            panel_name="HA/Dec",
        )

    @staticmethod
    def _select_altaz_points_with_value(plot_points, value_key):
        """Select native Alt/Az points that have valid coordinates and requested values."""
        return [
            point
            for point in plot_points
            if point.get("native_frame") == "altaz"
            and point.get("azimuth") is not None
            and point.get("zenith") is not None
            and point.get(value_key) is not None
        ]

    def plot_azimuth_zenith_projection_with_color_scale(
        self, value_key, value_label, output_file_stem, plot_points=None
    ):
        """Plot azimuth/zenith points with a color scale for one value column."""
        plot_points = self.normalize_grid_points() if plot_points is None else plot_points
        value_label_with_unit = self._format_value_label_with_unit(
            plot_points,
            value_key,
            value_label,
        )
        colored_points = self._select_altaz_points_with_value(plot_points, value_key)

        if not colored_points:
            logger.warning(f"No azimuth/zenith points with '{value_key}' available for plotting")
            return

        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(1, 1, 1, projection="polar")
        self._configure_altaz_axis(axis)

        azimuth_values = np.radians([point["azimuth"] for point in colored_points])
        zenith_values = [point["zenith"] for point in colored_points]
        color_values = [self._plot_value(point[value_key]) for point in colored_points]

        scatter = axis.scatter(
            azimuth_values,
            zenith_values,
            c=color_values,
            cmap="viridis",
            s=DEFAULT_MARKER_SIZE * 3,
            edgecolors="black",
            linewidths=0.3,
        )

        colorbar = figure.colorbar(scatter, ax=axis, pad=0.12)
        colorbar.set_label(value_label_with_unit)
        axis.set_title(f"Local Azimuth / Zenith colored by {value_label_with_unit}", pad=18)

        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        output_file = self.output_path / f"{output_file_stem}.{DEFAULT_OUTPUT_FILE_EXTENSION}"
        figure.savefig(output_file, bbox_inches="tight", dpi=300)
        logger.info(f"Saved azimuth/zenith color-scale plot to {output_file}")
        plt.close(figure)

    @staticmethod
    def _available_plot_value_keys(plot_points):
        """Return value keys present in at least one normalized Alt/Az point."""
        return [
            value_key
            for value_key in PLOT_VALUE_KEYS
            if any(
                point["native_frame"] == "altaz" and point.get(value_key) is not None
                for point in plot_points
            )
        ]

    def plot_limit_projections(self):
        """Plot all supported production-grid limit values."""
        plot_points = self.normalize_grid_points()
        for value_key in self._available_plot_value_keys(plot_points):
            self.plot_azimuth_zenith_projection_with_color_scale(
                value_key=value_key,
                value_label=value_key,
                output_file_stem=_azimuth_zenith_output_file_stem(value_key),
                plot_points=plot_points,
            )
            self.plot_zenith_limits_for_azimuths(
                value_key=value_key,
                value_label=value_key,
                output_file_stem=_zenith_profile_output_file_stem(value_key),
                plot_points=plot_points,
            )

    @staticmethod
    def _circular_azimuth_difference_degrees(first_azimuth, second_azimuth):
        """Return minimal absolute difference between two azimuth angles."""
        return abs(((first_azimuth - second_azimuth + 180.0) % 360.0) - 180.0)

    def plot_zenith_limits_for_azimuths(
        self,
        value_key,
        value_label,
        output_file_stem,
        azimuth_targets=(0.0, 180.0),
        azimuth_tolerance_deg=1e-6,
        plot_points=None,
    ):
        """Plot value versus zenith for selected azimuth pointings."""
        plot_points = self.normalize_grid_points() if plot_points is None else plot_points
        value_label_with_unit = self._format_value_label_with_unit(
            plot_points,
            value_key,
            value_label,
        )
        altaz_points = self._select_altaz_points_with_value(plot_points, value_key)

        if not altaz_points:
            logger.warning(
                f"No azimuth/zenith points with '{value_key}' available for zenith profiling"
            )
            return

        figure, axis = plt.subplots(figsize=(8, 5))
        plotted_series = 0
        for azimuth_target in azimuth_targets:
            selected_points = [
                point
                for point in altaz_points
                if self._circular_azimuth_difference_degrees(point["azimuth"], azimuth_target)
                <= azimuth_tolerance_deg
            ]
            if not selected_points:
                continue

            selected_points = sorted(selected_points, key=lambda point: point["zenith"])
            zenith_values = [point["zenith"] for point in selected_points]
            value_values = [self._plot_value(point[value_key]) for point in selected_points]
            axis.plot(zenith_values, value_values, marker="o", label=f"az={azimuth_target:.0f} deg")
            plotted_series += 1

        if plotted_series == 0:
            logger.warning(
                f"No zenith profile points for '{value_key}' at azimuths {azimuth_targets}"
            )
            plt.close(figure)
            return

        axis.set_xlabel("Zenith angle [deg]")
        axis.set_ylabel(value_label_with_unit)
        axis.set_title(f"{value_label_with_unit} vs zenith angle")
        axis.grid(True, color="gray", alpha=0.5, linestyle="--")
        axis.legend(loc="best")
        figure.tight_layout()

        output_file = self.output_path / f"{output_file_stem}.{DEFAULT_OUTPUT_FILE_EXTENSION}"
        figure.savefig(output_file, bbox_inches="tight", dpi=300)
        logger.info(f"Saved zenith profile plot to {output_file}")
        plt.close(figure)
