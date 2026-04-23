"""Plot production-grid points on sky coordinate projections."""

import logging
import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.utils import iers

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_FILE_STEM = "production_grid_sky_projection"
DEFAULT_OUTPUT_FILE_EXTENSION = "png"
DEFAULT_MARKER_SIZE = 8
DEFAULT_GRID_LINE_WIDTH = 0.6
GRID_GROUP_ROUND_DECIMALS = 6


class ProductionGridPlotter:
    """
    Plot production grid points on sky coordinate projections.

    Parameters
    ----------
    grid_points_file : str or Path
        Path to the ECSV file containing grid points.
    site_location_lat : float or astropy.units.Quantity
        Site latitude in degrees.
    site_location_lon : float or astropy.units.Quantity
        Site longitude in degrees.
    site_location_height : float or astropy.units.Quantity
        Site height in meters.
    observation_time : str, optional
        Observation time in UTC ISO format used only for coordinate transformations
        during plotting (e.g. RA/Dec <-> Alt/Az and horizon visibility). It does
        not modify or re-generate the underlying grid points from the input file.
        If None, metadata from the grid file is used.
    output_path : str or Path
        Path to save output plots.
    """

    def __init__(
        self,
        grid_points_file,
        site_location_lat,
        site_location_lon,
        site_location_height,
        observation_time,
        output_path,
    ):
        """Initialize the ProductionGridPlotter."""
        if os.getenv("SIMTOOLS_OFFLINE_IERS", "0") == "1":
            iers.conf.auto_download = False
            iers.conf.auto_max_age = None

        self.grid_points_file = Path(grid_points_file)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        latitude = u.Quantity(site_location_lat, unit=u.deg)
        longitude = u.Quantity(site_location_lon, unit=u.deg)
        height = u.Quantity(site_location_height, unit=u.m)

        self.location = EarthLocation(
            lat=latitude,
            lon=longitude,
            height=height,
        )
        self.grid_metadata = {}
        self.grid_points = self._load_grid_points()

        observation_time_utc = observation_time or self.grid_metadata.get("observing_time_utc")
        if observation_time_utc is None:
            observation_time_utc = "2025-01-01 00:00:00"
        self.time = Time(observation_time_utc, scale="utc")

        logger.info(f"Loaded {len(self.grid_points)} grid points from {self.grid_points_file}")
        logger.info(
            f"Site location: lat={latitude.to_value(u.deg)} deg, "
            f"lon={longitude.to_value(u.deg)} deg"
        )
        logger.info(f"Observation time (UTC): {self.time.isot}")

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
        return self._convert_ecsv_table_to_grid_points(grid_table)

    @staticmethod
    def _convert_ecsv_table_to_grid_points(grid_table):
        """Convert an ECSV grid-point table to plain dictionaries."""
        return [
            {
                column_name: (
                    row[column_name].item()
                    if isinstance(row[column_name], np.generic)
                    else row[column_name]
                )
                for column_name in grid_table.colnames
                if not np.ma.is_masked(row[column_name])
            }
            for row in grid_table
        ]

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
        if isinstance(value, dict):
            if "value" in value:
                return float(value["value"])
            if "lower" in value and isinstance(value["lower"], dict) and "value" in value["lower"]:
                return float(value["lower"]["value"])
            return None
        if value is None:
            return None
        return float(value)

    def _normalize_grid_point(self, point):
        """
        Normalize a grid point to both Alt/Az and RA/Dec when possible.

        Parameters
        ----------
        point : dict
            Grid-point dictionary.

        Returns
        -------
        dict or None
            Normalized point with native frame metadata.
        """
        azimuth = self._extract_quantity_value(point, "azimuth")
        zenith = self._extract_quantity_value(point, "zenith_angle")
        ra = self._extract_quantity_value(point, "ra")
        dec = self._extract_quantity_value(point, "dec")

        if azimuth is not None and zenith is not None:
            alt = (90.0 - zenith) * u.deg
            skycoord = SkyCoord(
                AltAz(
                    alt=alt,
                    az=(azimuth % 360.0) * u.deg,
                    obstime=self.time,
                    location=self.location,
                )
            )
            return {
                "native_frame": "altaz",
                "azimuth": azimuth % 360.0,
                "zenith": zenith,
                "ra": float(skycoord.icrs.ra.deg % 360.0),
                "dec": float(skycoord.icrs.dec.deg),
                "visible_in_altaz": True,
            }

        if ra is not None and dec is not None:
            skycoord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            altaz = skycoord.transform_to(AltAz(obstime=self.time, location=self.location))
            visible_in_altaz = bool(altaz.alt.deg > 0.0)
            return {
                "native_frame": "radec",
                "azimuth": float(altaz.az.deg % 360.0) if visible_in_altaz else None,
                "zenith": float(90.0 - altaz.alt.deg) if visible_in_altaz else None,
                "ra": float(ra % 360.0),
                "dec": float(dec),
                "visible_in_altaz": visible_in_altaz,
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
    def _group_radec_points_by_coordinate(plot_points, coordinate_key, sort_key):
        """Group RA/Dec points by one coordinate and sort by the orthogonal one."""
        grouped_points = {}
        for point in plot_points:
            coordinate_value = round(point[coordinate_key], GRID_GROUP_ROUND_DECIMALS)
            grouped_points.setdefault(coordinate_value, []).append(point)

        ordered_groups = []
        for coordinate_value in sorted(grouped_points):
            group = grouped_points[coordinate_value]
            if len(group) < 2:
                continue
            ordered_groups.append(sorted(group, key=lambda point: point[sort_key]))
        return ordered_groups

    def infer_radec_grid_tracks(self, plot_points):
        """Infer RA and Dec grid tracks from native RA/Dec grid points."""
        native_radec_points = self._split_points_by_frame(plot_points, "radec")
        return {
            "declination_tracks": self._group_radec_points_by_coordinate(
                native_radec_points,
                coordinate_key="dec",
                sort_key="ra",
            ),
            "right_ascension_tracks": self._group_radec_points_by_coordinate(
                native_radec_points,
                coordinate_key="ra",
                sort_key="dec",
            ),
        }

    @staticmethod
    def _split_visible_segments(mask):
        """Split a visibility mask into contiguous visible segments."""
        visible_indices = np.flatnonzero(mask)
        if len(visible_indices) < 2:
            return []

        split_indices = np.nonzero(np.diff(visible_indices) > 1)[0] + 1
        segments = np.split(visible_indices, split_indices)
        return [segment for segment in segments if len(segment) >= 2]

    def plot_sky_projection(self, plot_ra_dec_tracks=False, dec_values=None):
        """
        Create sky projection plots with Alt/Az and RA/Dec grid points.

        Parameters
        ----------
        plot_ra_dec_tracks : bool
            Whether to plot RA/Dec coordinate tracks.
        dec_values : list of float, optional
            List of declination values to plot as tracks.
        """
        figure = plt.figure(figsize=(15, 7))
        altaz_axis = figure.add_subplot(1, 2, 1, projection="polar")
        radec_axis = figure.add_subplot(1, 2, 2)

        plot_points = self.normalize_grid_points()
        altaz_count = self._plot_altaz_points(altaz_axis, plot_points)
        radec_count = self._plot_radec_points(radec_axis, plot_points)
        inferred_grid_count = 0

        if plot_ra_dec_tracks and dec_values:
            self._plot_ra_dec_tracks(altaz_axis, dec_values)
        elif plot_ra_dec_tracks:
            inferred_grid_count = self._plot_inferred_radec_grid(altaz_axis, plot_points)

        if altaz_count > 0 or (plot_ra_dec_tracks and (dec_values or inferred_grid_count > 0)):
            altaz_axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.15))

        if radec_count > 0:
            radec_axis.legend(loc="upper right")

        figure.suptitle(
            f"Production Grid Points\nObservation time: {self.time.iso}",
            fontsize=14,
            y=0.98,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        output_file = self.output_path / (
            f"{DEFAULT_OUTPUT_FILE_STEM}.{DEFAULT_OUTPUT_FILE_EXTENSION}"
        )
        figure.savefig(output_file, bbox_inches="tight", dpi=300)
        logger.info(f"Saved sky projection plot to {output_file}")

        plt.close(figure)

    def _configure_altaz_axis(self, axis):
        """Configure the Alt/Az polar projection axis."""
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rmax(90)
        axis.set_rticks(np.arange(10, 91, 10))
        axis.grid(True, color="gray", alpha=0.5, linestyle="--")
        axis.text(0, 0, "Zenith", ha="center", va="center", fontsize=11, fontweight="bold")
        axis.set_title("Local Alt/Az", pad=18)

    def _configure_radec_axis(self, axis, plot_points):
        """Configure the RA/Dec axis."""
        right_ascensions = (
            np.array([point["ra"] for point in plot_points]) if plot_points else np.array([])
        )
        declinations = (
            np.array([point["dec"] for point in plot_points]) if plot_points else np.array([])
        )

        axis.grid(True, color="gray", alpha=0.5, linestyle="--")
        axis.set_xlabel("Right Ascension [deg]")
        axis.set_ylabel("Declination [deg]")
        axis.set_title("Equatorial RA/Dec")

        if right_ascensions.size > 0:
            ra_min = np.floor(np.min(right_ascensions) / 10.0) * 10.0
            ra_max = np.ceil(np.max(right_ascensions) / 10.0) * 10.0
            if np.isclose(ra_min, ra_max):
                ra_min -= 5.0
                ra_max += 5.0
            axis.set_xlim(ra_max, ra_min)

        if declinations.size > 0:
            dec_min = np.floor(np.min(declinations) / 5.0) * 5.0
            dec_max = np.ceil(np.max(declinations) / 5.0) * 5.0
            if np.isclose(dec_min, dec_max):
                dec_min -= 5.0
                dec_max += 5.0
            axis.set_ylim(dec_min, dec_max)

    @staticmethod
    def _scatter_group(axis, plot_points, label, color, x_key, y_key, x_transform=None):
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
            label=label,
            zorder=10,
        )
        return len(plot_points)

    def _plot_frame_points(
        self,
        axis,
        plot_points,
        primary_frame,
        secondary_frame,
        primary_label,
        secondary_label,
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
            label=primary_label,
            color=primary_color,
            x_key=x_key,
            y_key=y_key,
            x_transform=x_transform,
        )
        plotted_points += self._scatter_group(
            axis,
            secondary_points,
            label=secondary_label,
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
            secondary_frame="radec",
            primary_label="Native Alt/Az points",
            secondary_label="RA/Dec transformed to Alt/Az",
            primary_color="tab:blue",
            secondary_color="tab:orange",
            x_key="azimuth",
            y_key="zenith",
            panel_name="Alt/Az",
            require_altaz_visibility=True,
            x_transform=np.radians,
        )

        hidden_points = sum(not point["visible_in_altaz"] for point in plot_points)
        if hidden_points > 0:
            logger.info(f"Skipping {hidden_points} RA/Dec points below the horizon in Alt/Az panel")
        return plotted_points

    def _plot_radec_points(self, axis, plot_points):
        """
        Plot grid points on the RA/Dec projection.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Cartesian axes.
        plot_points : list of dict
            Normalized plot points.
        """
        self._configure_radec_axis(axis, plot_points)
        return self._plot_frame_points(
            axis=axis,
            plot_points=plot_points,
            primary_frame="radec",
            secondary_frame="altaz",
            primary_label="Native RA/Dec points",
            secondary_label="Alt/Az transformed to RA/Dec",
            primary_color="tab:orange",
            secondary_color="tab:blue",
            x_key="ra",
            y_key="dec",
            panel_name="RA/Dec",
        )

    def _plot_radec_track_group(self, axis, group, color, label=None, linestyle="-"):
        """Plot a single inferred RA/Dec track group on the Alt/Az panel."""
        sky_coordinates = SkyCoord(
            ra=np.array([point["ra"] for point in group]) * u.deg,
            dec=np.array([point["dec"] for point in group]) * u.deg,
            frame="icrs",
        )
        altaz = sky_coordinates.transform_to(AltAz(obstime=self.time, location=self.location))
        visible_segments = self._split_visible_segments(altaz.alt.deg > 0.0)

        plotted_segments = 0
        show_label = label
        for segment in visible_segments:
            axis.plot(
                altaz.az.rad[segment],
                90.0 - altaz.alt.deg[segment],
                color=color,
                lw=DEFAULT_GRID_LINE_WIDTH,
                linestyle=linestyle,
                alpha=0.8,
                label=show_label,
                zorder=3,
            )
            plotted_segments += 1
            show_label = None
        return plotted_segments

    def _plot_inferred_radec_grid(self, axis, plot_points):
        """Plot thin inferred RA/Dec grid lines on the Alt/Az panel."""
        track_groups = self.infer_radec_grid_tracks(plot_points)

        plotted_tracks = 0
        for index, group in enumerate(track_groups["declination_tracks"]):
            plotted_tracks += self._plot_radec_track_group(
                axis,
                group,
                color="0.45",
                linestyle="-",
                label="Inferred Dec grid" if index == 0 else None,
            )

        for index, group in enumerate(track_groups["right_ascension_tracks"]):
            plotted_tracks += self._plot_radec_track_group(
                axis,
                group,
                color="0.6",
                linestyle=":",
                label="Inferred RA grid" if index == 0 else None,
            )

        if plotted_tracks == 0:
            logger.info("No inferred RA/Dec grid tracks available for plotting")
        else:
            logger.info(f"Plotted {plotted_tracks} inferred RA/Dec grid track segments")

        return plotted_tracks

    def _plot_ra_dec_tracks(self, axis, dec_values):
        """
        Plot RA/Dec coordinate tracks on the polar projection.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Polar projection axes.
        dec_values : list of float
            List of declination values in degrees.
        """
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(dec_values)))

        for dec_val, color in zip(dec_values, colors):
            right_ascension = np.linspace(0, 360, 400) * u.deg
            declination = dec_val * u.deg

            coords = SkyCoord(ra=right_ascension, dec=declination, frame="icrs")
            altaz = coords.transform_to(AltAz(obstime=self.time, location=self.location))

            az_track = altaz.az.rad
            zenith_track = 90 - altaz.alt.deg

            mask = altaz.alt.deg > 0
            if np.any(mask):
                axis.plot(
                    az_track[mask],
                    zenith_track[mask],
                    color=color,
                    lw=2,
                    label=f"Dec = {dec_val:.1f} deg",
                    alpha=0.7,
                )

        logger.info(f"Plotted RA/Dec tracks for {len(dec_values)} declination values")
