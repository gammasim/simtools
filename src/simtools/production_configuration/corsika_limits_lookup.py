"""Lookup-table access and interpolation for CORSIKA production limits."""

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import QhullError  # pylint: disable=no-name-in-module

from simtools.utils.value_conversion import get_value_in_unit

_LOOKUP_FIELDS = (
    "lower_energy_threshold",
    "upper_scatter_radius",
    "viewcone_radius",
)

_LOOKUP_FIELD_UNITS = {
    "lower_energy_threshold": "TeV",
    "upper_scatter_radius": "m",
    "viewcone_radius": "deg",
}

_POINT_LIMIT_KEYS = {
    "lower_energy_threshold": "lower_energy_threshold",
    "upper_scatter_radius": "scatter_radius",
    "viewcone_radius": "viewcone_radius",
}


def attach_lookup_limits_to_point(point, limits):
    """Attach interpolated CORSIKA limits to a grid point."""
    for lookup_key, point_key in _POINT_LIMIT_KEYS.items():
        point[point_key] = limits[lookup_key] * u.Unit(_LOOKUP_FIELD_UNITS[lookup_key])


class CorsikaLimitsLookup:
    """Read and interpolate CORSIKA limits for production grids."""

    def __init__(self, lookup_table, array_layout_name=None):
        """
        Initialize lookup-table access.

        Parameters
        ----------
        lookup_table : str or Path
            Path to the lookup-table ECSV file.
        array_layout_name : str, optional
            Array layout name used to select lookup-table rows.
        """
        self.lookup_table = lookup_table
        self.array_layout_name = array_layout_name
        self.lookup_points_for_interpolation = None
        self.lookup_values_for_interpolation = None
        self.lookup_interpolators_for_point = None
        self.lookup_interpolation_axes = None

    def load_matching_lookup_arrays(self):
        """
        Load and filter lookup-table arrays for the selected array layout.

        Returns
        -------
        dict
            Lookup arrays for interpolation.
        """
        lookup_table = Table.read(self.lookup_table, format="ascii.ecsv")
        if self.array_layout_name is None:
            matching_rows = list(lookup_table)
        else:
            matching_rows = [
                row for row in lookup_table if str(row["array_name"]) == str(self.array_layout_name)
            ]

        if not matching_rows:
            raise ValueError(
                "No matching rows in the lookup table for "
                f"array_layout_name: {self.array_layout_name}"
            )

        def extract_array(field, transform=lambda x: x):
            return np.array([transform(row[field]) for row in matching_rows])

        zeniths = extract_array("zenith")
        azimuths = extract_array("azimuth", lambda x: x % 360)
        nsb_values = extract_array("nsb_level", float)

        return {
            "points": np.column_stack((zeniths, azimuths, nsb_values)),
            "lower_energy_threshold": extract_array("lower_energy_limit"),
            "upper_scatter_radius": extract_array("upper_radius_limit"),
            "viewcone_radius": extract_array("viewcone_radius"),
        }

    def prepare_point_interpolators(self):
        """
        Prepare lookup arrays for per-point interpolation.

        Returns
        -------
        dict
            Interpolators keyed by lookup quantity.
        """
        lookup_arrays = self.load_matching_lookup_arrays()
        self.lookup_points_for_interpolation = self._build_wrapped_interpolation_points(
            lookup_arrays["points"]
        )
        self.lookup_interpolation_axes = self._get_varying_axes(lookup_arrays["points"])
        self.lookup_values_for_interpolation = {
            key: self._repeat_wrapped_lookup_values(lookup_arrays[key]) for key in _LOOKUP_FIELDS
        }

        if len(self.lookup_interpolation_axes) < 2:
            raise ValueError(
                "Lookup table does not contain enough unique points for interpolation. "
                "At least two varying dimensions are required among "
                "(zenith, azimuth, nsb_level)."
            )

        interpolation_points = self.lookup_points_for_interpolation[
            :, self.lookup_interpolation_axes
        ]
        try:
            self.lookup_interpolators_for_point = {
                key: LinearNDInterpolator(
                    interpolation_points,
                    self.lookup_values_for_interpolation[key],
                    fill_value=np.nan,
                )
                for key in _LOOKUP_FIELDS
            }
        except QhullError as exc:
            raise ValueError(
                "Lookup table does not contain enough unique points for interpolation in "
                f"{len(self.lookup_interpolation_axes)}D among (zenith, azimuth, nsb_level). "
                "Provide a denser lookup table "
                "or run without --corsika_limits if limits are not required for this use case."
            ) from exc
        return self.lookup_interpolators_for_point

    def interpolate_grid_limits(self, target_values):
        """
        Interpolate lookup values on a regular zenith/azimuth/NSB grid.

        Parameters
        ----------
        target_values : dict
            Generated target-axis values.

        Returns
        -------
        dict
            Interpolated lookup values on the target grid.
        """
        lookup_arrays = self.load_matching_lookup_arrays()
        points = self._build_wrapped_interpolation_points(lookup_arrays["points"])
        interpolation_axes = self._get_varying_axes(lookup_arrays["points"])

        if len(interpolation_axes) < 2:
            raise ValueError(
                "Lookup table does not contain enough unique points for interpolation. "
                "At least two varying dimensions are required among "
                "(zenith, azimuth, nsb_level)."
            )

        target_grid = (
            np.array(
                np.meshgrid(
                    target_values["zenith_angle"].value,
                    target_values["azimuth"].value,
                    target_values["nsb_level"].value,
                    indexing="ij",
                )
            )
            .reshape(3, -1)
            .T
        )
        interpolation_points = points[:, interpolation_axes]
        interpolation_target = target_grid[:, interpolation_axes]

        def interpolate(values):
            return griddata(
                interpolation_points,
                self._repeat_wrapped_lookup_values(values),
                interpolation_target,
                method="linear",
                fill_value=np.nan,
            ).reshape(
                len(target_values["zenith_angle"]),
                len(target_values["azimuth"]),
                len(target_values["nsb_level"]),
            )

        return {key: interpolate(lookup_arrays[key]) for key in _LOOKUP_FIELDS}

    def interpolate_point(self, zenith, azimuth, nsb=1.0):
        """
        Interpolate lookup-table limits for a single point.

        Parameters
        ----------
        zenith : float or Quantity
            Zenith angle.
        azimuth : float or Quantity
            Azimuth angle.
        nsb : float or Quantity, optional
            Night-sky background level.

        Returns
        -------
        dict
            Interpolated lower-energy threshold, scatter radius, and viewcone radius.
        """
        if self.lookup_interpolators_for_point is None:
            self.prepare_point_interpolators()

        target = np.array(
            [
                [
                    get_value_in_unit(zenith, "deg"),
                    get_value_in_unit(azimuth, "deg") % 360.0,
                    get_value_in_unit(nsb),
                ]
            ],
            dtype=float,
        )
        interpolation_target = target[:, self.lookup_interpolation_axes]
        return {
            "lower_energy_threshold": float(
                self.lookup_interpolators_for_point["lower_energy_threshold"](interpolation_target)[
                    0
                ]
            ),
            "upper_scatter_radius": float(
                self.lookup_interpolators_for_point["upper_scatter_radius"](interpolation_target)[0]
            ),
            "viewcone_radius": float(
                self.lookup_interpolators_for_point["viewcone_radius"](interpolation_target)[0]
            ),
        }

    @staticmethod
    def _get_varying_axes(points):
        """Return coordinate-axis indices with more than one unique value."""
        return [axis for axis in range(points.shape[1]) if np.unique(points[:, axis]).size > 1]

    @staticmethod
    def _repeat_wrapped_lookup_values(values):
        """Repeat lookup values for azimuth wrapping."""
        return np.tile(values, 3)

    def _build_wrapped_interpolation_points(self, points):
        """Return lookup points extended across azimuth wrap boundaries."""
        azimuths_wrapped = np.concatenate([points[:, 1] + shift for shift in (0, 360, -360)])
        return np.column_stack(
            (
                self._repeat_wrapped_lookup_values(points[:, 0]),
                azimuths_wrapped,
                self._repeat_wrapped_lookup_values(points[:, 2]),
            )
        )
