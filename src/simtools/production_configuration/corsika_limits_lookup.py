"""Lookup-table access and interpolation for CORSIKA production limits."""

import logging

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
from scipy.spatial import QhullError  # pylint: disable=no-name-in-module

from simtools.utils.value_conversion import get_value_in_unit

logger = logging.getLogger(__name__)

_LOOKUP_FIELDS = (
    "lower_energy_limit",
    "upper_radius_limit",
    "viewcone_radius",
    "br_energy_min",
    "br_energy_max",
)

_LEGACY_LOOKUP_FIELDS = {
    "lower_energy_threshold": ("lower_energy_limit", u.TeV),
    "upper_scatter_radius": ("upper_radius_limit", u.m),
    "viewcone_radius": ("viewcone_radius", u.deg),
}


def attach_lookup_limits_to_point(point, limits, lookup_field_units=None):
    """Attach interpolated CORSIKA limits to a grid point."""
    for lookup_key, value in limits.items():
        target_key, default_unit = _LEGACY_LOOKUP_FIELDS.get(lookup_key, (lookup_key, None))
        if target_key not in _LOOKUP_FIELDS:
            continue
        if isinstance(value, u.Quantity):
            point[target_key] = value
        elif lookup_field_units is not None and target_key in lookup_field_units:
            point[target_key] = value * lookup_field_units[target_key]
        elif default_unit is not None:
            point[target_key] = value * default_unit
        else:
            point[target_key] = value


class CorsikaLimitsLookup:
    """
    Read and interpolate CORSIKA limits for production grids.

    Parameters
    ----------
    lookup_table : str or Path
        Path to the lookup-table ECSV file.
    array_layout_name : str, optional
        Array layout name used to select lookup-table rows.
    """

    def __init__(self, lookup_table, array_layout_name=None):
        """Initialize lookup-table access."""
        self.lookup_table = lookup_table
        self.array_layout_name = array_layout_name
        self.lookup_points_for_interpolation = None
        self.lookup_values_for_interpolation = None
        self.lookup_interpolators_for_point = None
        self.lookup_nearest_interpolators_for_point = None
        self.lookup_interpolation_axes = None
        self.lookup_points = None
        self.available_lookup_fields = None
        self.lookup_field_units = None

    def load_matching_lookup_arrays(self):
        """
        Load and filter lookup-table arrays for the selected array layout.

        Returns
        -------
        dict
            Lookup arrays for interpolation.
        """
        lookup_table = Table.read(self.lookup_table, format="ascii.ecsv")
        logger.info(
            "Loaded lookup table with %d rows and columns: %s",
            len(lookup_table),
            lookup_table.colnames,
        )
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

        self.available_lookup_fields = [
            field_name for field_name in _LOOKUP_FIELDS if field_name in lookup_table.colnames
        ]
        self.lookup_field_units = {}
        for field_name in self.available_lookup_fields:
            column_unit = lookup_table[field_name].unit
            if column_unit is None:
                raise ValueError(f"Lookup table column '{field_name}' is missing a unit.")
            self.lookup_field_units[field_name] = u.Unit(column_unit)

        lookup_arrays = {
            "points": np.column_stack((zeniths, azimuths, nsb_values)),
        }
        for field_name in self.available_lookup_fields:
            lookup_arrays[field_name] = extract_array(field_name)
        return lookup_arrays

    def prepare_point_interpolators(self):
        """
        Prepare lookup arrays for per-point interpolation.

        Returns
        -------
        dict
            Interpolators keyed by lookup quantity.
        """
        lookup_arrays = self.load_matching_lookup_arrays()
        self.lookup_points = lookup_arrays["points"]
        self.lookup_interpolation_axes = self._get_varying_axes(lookup_arrays["points"])

        if 1 in self.lookup_interpolation_axes:
            self.lookup_points_for_interpolation = self._build_wrapped_interpolation_points(
                lookup_arrays["points"]
            )
            self.lookup_values_for_interpolation = {
                key: self._repeat_wrapped_lookup_values(lookup_arrays[key])
                for key in self.available_lookup_fields
            }
        else:
            self.lookup_points_for_interpolation = lookup_arrays["points"]
            self.lookup_values_for_interpolation = {
                key: lookup_arrays[key] for key in self.available_lookup_fields
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
                for key in self.available_lookup_fields
            }
            self.lookup_nearest_interpolators_for_point = {
                key: NearestNDInterpolator(
                    interpolation_points,
                    self.lookup_values_for_interpolation[key],
                )
                for key in self.available_lookup_fields
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
            interpolated = griddata(
                interpolation_points,
                self._repeat_wrapped_lookup_values(values),
                interpolation_target,
                method="linear",
                fill_value=np.nan,
            )
            if np.isnan(interpolated).any():
                nearest_values = griddata(
                    interpolation_points,
                    self._repeat_wrapped_lookup_values(values),
                    interpolation_target,
                    method="nearest",
                )
                interpolated = np.where(np.isnan(interpolated), nearest_values, interpolated)
            return interpolated.reshape(
                len(target_values["zenith_angle"]),
                len(target_values["azimuth"]),
                len(target_values["nsb_level"]),
            )

        return {key: interpolate(lookup_arrays[key]) for key in self.available_lookup_fields}

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
            Night-sky background rate stored in the lookup table.

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
        interpolated_limits = {}
        for key in self.available_lookup_fields:
            interpolated_value = float(
                self.lookup_interpolators_for_point[key](interpolation_target)[0]
            )
            if np.isnan(interpolated_value):
                interpolated_value = float(
                    self.lookup_nearest_interpolators_for_point[key](interpolation_target)[0]
                )
            interpolated_limits[key] = interpolated_value * self.lookup_field_units[key]
        return interpolated_limits

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
