"""Lookup-table access and interpolation for CORSIKA production limits."""

import json

import numpy as np
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import QhullError  # pylint: disable=no-name-in-module

from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)


def _value_in_unit(value, unit=None):
    """Return a scalar value converted to the requested unit when possible."""
    if hasattr(value, "to_value"):
        return value.to_value(unit) if unit is not None else value.value
    return value


class CorsikaLimitsLookup:
    """
    Read and interpolate CORSIKA limits for production grids.

    The lookup table is shared by both grid serialization and backend-specific
    job-spec generation.
    """

    def __init__(self, lookup_table, telescope_ids=None, simtel_file=None):
        """
        Initialize lookup-table access.

        Parameters
        ----------
        lookup_table : str or Path
            Path to the lookup-table ECSV file.
        telescope_ids : list or str, optional
            Telescope selection used to filter lookup-table rows.
        simtel_file : str, optional
            Path to a sim_telarray file used to resolve numeric telescope IDs.
        """
        self.lookup_table = lookup_table
        self.telescope_ids = telescope_ids
        self.simtel_file = simtel_file
        self._simtel_id_to_name = (
            get_sim_telarray_telescope_id_to_telescope_name_mapping(simtel_file)
            if simtel_file
            else {}
        )
        self.lookup_points_for_interpolation = None
        self.lookup_values_for_interpolation = None
        self.lookup_interpolators_for_point = None

    @staticmethod
    def _coerce_identifier_container(value):
        """Coerce identifier input into a list."""
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return json.loads(stripped) if stripped.startswith("[") else [stripped]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def coerce_identifier_container(value):
        """Coerce identifier input into a list."""
        return CorsikaLimitsLookup._coerce_identifier_container(value)

    def _normalize_lookup_identifier(self, identifier):
        """Normalize one telescope identifier and report if it is numeric."""
        if isinstance(identifier, (int, np.integer)):
            return self._simtel_id_to_name.get(int(identifier), str(int(identifier))), True

        text = str(identifier).strip()
        if text.lstrip("+-").isdigit():
            return self._simtel_id_to_name.get(int(text), text), True
        return text, False

    def normalize_lookup_identifier(self, identifier):
        """Normalize one telescope identifier and report if it is numeric."""
        return self._normalize_lookup_identifier(identifier)

    def _normalized_identifier_set(self, identifiers):
        """Return normalized telescope identifiers as a set."""
        return {
            self._normalize_lookup_identifier(identifier)[0]
            for identifier in self._coerce_identifier_container(identifiers)
        }

    def normalized_identifier_set(self, identifiers):
        """Return normalized telescope identifiers as a set."""
        return self._normalized_identifier_set(identifiers)

    def _lookup_contains_numeric_telescope_ids(self, lookup_table):
        """Return True when any lookup-table telescope identifier is numeric."""
        return any(
            any(
                self._normalize_lookup_identifier(identifier)[1]
                for identifier in self._coerce_identifier_container(row["telescope_ids"])
            )
            for row in lookup_table
        )

    def lookup_contains_numeric_telescope_ids(self, lookup_table):
        """Return True when any lookup-table telescope identifier is numeric."""
        return self._lookup_contains_numeric_telescope_ids(lookup_table)

    @property
    def simtel_id_to_name(self):
        """Return the sim_telarray telescope-ID mapping."""
        return self._simtel_id_to_name

    @simtel_id_to_name.setter
    def simtel_id_to_name(self, value):
        """Set the sim_telarray telescope-ID mapping."""
        self._simtel_id_to_name = value

    def load_matching_lookup_arrays(self):
        """
        Load and filter lookup-table arrays for the selected telescope IDs.

        Returns
        -------
        dict
            Lookup arrays for interpolation.
        """
        lookup_table = Table.read(self.lookup_table, format="ascii.ecsv")
        selected_telescope_ids = self._normalized_identifier_set(self.telescope_ids)

        matching_rows = [
            row
            for row in lookup_table
            if selected_telescope_ids == self._normalized_identifier_set(row["telescope_ids"])
        ]

        if not matching_rows:
            has_numeric_lookup_ids = self._lookup_contains_numeric_telescope_ids(lookup_table)

            if has_numeric_lookup_ids and not self.simtel_file:
                raise ValueError(
                    "Lookup table telescope selections contain numeric IDs. "
                    "Provide --simtel_file to map those IDs to telescope names."
                )

            raise ValueError(
                f"No matching rows in the lookup table for telescope_ids: {self.telescope_ids}"
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
        points = lookup_arrays["points"]
        azimuths = points[:, 1]
        azimuths_wrapped = np.concatenate([azimuths + shift for shift in (0, 360, -360)])

        def repeat_3(arr):
            """Repeat an array three times."""
            return np.tile(arr, 3)

        self.lookup_points_for_interpolation = np.column_stack(
            (
                repeat_3(points[:, 0]),
                azimuths_wrapped,
                repeat_3(points[:, 2]),
            )
        )
        self.lookup_values_for_interpolation = {
            "lower_energy_threshold": repeat_3(lookup_arrays["lower_energy_threshold"]),
            "upper_scatter_radius": repeat_3(lookup_arrays["upper_scatter_radius"]),
            "viewcone_radius": repeat_3(lookup_arrays["viewcone_radius"]),
        }
        try:
            self.lookup_interpolators_for_point = {
                key: LinearNDInterpolator(
                    self.lookup_points_for_interpolation,
                    self.lookup_values_for_interpolation[key],
                    fill_value=np.nan,
                )
                for key in ("lower_energy_threshold", "upper_scatter_radius", "viewcone_radius")
            }
        except QhullError as exc:
            raise ValueError(
                "Lookup table does not contain enough unique points for 3D interpolation "
                "(zenith, azimuth, nsb_level). Provide a denser lookup table "
                "or run without --lookup_table if limits are not required for this use case."
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
        points_base = lookup_arrays["points"]
        lower_energy_thresholds = lookup_arrays["lower_energy_threshold"]
        upper_scatter_radii = lookup_arrays["upper_scatter_radius"]
        viewcone_radii = lookup_arrays["viewcone_radius"]

        zeniths = points_base[:, 0]
        azimuths = points_base[:, 1]
        nsb_values = points_base[:, 2]
        azimuths_wrapped = np.concatenate([azimuths + shift for shift in (0, 360, -360)])

        def repeat_3(arr):
            """Repeat an array three times."""
            return np.tile(arr, 3)

        points = np.column_stack(
            (
                repeat_3(zeniths),
                azimuths_wrapped,
                repeat_3(nsb_values),
            )
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

        def interpolate(values):
            return griddata(
                points, repeat_3(values), target_grid, method="linear", fill_value=np.nan
            ).reshape(
                len(target_values["zenith_angle"]),
                len(target_values["azimuth"]),
                len(target_values["nsb_level"]),
            )

        return {
            "lower_energy_threshold": interpolate(lower_energy_thresholds),
            "upper_scatter_radius": interpolate(upper_scatter_radii),
            "viewcone_radius": interpolate(viewcone_radii),
        }

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
                    _value_in_unit(zenith),
                    _value_in_unit(azimuth) % 360.0,
                    _value_in_unit(nsb),
                ]
            ],
            dtype=float,
        )
        return {
            "lower_energy_threshold": float(
                self.lookup_interpolators_for_point["lower_energy_threshold"](target)[0]
            ),
            "upper_scatter_radius": float(
                self.lookup_interpolators_for_point["upper_scatter_radius"](target)[0]
            ),
            "viewcone_radius": float(
                self.lookup_interpolators_for_point["viewcone_radius"](target)[0]
            ),
        }
