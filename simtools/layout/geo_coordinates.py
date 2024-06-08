"""Definition of geospatial coordinate systems."""

import logging

import astropy.units as u
import numpy as np
import pyproj


class GeoCoordinates:
    """
    Geospatial Coordinate systems.

    Defines UTM, WGS84 and ground (sim_telarray) coordinate systems.

    """

    def __init__(self):
        """Initialize GeoCoordinates."""
        self._logger = logging.getLogger(__name__)

    def crs_utm(self, epsg):
        """
        UTM coordinate system definition.

        Parameters
        ----------
        epsg: int
            EPSG code for UTM zone.

        Returns
        -------
        pyproj.CRS
            UTM coordinate system.

        """
        if epsg:
            crs_utm = pyproj.CRS.from_user_input(epsg)
            self._logger.debug(f"UTM coordinate system: {crs_utm}")
            return crs_utm

        return None

    @staticmethod
    def crs_wgs84():
        """
        WGS84 coordinate system definition.

        Returns
        -------
        pyproj.CRS
            WGS84 coordinate system.

        """
        return pyproj.CRS("EPSG:4326")

    def crs_local(self, reference_point):
        """
        Local coordinate system definition.

        This is a cartesian coordinate system with the origin at the array center.
        X-axis points towards geographic North, y-axis towards geographic West.

        Parameters
        ----------
        reference_point: simtools.layout.telescope_position
            Reference coordinate.

        Returns
        -------
        pyproj.CRS
            local coordinate system.

        """
        try:
            if self._valid_reference_point(reference_point):
                _center_lat, _center_lon, _ = reference_point.get_coordinates("mercator")
                _scale_factor_k_0 = self._coordinate_scale_factor(reference_point)
                proj4_string = (
                    "+proj=tmerc +ellps=WGS84 +datum=WGS84"
                    f" +lon_0={_center_lon} +lat_0={_center_lat}"
                    f" +axis=nwu +units=m +k_0={_scale_factor_k_0}"
                )
                crs_local = pyproj.CRS.from_proj4(proj4_string)
                self._logger.debug(f"Ground (sim_telarray) coordinate system: {crs_local}")
                return crs_local
        except AttributeError:
            self._logger.error("Failed to derive local coordinate system. Missing reference point")
            raise

        return None

    def _valid_reference_point(self, reference_point):
        """
        Check if reference point has valid long/lat coordinates (including altitude).

        This is required to derive the local coordinate system.
        Try if a conversion from UTM coordinates to long/lat is possible.

        Parameters
        ----------
        reference_point: simtools.layout.telescope_position
            Reference coordinate.

        Returns
        -------
        bool
            True if reference point has valid coordinates.

        """
        _center_lat, _center_lon, _center_alt = reference_point.get_coordinates("mercator")
        if np.isnan(_center_alt.value):
            self._logger.debug("Missing array center altitude")
            return False

        if np.isnan(_center_lat.value) or np.isnan(_center_lon.value):
            self._logger.debug(
                "Invalid array center coordinates "
                f"(lat={_center_lat}, lon={_center_lon}, alt={_center_alt})"
            )
            return False

        return True

    def _coordinate_scale_factor(self, reference_point):
        """
        Derive coordinate scale factor for transformation into local coordinate system.

        Depends on latitude and altitude of array center.
        Ignores transformation of geodetic height to geocentric height
        (this introduces an error on the sub-mm scale).

        Parameters
        ----------
        reference_point: simtools.layout.telescope_position
            Reference coordinate.

        Returns
        -------
        k_0: float
            Scale factor for local coordinate system.

        Raises
        ------
        AttributeError
            If reference_point does not have a valid center or UTM system is not defined.

        """
        try:
            _center_lat, _, _centre_altitude = reference_point.get_coordinates("mercator")
        except AttributeError:
            self._logger.debug("Missing array center, cannot derive coordinate scale factor")
            raise

        crs_utm = self.crs_wgs84()
        _semi_major_axis = crs_utm.geodetic_crs.ellipsoid.semi_major_metre * u.m
        _semi_minor_axis = crs_utm.geodetic_crs.ellipsoid.semi_minor_metre * u.m

        _local_geo_radius = self._geocentric_radius(_center_lat, _semi_major_axis, _semi_minor_axis)

        return (_local_geo_radius.value + _centre_altitude.value) / _local_geo_radius.value

    def _geocentric_radius(self, latitude, semi_major_axis, semi_minor_axis):
        """
        Calculate geocentric radius from WGS84 ellipsoid parameters.

        derivation:
        https://gis.stackexchange.com/questions/20200/how-do-you-compute-the-earths-radius-at-a-given-geodetic-latitude

        Parameters
        ----------
        latitude: astropy.Quantity
            Latitude in degrees.
        semi_major_axis: astropy.Quantity
            Semi-major axis of ellipsoid in meters.
        semi_minor_axis: astropy.Quantity
            Semi-minor axis of ellipsoid in meters.

        Returns
        -------
        float
            Ellipsoid radius at given latitude.

        """
        _lat_rad = np.deg2rad(latitude)
        _numerator = (semi_major_axis**2 * np.cos(_lat_rad)) ** 2 + (
            semi_minor_axis**2 * np.sin(_lat_rad)
        ) ** 2

        _denominator = (
            semi_major_axis**2 * np.cos(_lat_rad) ** 2 + semi_minor_axis**2 * np.sin(_lat_rad) ** 2
        )
        return np.sqrt(_numerator / _denominator)
