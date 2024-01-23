import logging

import astropy.units as u
import numpy as np
import pyproj

__all__ = ["InvalidCoordSystem", "TelescopePosition"]


class InvalidCoordSystem(Exception):
    """Exception for invalid coordinate system."""


class TelescopePosition:
    """
    Store and perform coordinate transformations for an array element position.

    The definition of x_coord and y_coord in this class depend on the \
    coordinate system (e.g., (x_coord, y_coord) == (UTM_east, UTM_north)). \
    Altitude describes always the element height above sea level.

    Parameters
    ----------
    name: str
        Name of the telescope (e.g LST-01, SST-05, ...).

    """

    def __init__(self, name=None):
        """
        Initialize TelescopePosition.
        """

        self._logger = logging.getLogger(__name__)

        self.name = name
        self.asset_code = None
        self.sequence_number = None
        self.geo_code = None
        self.crs = self._default_coordinate_system_definition()

    def __str__(self):
        """
        String representation of TelescopePosition.

        """
        telstr = self.name
        if self.has_coordinates("ground"):
            telstr += (
                f"\t Ground x(->North): {self.crs['ground']['xx']['value']:0.2f} "
                f"y(->West): {self.crs['ground']['yy']['value']:0.2f}"
            )
        if self.has_coordinates("utm"):
            telstr += (
                f"\t UTM East: {self.crs['utm']['xx']['value']:0.2f} "
                f"UTM North: {self.crs['utm']['yy']['value']:0.2f}"
            )
        if self.has_coordinates("mercator"):
            telstr += (
                f"\t Longitude: {self.crs['mercator']['xx']['value']:0.5f} "
                f"Latitude: {self.crs['mercator']['yy']['value']:0.5f}"
            )
        for _crs_name, _crs_now in self.crs.items():
            if self.has_altitude(_crs_name):
                telstr += f"\t Alt: {_crs_now['zz']['value']:0.2f}"
                break

        return telstr

    def print_compact_format(
        self, crs_name, print_header=False, corsika_obs_level=None, corsika_sphere_center=None
    ):
        """
        Print array element coordinates in compact format.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for printing.
        print_header: bool
            Print table header.
        corsika_obs_level: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsika_sphere_center: astropy.Quantity
            CORSIKA sphere center in equivalent units of meter.

        Raises
        ------
        InvalidCoordSystem
           if coordinate system is not defined.
        """

        try:
            _zz = self.crs[crs_name]["zz"]["value"]
            _zz_header = self.crs[crs_name]["zz"]["name"]
            if (
                crs_name == "ground"
                and corsika_obs_level is not None
                and corsika_sphere_center is not None
            ):
                _zz = (
                    self.convert_telescope_altitude_to_corsika_system(
                        _zz * u.Unit(self.crs[crs_name]["zz"]["unit"]),
                        corsika_obs_level,
                        corsika_sphere_center,
                    )
                ).value
                _zz_header = "position_z"

            if crs_name == "mercator":
                telstr = (
                    f"{self.name} {self.crs[crs_name]['xx']['value']:10.8f} "
                    f"{self.crs[crs_name]['yy']['value']:10.8f} {_zz:10.2f}"
                )
            else:
                telstr = (
                    f"{self.name} {self.crs[crs_name]['xx']['value']:10.2f} "
                    f"{self.crs[crs_name]['yy']['value']:10.2f} {_zz:10.2f}"
                )
            headerstr = (
                f"telescope_name {self.crs[crs_name]['xx']['name']} "
                f"{self.crs[crs_name]['yy']['name']} {_zz_header}"
            )

            if self.geo_code is not None:
                telstr += f"  {self.geo_code}"
                headerstr += "  geo_code"
            if print_header:
                print(headerstr)
            print(telstr)
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystem from e

    def get_coordinates(self, crs_name, coordinate_field=None):
        """
        Get coordinates in a given coordinate system.

        Parameters
        ----------
        crs_name: str
            name of coordinate system.
        coordinate_field: str
            return specified field of coordinate descriptor.

        Returns
        -------
        x, y, z coordinate including corresponding unit (or coordinate field). If coordinate_field \
        is None, returns value x unit.

        Raises
        ------
        InvalidCoordSystem
           if coordinate system is not defined

        """
        if coordinate_field is None:
            try:
                return (
                    self.crs[crs_name]["xx"]["value"] * self.crs[crs_name]["xx"]["unit"],
                    self.crs[crs_name]["yy"]["value"] * self.crs[crs_name]["yy"]["unit"],
                    self.crs[crs_name]["zz"]["value"] * self.crs[crs_name]["zz"]["unit"],
                )
            except KeyError as e:
                self._logger.error(f"Invalid coordinate system ({crs_name})")
                raise InvalidCoordSystem from e
        else:
            try:
                return (
                    self.crs[crs_name]["xx"][coordinate_field],
                    self.crs[crs_name]["yy"][coordinate_field],
                    self.crs[crs_name]["zz"][coordinate_field],
                )
            except KeyError as e:
                self._logger.error(
                    f"Invalid coordinate system ({crs_name}) "
                    f"or coordinate field ({coordinate_field})"
                )
                raise InvalidCoordSystem from e

    def _get_coordinate_value(self, value, unit):
        """
        Return a value of a coordinate variable
        i) converted to the given unit, if input value has a unit
        ii) return input value without change, if no unit is given

        """

        if isinstance(value, u.Quantity):
            try:
                return value.to(unit).value
            except u.UnitsError:
                self._logger.error(f"Invalid unit given ({unit}) for value: {value})")
                raise

        return value

    def set_coordinates(self, crs_name, xx, yy, zz=None):
        """
        Set coordinates of an array element.

        Attributes
        ----------
        crs_name: str
            Name of coordinate system.
        xx: float
            x-coordinate.
        yy: float
            y-coordinate.
        zz: float
            z-coordinate (altitude).

        Raises
        ------
        InvalidCoordSystem
            If coordinate system is not known.

        """

        try:
            self.crs[crs_name]["xx"]["value"] = self._get_coordinate_value(
                xx, self.crs[crs_name]["xx"]["unit"]
            )
            self.crs[crs_name]["yy"]["value"] = self._get_coordinate_value(
                yy, self.crs[crs_name]["yy"]["unit"]
            )
            if zz is not None:
                self.crs[crs_name]["zz"]["value"] = self._get_coordinate_value(
                    zz, self.crs[crs_name]["zz"]["unit"]
                )
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystem from e

    def get_altitude(self):
        """ "
        Get altitude of an array element

        Returns
        -------
        astropy.Quantity
            telescope altitude.

        """
        for _crs in self.crs.values():
            if _crs["zz"]["value"]:
                return _crs["zz"]["value"] * u.Unit(_crs["zz"]["unit"])

        return np.nan

    def set_altitude(self, tel_altitude):
        """
        Set altitude of an array element. Assume that all coordinate system have same altitude \
        definition, meaning altitude is set for all systems here.

        Attributes
        ----------
        tel_altitude: astropy.Quantity
        """

        for _crs in self.crs.values():
            _crs["zz"]["value"] = self._get_coordinate_value(tel_altitude, _crs["zz"]["unit"])

    def _convert(self, crs_from, crs_to, xx, yy):
        """
        Coordinate transformation of telescope positions. Returns np.nan for failed transformations\
        (and not inf, as pyproj does)

        Parameters
        ----------
        crs_from: pyproj.crs.crs.CRS
            Projection of input data
        crs_to: pyproj.crs.crs.CRS
            Projection of output data
        xx: scalar
            Input x coordinate
        yy: scalar
            Input y coordinate

        Returns
        -------
        scalar
            Output x coordinate
        scalar
            Output y coordinate

        Raises
        ------
        pyproj.exceptions.CRSError
            If input or output projection (coordinate system) is not defined

        """
        try:
            transformer = pyproj.Transformer.from_crs(crs_from, crs_to)
        except pyproj.exceptions.CRSError:
            self._logger.error("Invalid coordinate system")
            raise
        if xx is None or yy is None:
            return np.nan, np.nan
        _to_x, _to_y = transformer.transform(xx=xx, yy=yy)
        if np.isinf(_to_x) or np.isinf(_to_y):
            return np.nan, np.nan
        return _to_x, _to_y

    def _get_reference_system_from(self):
        """
        Return coordinate system and coordinates for a fully defined system. The first fully\
        defined system from self.crs is returned.

        Returns
        -------
        string
            Name of coordinate system
        pyproj.crs.crs.CRS
            Project of coordinate system

        """

        for _crs_name, _crs in self.crs.items():
            if self.has_coordinates(_crs_name, crs_check=True):
                return _crs_name, _crs
        return None, None

    def has_coordinates(self, crs_name, crs_check=False):
        """
        Check if coordinates are set for a given coordinate system.

        Attributes
        ----------
        crs_name: str
            Name of coordinate system.
        crs_check: bool
            Check that projection system is defined.

        Returns
        -------
        bool
            True if coordinate system is defined.

        Raises
        ------
        InvalidCoordSystem
            If coordinate system is not known.
        """
        try:
            if not self.crs[crs_name]["crs"] and crs_check:
                return False
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystem from e

        return np.all(
            np.isfinite(
                np.array(
                    [self.crs[crs_name]["xx"]["value"], self.crs[crs_name]["yy"]["value"]],
                    dtype=np.float64,
                )
            )
        )

    def has_altitude(self, crs_name=None):
        """
        Return True if array element has altitude defined.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be checked for altitude. If None: check if altitude is \
            define for any system.

        Returns
        -------
        bool
            array element has altitude defined.

        Raises
        ------
        InvalidCoordSystem
            If coordinate system is not known

        """
        if crs_name is None:
            for _crs_name in self.crs:
                if self.has_altitude(_crs_name):
                    return True
            return False

        try:
            return (
                self.crs[crs_name]["zz"]["value"] is not np.nan
                and self.crs[crs_name]["zz"]["value"] is not None
            )
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystem from e

    def _set_coordinate_system(self, crs_name, crs_system):
        """
        Set a coordinate system with a given name.

        Attributes
        ----------
        crs_name: str
            Name of coordinate system.
        crs_system: pyproj.crs.crs.CRS
             Project of coordinate system.

        Raises
        ------
        InvalidCoordSystem
            If coordinate system is not known.

        """
        try:
            self.crs[crs_name]["crs"] = crs_system
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystem from e

    @staticmethod
    @u.quantity_input(tel_altitude=u.m, corsika_obs_level=u.m, corsika_sphere_center=u.m)
    def convert_telescope_altitude_to_corsika_system(
        tel_altitude, corsika_obs_level, corsika_sphere_center
    ):
        """
        Convert telescope altitude to CORSIKA system (pos_z).

        Parameters
        ----------
        tel_altitude: astropy.Quantity
            Telescope altitude in equivalent units of meter.
        corsika_ob_level: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsika_sphere_center: astropy.Quantity
            CORSIKA sphere center in equivalent units of meter.

        Returns
        -------
        astropy.units.m
            Z-position of a telescope in CORSIKA system.
        """

        return (tel_altitude - corsika_obs_level + corsika_sphere_center).to(u.m)

    @staticmethod
    @u.quantity_input(tel_corsika_z=u.m, corsika_obs_level=u.m, corsika_sphere_center=u.m)
    def convert_telescope_altitude_from_corsika_system(
        tel_corsika_z, corsika_obs_level=None, corsika_sphere_center=None
    ):
        """
        Convert Corsika (pos_z) to altitude.

        Parameters
        ----------
        tel_corsika_z: astropy.Quantity
            Telescope z-position in CORSIKA system in equivalent units of meter.
        corsika_ob_level: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsika_sphere_center: astropy.Quantity
            CORSIKA sphere center in equivalent units of meter.

        Returns
        -------
        astropy.units.m
            Telescope altitude (above sea level)
        """
        return tel_corsika_z + corsika_obs_level - corsika_sphere_center

    def convert_all(self, crs_local=None, crs_wgs84=None, crs_utm=None):
        """
        Perform conversions and fill coordinate variables.

        Parameters
        ----------
        crs_local: pyproj.crs.crs.CRS
            Pyproj CRS of the local coordinate system.
        crs_wgs84: pyproj.crs.crs.CRS
            Pyproj CRS of the mercator coordinate system.
        crs_utm: pyproj.crs.crs.CRS
            Pyproj CRS of the utm coordinate system.

        """

        self._set_coordinate_system("ground", crs_local)
        self._set_coordinate_system("utm", crs_utm)
        self._set_coordinate_system("mercator", crs_wgs84)

        _crs_from_name, _crs_from = self._get_reference_system_from()
        if _crs_from is None:
            return

        for _crs_to_name, _crs_to in self.crs.items():
            if _crs_to_name == _crs_from_name:
                continue
            if not self.has_coordinates(_crs_to_name) and _crs_to["crs"] is not None:
                _x, _y = self._convert(
                    crs_from=_crs_from["crs"],
                    crs_to=_crs_to["crs"],
                    xx=_crs_from["xx"]["value"],
                    yy=_crs_from["yy"]["value"],
                )
                self.set_coordinates(
                    _crs_to_name, _x, _y, _crs_from["zz"]["value"] * _crs_from["zz"]["unit"]
                )

    @staticmethod
    def _default_coordinate_system_definition():
        """
        Definition of coordinate system including axes and default axes units. Follows convention\
        from pyproj for x and y coordinates.

        Returns
        -------
        dict
           coordinate system definition

        """

        return {
            "ground": {
                "crs": None,
                "xx": {"name": "position_x", "value": np.nan, "unit": u.Unit("m")},
                "yy": {"name": "position_y", "value": np.nan, "unit": u.Unit("m")},
                "zz": {"name": "altitude", "value": np.nan, "unit": u.Unit("m")},
            },
            "mercator": {
                "crs": None,
                "xx": {"name": "latitude", "value": np.nan, "unit": u.Unit("deg")},
                "yy": {"name": "longitude", "value": np.nan, "unit": u.Unit("deg")},
                "zz": {"name": "altitude", "value": np.nan, "unit": u.Unit("m")},
            },
            "utm": {
                "crs": None,
                "xx": {"name": "utm_east", "value": np.nan, "unit": u.Unit("m")},
                "yy": {"name": "utm_north", "value": np.nan, "unit": u.Unit("m")},
                "zz": {"name": "altitude", "value": np.nan, "unit": u.Unit("m")},
            },
        }
