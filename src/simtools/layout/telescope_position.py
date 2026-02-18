"""Telescope positions and coordinate transformations."""

import logging

import astropy.units as u
import numpy as np
import pyproj


class InvalidCoordSystemErrorError(Exception):
    """Exception for invalid coordinate system."""


class TelescopePosition:
    """
    Store a telescope position and perform coordinate transformations.

    The definition of x_coord and y_coord in this class depend on the \
    coordinate system (e.g., (x_coord, y_coord) == (UTM_east, UTM_north)). \
    Altitude describes always the element height above sea level, position_z
    the height above a reference altitude (e.g., CORSIKA observation level).

    Each array element stores its position in multiple coordinate systems
    with the TelescopePosition.crs dictionary. The definition of coordinate
    systems and corresponding axes is as follows::

        {
            "ground": {
                "crs": <pyproj.CRS>,
                "xx": {"name": "position_x", "value": float, "unit": m},
                "yy": {"name": "position_y", "value": float, "unit": m},
                "zz": {"name": "position_z", "value": float, "unit": m},
            },
            "utm": {
                "crs": <pyproj.CRS>,
                "xx": {"name": "utm_east", "value": float, "unit": m},
                "yy": {"name": "utm_north", "value": float, "unit": m},
                "zz": {"name": "altitude", "value": float, "unit": m},
            },
            "mercator": {
                "crs": <pyproj.CRS>,
                "xx": {"name": "latitude", "value": float, "unit": deg},
                "yy": {"name": "longitude", "value": float, "unit": deg},
                "zz": {"name": "altitude", "value": float, "unit": m},
            },
            "auxiliary": {
                "telescope_sphere_radius": {...},
                "telescope_axis_height": {...},
            }
        }

    Parameters
    ----------
    name: str
        Name of the telescope (e.g LSTN-01, SSTS-05, ...).

    """

    def __init__(self, name=None):
        """Initialize TelescopePosition."""
        self._logger = logging.getLogger(__name__)

        self.name = name
        self.asset_code = None
        self.sequence_number = None
        self.geo_code = None
        self.crs = self._default_coordinates()

    def __str__(self):
        """Return string representation of TelescopePosition."""
        tel_str = self.name
        if self.has_coordinates("ground"):
            tel_str += (
                f"\t Ground x(->North): {self.crs['ground']['xx']['value']:0.2f} "
                f"y(->West): {self.crs['ground']['yy']['value']:0.2f}"
            )
        if self.has_coordinates("utm"):
            tel_str += (
                f"\t UTM East: {self.crs['utm']['xx']['value']:0.2f} "
                f"UTM North: {self.crs['utm']['yy']['value']:0.2f}"
            )
        if self.has_coordinates("mercator"):
            tel_str += (
                f"\t Longitude: {self.crs['mercator']['xx']['value']:0.5f} "
                f"Latitude: {self.crs['mercator']['yy']['value']:0.5f}"
            )
        for _crs_name, _crs_now in self.crs.items():
            if self.is_coordinate_system(_crs_name) and self.has_altitude(_crs_name):
                tel_str += f"\t Alt: {_crs_now['zz']['value']:0.2f}"
                break

        return tel_str

    def print_compact_format(
        self,
        crs_name,
        print_header=False,
        corsika_observation_level=None,
    ):
        """
        Print array element coordinates in compact format.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for printing.
        print_header: bool
            Print table header.
        corsika_observation_level: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.

        Raises
        ------
        InvalidCoordSystemErrorError
           if coordinate system is not defined.
        """
        try:
            _zz = self.crs[crs_name]["zz"]["value"]
            _zz_header = self.crs[crs_name]["zz"]["name"]
            if crs_name == "ground" and corsika_observation_level is not None:
                _zz = (
                    self.convert_telescope_altitude_to_corsika_system(
                        _zz * u.Unit(self.crs[crs_name]["zz"]["unit"]),
                        corsika_observation_level,
                        self.crs["auxiliary"]["telescope_axis_height"]["value"]
                        * u.Unit(self.crs["auxiliary"]["telescope_axis_height"]["unit"]),
                    )
                ).value
                _zz_header = "position_z"

            if crs_name == "mercator":
                tel_str = (
                    f"{self.name} {self.crs[crs_name]['xx']['value']:10.8f} "
                    f"{self.crs[crs_name]['yy']['value']:10.8f} {_zz:10.2f}"
                )
            else:
                tel_str = (
                    f"{self.name} {self.crs[crs_name]['xx']['value']:10.2f} "
                    f"{self.crs[crs_name]['yy']['value']:10.2f} {_zz:10.2f}"
                )
            header_str = (
                f"telescope_name {self.crs[crs_name]['xx']['name']} "
                f"{self.crs[crs_name]['yy']['name']} {_zz_header}"
            )

            if self.geo_code is not None:
                tel_str += f"  {self.geo_code}"
                header_str += "  geo_code"
            if print_header:
                print(header_str)
            print(tel_str)
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystemErrorError from e

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
        InvalidCoordSystemErrorError
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
                raise InvalidCoordSystemErrorError from e
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
                raise InvalidCoordSystemErrorError from e

    def _get_coordinate_value(self, value, unit):
        """
        Return a value of a coordinate variable.

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
        InvalidCoordSystemErrorError
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
            raise InvalidCoordSystemErrorError from e

    def get_altitude(self):
        """
        Get altitude of an array element.

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
        Set altitude of an array element.

        Assume that all coordinate system have same altitude definition, meaning altitude
        is set for all systems here.

        Attributes
        ----------
        tel_altitude: astropy.Quantity
        """
        for _crs in self.crs.values():
            try:
                _crs["zz"]["value"] = self._get_coordinate_value(tel_altitude, _crs["zz"]["unit"])
            except KeyError:
                pass

    def _convert(self, crs_from, crs_to, xx, yy):
        """
        Coordinate transformation of telescope positions.

        Returns np.nan for failed transformations (and not inf, as pyproj does).

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
        Return coordinate system and coordinates for a fully defined system.

        The first fully defined system from self.crs is returned.

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

    def is_coordinate_system(self, crs_name):
        """
        Check if crs_name describes a coordinate system or auxiliary information.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system.

        Returns
        -------
        bool
            True if coordinate system is defined.

        Raises
        ------
        InvalidCoordSystemErrorError
            If coordinate system is not known.


        """
        try:
            return "crs" in self.crs[crs_name]
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystemErrorError from e

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
        InvalidCoordSystemErrorError
            If coordinate system is not known.
        """
        if not self.is_coordinate_system(crs_name):
            return False
        try:
            if not self.crs[crs_name]["crs"] and crs_check:
                return False
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystemErrorError from e

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
        InvalidCoordSystemErrorError
            If coordinate system is not known

        """
        if crs_name is None:
            for _crs_name in self.crs:
                if self.has_altitude(_crs_name):
                    return True
            return False

        if not self.is_coordinate_system(crs_name):
            return False

        try:
            return (
                self.crs[crs_name]["zz"]["value"] is not np.nan
                and self.crs[crs_name]["zz"]["value"] is not None
            )
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystemErrorError from e

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
        InvalidCoordSystemErrorError
            If coordinate system is not known.

        """
        try:
            self.crs[crs_name]["crs"] = crs_system
        except KeyError as e:
            self._logger.error(f"Invalid coordinate system ({crs_name})")
            raise InvalidCoordSystemErrorError from e

    @staticmethod
    @u.quantity_input(tel_altitude=u.m, corsika_observation_level=u.m, telescope_axis_height=u.m)
    def convert_telescope_altitude_to_corsika_system(
        tel_altitude, corsika_observation_level, telescope_axis_height
    ):
        """
        Convert telescope altitude to CORSIKA system (pos_z).

        Parameters
        ----------
        tel_altitude: astropy.Quantity
            Telescope altitude in equivalent units of meter.
        corsika_ob_level: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        telescope_axis_height: astropy.Quantity
            Height of telescope elevation axis above ground level in equivalent units of meter.

        Returns
        -------
        astropy.units.m
            Z-position of a telescope in CORSIKA system.
        """
        return (tel_altitude - corsika_observation_level + telescope_axis_height).to(u.m)

    @staticmethod
    @u.quantity_input(tel_corsika_z=u.m, corsika_observation_level=u.m, telescope_axis_height=u.m)
    def convert_telescope_altitude_from_corsika_system(
        tel_corsika_z, corsika_observation_level=None, telescope_axis_height=None
    ):
        """
        Convert Corsika (pos_z) to altitude.

        Parameters
        ----------
        tel_corsika_z: astropy.Quantity
            Telescope z-position in CORSIKA system in equivalent units of meter.
        corsika_observation_level: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        telescope_axis_height: astropy.Quantity
            Height of telescope elevation axis above ground level in equivalent units of meter.

        Returns
        -------
        astropy.units.m
            Telescope altitude (above sea level)
        """
        return tel_corsika_z + corsika_observation_level - telescope_axis_height

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
            if _crs_to_name == _crs_from_name or not self.is_coordinate_system(_crs_to_name):
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

    def get_axis_height(self):
        """
        Get telescope axis height.

        Returns
        -------
        astropy.quantity
            Telescope axis height.

        """
        return self.crs["auxiliary"]["telescope_axis_height"]["value"] * u.Unit(
            self.crs["auxiliary"]["telescope_axis_height"]["unit"]
        )

    def get_sphere_radius(self):
        """
        Get telescope sphere radius.

        Returns
        -------
        astropy.quantity
            Telescope sphere radius

        """
        return self.crs["auxiliary"]["telescope_sphere_radius"]["value"] * u.Unit(
            self.crs["auxiliary"]["telescope_sphere_radius"]["unit"]
        )

    def set_auxiliary_parameter(self, parameter_name, quantity):
        """
        Set auxiliary parameter.

        Parameters
        ----------
        parameter_name: str
            Name of parameter.
        quantity: astropy.units.Quantity
            Quantity of parameter.

        """
        self.crs["auxiliary"][parameter_name]["value"] = quantity.value
        self.crs["auxiliary"][parameter_name]["unit"] = quantity.unit

    @staticmethod
    def _default_coordinates():
        """
        Coordinate definition for a telescope position.

        Includes all coordinate systems and auxiliary information. Includes axes and
        default axes units. Naming convention follows pyproj for x and y coordinates.
        Includes auxiliary telescope data required for CORSIKA.

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
                "zz": {"name": "position_z", "value": np.nan, "unit": u.Unit("m")},
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
            "auxiliary": {
                "telescope_sphere_radius": {"value": np.nan, "unit": u.Unit("m")},
                "telescope_axis_height": {"value": np.nan, "unit": u.Unit("m")},
            },
        }
