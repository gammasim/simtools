import logging

import astropy.units as u
import numpy as np
import pyproj


class InvalidCoordSystem(Exception):
    pass


class MissingInputForConvertion(Exception):
    pass


class TelescopePosition:
    """
    Store and perform coordinate transformations for an array element position.

    The definition of x_coord and y_coord in this class depend on the \
    coordinate system (e.g., (x_coord, y_coord) == (UTM_east, UTM_north)). \
    Altitude describes always the element height above sea level.

    Attributes
    ----------
    name: str
        Name of the array element (e.g L-01, S-05, ...).

    Methods
    -------
    getCoordinates(self, crs_name)
        Get spatial coordinates (x, y)
    setCoordinates(self, posX, posY, posZ):
        Set spatial coordinates (x, y) and altitude of element
    hasCoordinates(crs_name, crs_check)
        Return True if tel
    getAltitude(self)
        Get altitude.
    setAltitude(self, altitude)
        Set altitude.
    hasAltitude(self, crs_name):
        Return True if tel has altitude.
    convertTelescopeAltitudeToCorsikaSystem(telAltitude, corsikaObsLevel, corsikaSphereCenter)
        Convert telescope altitude to CORSIKA system (posZ).
    convertTelescopeAltitudeFromCorsikaSystem(telCorsikaZ, corsikaObsLevel, corsikaSphereCenter):
        Convert Corsika (posZ) to altitude.
    convertAll(crsLocal, wgs84, crsUtm)
        Perform conversions and fill coordinate variables.
    """

    def __init__(self, name=None):
        """
        TelescopePosition init.

        Parameters
        ----------
        name: str
            Name of the telescope (e.g L-01, S-05, ...)
        """

        self._logger = logging.getLogger(__name__)

        self.name = name
        self.asset_code = None
        self.sequence_number = None
        self.geo_code = None
        self.crs = self._default_coordinate_system_definition()

    def __str__(self):
        telstr = self.name
        if self.hasCoordinates("corsika"):
            telstr += "\t CORSIKA x(->North): {0:0.2f} y(->West): {1:0.2f}".format(
                self.crs["corsika"]["xx"]["value"], self.crs["corsika"]["yy"]["value"]
            )
        if self.hasCoordinates("utm"):
            telstr += "\t UTM East: {0:0.2f} UTM North: {1:0.2f}".format(
                self.crs["utm"]["xx"]["value"], self.crs["utm"]["yy"]["value"]
            )
        if self.hasCoordinates("mercator"):
            telstr += "\t Longitude: {0:0.5f} Latitude: {1:0.5f}".format(
                self.crs["mercator"]["xx"]["value"], self.crs["mercator"]["yy"]["value"]
            )
        for _crs_name in self.crs:
            if self.hasAltitude(_crs_name):
                telstr += "\t Alt: {:0.2f}".format(self.crs[_crs_name]["zz"]["value"])
                break

        return telstr

    def printCompactFormat(self, crs_name, print_header=False):
        """
        Print array element coordinates in compact format.

        Parameters
        ----------
        crs_name: str
            name of coordinate system

        Raises
        ------
        InvalidCoordSystem
           if coordinate system is not defined

        """
        try:
            telstr = "{0} {1:10.2f} {2:10.2f} {3:10.2f}".format(
                self.name,
                self.crs[crs_name]["xx"]["value"],
                self.crs[crs_name]["yy"]["value"],
                self.crs[crs_name]["zz"]["value"],
            )
            headerstr = "{0} {1} {2} {3}".format(
                "telescope_name",
                self.crs[crs_name]["xx"]["name"],
                self.crs[crs_name]["yy"]["name"],
                self.crs[crs_name]["zz"]["name"],
            )

            if self.geo_code is not None:
                telstr += "  {0}".format(self.geo_code)
                headerstr += "  geo_code"
            if print_header:
                print(headerstr)
            print(telstr)
        except KeyError:
            self._logger.error("Invalid coordinate system ({})".format(crs_name))
            raise InvalidCoordSystem

    def getCoordinates(self, crs_name):
        """
        Get coordinates in a given coordinate system

        Attributes
        ----------
        crs_name: str
            name of coordinate system

        Returns
        -------
        x, y, z coordinate including corresponding unit

        Raises
        ------
        InvalidCoordSystem
           if coordinate system is not defined

        """

        try:
            return (
                self.crs[crs_name]["xx"]["value"] * self.crs[crs_name]["xx"]["unit"],
                self.crs[crs_name]["yy"]["value"] * self.crs[crs_name]["yy"]["unit"],
                self.crs[crs_name]["zz"]["value"] * self.crs[crs_name]["zz"]["unit"],
            )
        except KeyError:
            self._logger.error("Invalid coordinate system ({})".format(crs_name))
            raise InvalidCoordSystem

    def _getCoordinateValue(self, value, unit):
        """
        Return a value of a coordinate variable
        i) converted to the given unit, if input value has a unit
        ii) multiplied by the default unit, if input value has no unit assigned

        """

        if isinstance(value, u.Quantity):
            try:
                return value.to(unit).value
            except u.UnitsError:
                self._logger.error("Invalid unit given ({}) for value: {})".format(unit, value))
                raise

        return value

    def setCoordinates(self, crs_name, xx, yy, zz=None):
        """
        Set coordinates of an array element.

        Attributes
        ----------
        crs_name: str
            Name of coordinate system
        xx: float
            x-coordinate
        yy: float
            y-coordinate
        zz: float
            z-coordinate (altitude)

        Raises
        ------
        InvalidCoordSystem
            If coordinate system is not known

        """

        try:
            self.crs[crs_name]["xx"]["value"] = self._getCoordinateValue(
                xx, self.crs[crs_name]["xx"]["unit"]
            )
            self.crs[crs_name]["yy"]["value"] = self._getCoordinateValue(
                yy, self.crs[crs_name]["yy"]["unit"]
            )
            if zz is not None:
                self.crs[crs_name]["zz"]["value"] = self._getCoordinateValue(
                    zz, self.crs[crs_name]["zz"]["unit"]
                )
        except KeyError:
            self._logger.error("Invalid coordinate system ({})".format(crs_name))
            raise InvalidCoordSystem

    def getAltitude(self):
        """ "
        Get altitude of an array element

        Returns
        -------
        astropy.Quantity
            telescope altitidue

        """
        for _crs in self.crs.values():
            if _crs["zz"]["value"]:
                return _crs["zz"]["value"] * u.Unit(_crs["zz"]["unit"])

    def setAltitude(self, telAltitude):
        """
        Set altitude of an array element.
        Assume that all coordinate system have same altitude definition,
        meaning altitude is set for all systems here.

        Attributes
        ----------
        telAltitude: astropy.Quantity

        """

        for _crs in self.crs.values():
            _crs["zz"]["value"] = self._getCoordinateValue(telAltitude, _crs["zz"]["unit"])

    def _convert(self, crs_from, crs_to, xx, yy):
        """
        Coordinate transformation of telescope positions.
        Returns np.nan for failed transformations (and not inf, as pyproj does)

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
        Return coordinate system and coordinatee for a fully defined system.
        The firstfully defined system from self.crs is returned.

        Returns
        -------
        string
            Name of coordinate system
        pyproj.crs.crs.CRS
            Project of coordinate system

        """

        for _crs_name, _crs in self.crs.items():
            if self.hasCoordinates(_crs_name, crs_check=True):
                return _crs_name, _crs
        return None, None

    def hasCoordinates(self, crs_name, crs_check=False):
        """
        Check if coordinates are set for a given coordinate system.

        Attributes
        ----------
        crs_name: str
            Name of coordinate system
        crs_check: bool
            Check that projection system is defined

        Returns
        -------
        bool
            True if coordinate system is defined

        Raises
        ------
        InvalidCoordSystem
            If coordinate system is not known
        """
        try:
            if not self.crs[crs_name]["crs"] and crs_check:
                return False
        except KeyError:
            self._logger.error("Invalid coordinate system ({})".format(crs_name))
            raise InvalidCoordSystem

        return (
            self.crs[crs_name]["xx"]["value"] is not np.nan
            and self.crs[crs_name]["yy"]["value"] is not np.nan
            and self.crs[crs_name]["xx"]["value"] is not None
            and self.crs[crs_name]["yy"]["value"] is not None
        )

    def hasAltitude(self, crs_name=None):
        """
        Return True if array element has altitude defined.

        Attributes
        ----------
        crs_name: str
            Name of coordinate system to be checked for altitude.
            If none: check if altitude is define for any system.

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
                if self.hasAltitude(_crs_name):
                    return True
            return False

        try:
            return (
                self.crs[crs_name]["zz"]["value"] is not np.nan
                and self.crs[crs_name]["zz"]["value"] is not None
            )
        except KeyError:
            self._logger.error("Invalid coordinate system ({})".format(crs_name))
            raise InvalidCoordSystem

    def _setCoordinateSystem(self, crs_name, crs_system):
        """
        Set a coordinate system with a given name.

        Attributes
        ----------
        crs_name: str
            Name of coordinate system
        crs_system: pyproj.crs.crs.CRS
             Project of coordinate system

        Raises
        ------
        InvalidCoordSystem
            If coordinate system is not known

        """
        try:
            self.crs[crs_name]["crs"] = crs_system
        except KeyError:
            self._logger.error("Invalid coordinate system ({})".format(crs_name))
            raise InvalidCoordSystem

    @staticmethod
    @u.quantity_input(telAltitude=u.m, corsikaObsLevel=u.m, corsikaSphereCenter=u.m)
    def convertTelescopeAltitudeToCorsikaSystem(telAltitude, corsikaObsLevel, corsikaSphereCenter):
        """
        Convert telescope altitude to CORSIKA system (posZ).

        Attributes
        ----------
        telAltitude: astropy.Quantity
            Telescope altitude in equivalent units of meter.
        corsikaObLevel: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsikaSphereCenter: astropy.Quantity
            CORSIKA sphere center in equivalent units of meter.

        Returns
        -------
        Z-position of a telescope in CORSIKA system
        """

        return (telAltitude - corsikaObsLevel + corsikaSphereCenter).to(u.m)

    @staticmethod
    @u.quantity_input(corsikaObsLevel=u.m, corsikaSphereCenter=u.m)
    def convertTelescopeAltitudeFromCorsikaSystem(
        telCorsikaZ, corsikaObsLevel=None, corsikaSphereCenter=None
    ):
        """
        Convert Corsika (posZ) to altitude.

        Attributes
        ----------
        telCorsikaZ: astropy.Quantity
            Telescope z-position in CORSIKA system in equivalent units of meter.
        corsikaObLevel: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsikaSphereCenter: astropy.Quantity
            CORSIKA sphere center in equivalent units of meter.

        Returns
        -------
        Telescope altitude (above sea level)
        """

        return telCorsikaZ + corsikaObsLevel - corsikaSphereCenter

    def convertAll(self, crsLocal=None, crsWgs84=None, crsUtm=None):
        """
        Perform conversions and fill coordinate variables.

        Parameters
        ----------
        crsUtm: pyproj.crs.crs.CRS
            Pyproj CRS of the utm coordinate system
        crsLocal: pyproj.crs.crs.CRS
            Pyproj CRS of the local coordinate system
        crsWgs84: pyproj.crs.crs.CRS
            Pyproj CRS of the mercator coordinate system
        """

        self._setCoordinateSystem("corsika", crsLocal)
        self._setCoordinateSystem("utm", crsUtm)
        self._setCoordinateSystem("mercator", crsWgs84)

        _crs_from_name, _crs_from = self._get_reference_system_from()

        try:
            for _crs_to_name, _crs_to in self.crs.items():
                if _crs_to_name == _crs_from_name:
                    continue
                if not self.hasCoordinates(_crs_to_name) and _crs_to["crs"] is not None:
                    _x, _y = self._convert(
                        crs_from=_crs_from["crs"],
                        crs_to=_crs_to["crs"],
                        xx=_crs_from["xx"]["value"],
                        yy=_crs_from["yy"]["value"],
                    )
                    self.setCoordinates(
                        _crs_to_name, _x, _y, _crs_from["zz"]["value"] * _crs_from["zz"]["unit"]
                    )
        except (InvalidCoordSystem, TypeError):
            self._logger.error("No reference coordinate system defined")
            raise MissingInputForConvertion

    @staticmethod
    def _default_coordinate_system_definition():
        """
        Definition of coordinate system including axes and default axes units.
        Follows convention from pyproj for x and y coordinates.

        Returns
        -------
        dict
           coordinate system defintion

        """

        return {
            "corsika": {
                "crs": None,
                "xx": {"name": "posX", "value": np.nan, "unit": u.Unit("m")},
                "yy": {"name": "posY", "value": np.nan, "unit": u.Unit("m")},
                "zz": {"name": "posZ", "value": np.nan, "unit": u.Unit("m")},
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
