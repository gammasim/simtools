import logging
import astropy.units as u

import pyproj

from simtools.util.general import collectArguments


class InvalidCoordSystem(Exception):
    pass


class MissingInputForConvertion(Exception):
    pass


class TelescopeData:
    '''
    Store and do coordenate transformations with single telescope positions

    Attributes
    ----------
    name: str
        Name of the telescope (e.g L-01, S-05, ...).

    Methods
    -------
    getLocalCoordinates()
        Get the X and Y coordinates.
    getMercatorCoordinates()
        Get the latitude and longitude.
    getUtmCoordinates()
        Get utm north and east.
    convertLocalToMercator(crsLocal, wgs84)
        Convert telescope position from local to mercator.
    convertLocalToUtm(crsLocal, crsUtm)
        Convert telescope position from local to UTM.
    convertUtmToMercator(crsUtm, wgs84)
        Convert telescope position from UTM to mercator.
    convertUtmToLocal(crsUtm, crsLocal)
        Convert telescope position from UTM to local.
    convertAslToCorsika(corsikaObsLevel, corsikaSphereCenter)
        Convert telescope altitude to corsika (posZ).
    convertCorsikaToAsl(corsikaObsLevel, corsikaSphereCenter):
        Convert corsika (posZ) to altitude.
    convertAll(crsLocal, wgs84, crsUtm, corsikaObsLevel, corsikaSphereCenter)
        Perform all the necessary convertions in order to fill all the coordinate variables.
    '''

    ALL_INPUTS = {
        'posX': {'default': None, 'unit': u.m},
        'posY': {'default': None, 'unit': u.m},
        'posZ': {'default': None, 'unit': u.m},
        'longitude': {'default': None, 'unit': u.deg},
        'latitude': {'default': None, 'unit': u.deg},
        'utmEast': {'default': None, 'unit': u.m},
        'utmNorth': {'default': None, 'unit': u.m},
        'altitude': {'default': None, 'unit': u.m},
        'corsikaSphereCenter': {'default': None, 'unit': u.m},
        'corsikaSphereRadius': {'default': None, 'unit': u.m}
    }

    def __init__(
        self,
        name=None,
        prodId=dict(),
        logger=__name__,
        **kwargs
    ):
        '''
        TelescopeData init.

        Parameters
        ----------
        name: str
            Name of the telescope (e.g L-01, S-05, ...)
        prodId: dict
            ...
        logger: str
            Logger name to use in this instance
        **kwargs:
            Physical parameters with units (if applicable). Options: posX, posY, posZ,
            longitude, latitude, utmsEast, utmNorth, altitude, corsikaSphereRadius,
            corsikaSphereCenter
        '''

        self._logger = logging.getLogger(logger)
        self._logger.debug('Init TelescopeData')

        self.name = name
        self._prodId = prodId

        # Collecting arguments
        collectArguments(
            self,
            args=[*self.ALL_INPUTS],
            allInputs=self.ALL_INPUTS,
            **kwargs
        )

    def getLocalCoordinates(self):
        '''
        Get the X and Y coordinates.

        Returns
        -------
        (posX [u.m], posY [u.m])
        '''
        return self._posX * u.m, self._posY * u.m

    def getMercatorCoordinates(self):
        '''
        Get the latitude and longitude.

        Returns
        -------
        (latitude [u.deg], longitude [u.deg])
        '''
        return self._latitude * u.deg, self._longitude * u.deg

    def getUtmCoordinates(self):
        '''
        Get utm north and east.

        Returns
        -------
        (utmNorth [u.deg], utmEast [u.deg])
        '''
        return self._utmNorth * u.deg, self._utmEast * u.deg

    def __repr__(self):
        telstr = self.name + '\n'
        if self._posX is not None and self._posY is not None:
            telstr += '\t CORSIKA x(->North): {0:0.2f} y(->West): {1:0.2f} z: {2:0.2f}'.format(
                self._posX,
                self._posY,
                self._posZ
            )
        if self._utmEast is not None and self._utmNorth is not None:
            telstr += '\t UTM East: {0:0.2f} UTM North: {1:0.2f} Alt: {2:0.2f}'.format(
                self._utmEast,
                self._utmNorth,
                self._altitude
            )
        if self._longitude is not None and self._latitude is not None:
            telstr += '\t Longitude: {0:0.5f} Latitude: {1:0.5f} Alt: {2:0.2f}'.format(
                self._longitude,
                self._latitude,
                self._altitude
            )
        if len(self._prodId) > 0:
            telstr += '\t', self._prodId
        return telstr

    # def printShortTelescopeList(self):
    #     """
    #     print short list
    #     """
    #     print("{0} {1:10.2f} {2:10.2f}".format(self.name, self._posX, self._posY))

    def convertLocalToMercator(self, crsLocal, wgs84):
        '''
        Convert telescope position from local to mercator.

        Parameters
        ----------
        crsLocal: pyproj.crs.crs.CRS
            Pyproj CRS of the local coordinate system
        wgs84: pyproj.crs.crs.CRS
            Pyproj CRS of the mercator coordinate system

        Raises
        ------
        InvalidCoordSystem
            If crsLocal or wgs84 is not an instance of pyproj.crs.crs.CRS
        '''

        if self._altitude is not None and self._longitude is not None:
            self._logger.debug(
                'altitude and longitude are already set'
                ' - aborting convertion from local to mercator'
            )
            return

        if self._posX is None or self._posY is None:
            msg = 'posX and/or posY are not set - impossible to convert from local to mercator'
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        # Require valid coordinate systems
        if (
            not isinstance(crsLocal, pyproj.crs.crs.CRS)
            or not isinstance(wgs84, pyproj.crs.crs.CRS)
        ):
            msg = 'crsLocal and/or wgs84 is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        # Calculate lon/lat of a telescope
        self._latitude, self._longitude = pyproj.transform(
            crsLocal,
            wgs84,
            self._posX,
            self._posY
        )
        return

    def convertLocalToUtm(self, crsLocal, crsUtm):
        '''
        Convert telescope position from local to UTM.

        Parameters
        ----------
        crsLocal: pyproj.crs.crs.CRS
            Pyproj CRS of the local coordinate system
        crsUtm: pyproj.crs.crs.CRS
            Pyproj CRS of the utm coordinate system

        Raises
        ------
        InvalidCoordSystem
            If crsLocal or crsUtm is not an instance of pyproj.crs.crs.CRS
        '''
        if self._utmEast is not None and self._utmNorth is not None:
            self._logger.debug(
                'utm east and utm north are already set'
                ' - aborting convertion from local to UTM'
            )
            return

        # Require valid coordinate systems
        if (
            not isinstance(crsLocal, pyproj.crs.crs.CRS)
            or not isinstance(crsUtm, pyproj.crs.crs.CRS)
        ):
            msg = 'crsLocal and/or crsUtm is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        if self._posX is None or self._posY is None:
            msg = 'posX and/or posY are not set - impossible to convert from local to mercator'
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        # Calculate utms of a telescope
        self._utmEast, self._utmNorth = pyproj.transform(
            crsLocal,
            crsUtm,
            self._posX,
            self._posY
        )
        return

    def convertUtmToMercator(self, crsUtm, wgs84):
        '''
        Convert telescope position from UTM to mercator.

        Parameters
        ----------
        crsUtm: pyproj.crs.crs.CRS
            Pyproj CRS of the utm coordinate system
        wgs84: pyproj.crs.crs.CRS
            Pyproj CRS of the mercator coordinate system

        Raises
        ------
        InvalidCoordSystem
            If wgs84 or crsUtm is not an instance of pyproj.crs.crs.CRS
        '''

        if self._latitude is not None and self._longitude is not None:
            self._logger.debug(
                'altitude and longitude are already set'
                ' - aborting convertion from utm to mercator'
            )
            return

        if self._utmEast is None or self._utmNorth is None:
            msg = (
                'utm east and/or utm north are not set - '
                'impossible to convert from utm to mercator'
            )
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        # Require valid coordinate systems
        if (
            not isinstance(crsUtm, pyproj.crs.crs.CRS)
            or not isinstance(wgs84, pyproj.crs.crs.CRS)
        ):
            msg = 'crsUtm and/or wgs84 is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        # Calculate latitude and longitude
        self._latitude, self._longitude = pyproj.transform(
            crsUtm,
            wgs84,
            self._utmEast,
            self._utmNorth
        )
        return

    def convertUtmToLocal(self, crsUtm, crsLocal):
        '''
        Convert telescope position from UTM to local.

        Parameters
        ----------
        crsUtm: pyproj.crs.crs.CRS
            Pyproj CRS of the utm coordinate system
        crsLocal: pyproj.crs.crs.CRS
            Pyproj CRS of the local coordinate system

        Raises
        ------
        InvalidCoordSystem
            If crsLocal or crsUtm is not an instance of pyproj.crs.crs.CRS
        '''
        if self._posX is not None and self._posY is not None:
            self._logger.debug(
                'latitude and longitude are already set'
                ' - aborting convertion from utm to local'
            )
            return

        if self._utmEast is None or self._utmNorth is None:
            msg = (
                'utm east and/or utm north are not set - '
                'impossible to convert from utm to local'
            )
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        # Require valid coordinate systems
        if (
            not isinstance(crsUtm, pyproj.crs.crs.CRS)
            or not isinstance(crsLocal, pyproj.crs.crs.CRS)
        ):
            msg = 'crsUtm and/or crsLocal is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        # Calculate posX and posY
        self._posX, self._posY = pyproj.transform(
            crsUtm,
            crsLocal,
            self._utmEast,
            self._utmNorth
        )
        return

    @u.quantity_input(corsikaObsLevel=u.m, corsikaSphereCenter=u.m)
    def convertAslToCorsika(self, corsikaObsLevel=-1 * u.m, corsikaSphereCenter=-1 * u.m):
        '''
        Convert telescope altitude to corsika (posZ).
        corsikaObsLevel and/or corsikaSphereCenter can be given as arguments
        in case it has not been set at the initialization.
        If they are given, the internal one will be overwritten.

        Parameters
        ----------
        corsikaObLevel: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsikaSphereCenter: astropy.Quantity
            CORSIKA sphere center in equivalent units of meter.
        '''

        hasPars = self._corsikaObsLevel is not None and self._corsikaSphereCenter is not None
        givenPars = corsikaObsLevel.value > 0 and corsikaSphereCenter.value > 0

        if not hasPars and not givenPars:
            msg = 'Cannot convert to corsika because obs level and/or sphere center were not given'
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        if self._posZ is None and self._altitude is not None:

            if givenPars:
                self._posZ = (
                    self._altitude - corsikaObsLevel.to(u.m).value
                    + corsikaSphereCenter.to(u.m).value
                )
            else:  # hasPars
                self._posZ = (self._altitude - self._corsikaObsLevel + self._corsikaSphereCenter)

            return
        else:
            self._logger.debug(
                'Could not convert from asl to corsika because posZ is already '
                'set or because altitude is not set.'
            )
            return

    @u.quantity_input(corsikaObsLevel=u.m, corsikaSphereCenter=u.m)
    def convertCorsikaToAsl(self, corsikaObsLevel=-1 * u.m, corsikaSphereCenter=-1 * u.m):
        '''
        Convert corsika (posZ) to altitude.
        corsikaObsLevel and/or corsikaSphereCenter can be given as arguments
        in case it has not been set at the initialization.
        If they are given, the internal one will be overwritten.

        Parameters
        ----------
        corsikaObLevel: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsikaSphereCenter: astropy.Quantity
            CORSIKA sphere center in equivalent units of meter.
        '''

        hasPars = self._corsikaObsLevel is not None and self._corsikaSphereCenter is not None
        givenPars = corsikaObsLevel.value > 0 and corsikaSphereCenter.value > 0

        if not hasPars and not givenPars:
            msg = 'Cannot convert to corsika because obs level and/or sphere center were not given'
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        if self._posZ is not None and self._altitude is None:

            if givenPars:
                self._altitude = (
                    corsikaObsLevel.to(u.m).value + self._posZ - corsikaSphereCenter.to(u.m).value
                )
            else:  # hasPars
                self._altitude = (
                    self._corsikaObsLevel + self._posZ - self._corsikaSphereCenter
                )
            return
        else:
            self._logger.warning(
                'Could not convert from corsika to asl because posZ is not set '
                'or because altitude is already set.'
            )
            return

    @u.quantity_input(corsikaObsLevel=u.m, corsikaSphereCenter=u.m)
    def convertAll(
        self,
        crsLocal,
        wgs84,
        crsUtm,
        corsikaObsLevel=-1 * u.m,
        corsikaSphereCenter=-1 * u.m
    ):
        '''
        Perform all the necessary convertions in order to fill all the coordinate variables.

        Parameters
        ----------
        crsUtm: pyproj.crs.crs.CRS
            Pyproj CRS of the utm coordinate system
        crsLocal: pyproj.crs.crs.CRS
            Pyproj CRS of the local coordinate system
        wgs84: pyproj.crs.crs.CRS
            Pyproj CRS of the mercator coordinate system
        corsikaObLevel: astropy.Quantity
            CORSIKA observation level in equivalent units of meter.
        corsikaSphereCenter: dict
            CORSIKA sphere center in equivalent units of meter.
        '''

        # Starting by local <-> UTM <-> Mercator
        hasLocal = self._posX is not None and self._posY is not None
        hasUtm = self._utmEast is not None and self._utmNorth is not None
        hasMercator = self._latitude is not None and self._longitude is not None

        if hasLocal and not hasMercator:
            self.convertLocalToMercator(crsLocal, wgs84)
        if hasLocal and not hasUtm:
            self.convertLocalToUtm(crsLocal, crsUtm)
        if hasUtm and not hasLocal:
            self.convertUtmToLocal(crsUtm, crsLocal)
        if hasUtm and not hasMercator:
            self.convertUtmToMercator(crsUtm, wgs84)

        # Dealing with altitude <-> posZ
        hasCorsika = self._posZ is not None
        hasAsl = self._altitude is not None

        if hasCorsika and not hasAsl:
            self.convertCorsikaToAsl(corsikaObsLevel, corsikaSphereCenter)
        elif hasAsl and not hasCorsika:
            self.convertAslToCorsika(corsikaObsLevel, corsikaSphereCenter)
        else:
            # Nothing to be converted
            pass
    # End of convertAll
