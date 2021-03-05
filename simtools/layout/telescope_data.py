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
    '''

    ALL_INPUTS = {
        'posX': {'default': None, 'unit': u.m},
        'posY': {'default': None, 'unit': u.m},
        'posZ': {'default': None, 'unit': u.m},
        'longitude': {'default': None, 'unit': u.deg},
        'latitude': {'default': None, 'unit': u.deg},
        'utmEast': {'default': None, 'unit': u.m},
        'utmNorth': {'default': None, 'unit': u.m},
        'altitude': {'default': None, 'unit': u.m}
    }

    def __init__(
        self,
        name=None,
        prodId=dict(),
        logger=__name__,
        **kwargs
    ):
        '''
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
        return self._posX * u.m, self._posY * u.m

    def getMercatorCoordinates(self):
        return self._latitude * u.deg, self._longitude * u.deg

    def getUtmCoordinates(self):
        return self._utmNorth * u.deg, self._utmEast * u.deg

    def getTelescopeType(self, name):
        """
        guestimate telescope type from telescope name
        """
        if name[0] == 'L':
            return 'LST'
        elif name[0] == 'M':
            return 'MST'
        elif name[0] == 'S':
            return 'SST'
        return None

    def printTelescope(self):
        """
        print telescope name and positions
        """
        print('%s' % self.name)
        if self._posX is not None and self._posY is not None:
            print('\t CORSIKA x(->North): {0:0.2f} y(->West): {1:0.2f} z: {2:0.2f}'.format(
                self._posX,
                self._posY,
                self._posZ
            ))
        if self._utmEast is not None and self._utmNorth is not None:
            print('\t UTM East: {0:0.2f} UTM North: {1:0.2f} Alt: {2:0.2f}'.format(
                self._utmEast,
                self._utmNorth,
                self._altitude)
            )
        if self._longitude is not None and self._latitude is not None:
            print('\t Longitude: {0:0.5f} Latitude: {1:0.5f} Alt: {2:0.2f}'.format(
                self._longitude,
                self._latitude,
                self._altitude
            ))
        if len(self._prodId) > 0:
            print('\t', self._prodId)

    def printShortTelescopeList(self):
        """
        print short list
        """
        print("{0} {1:10.2f} {2:10.2f}".format(self.name, self._posX, self._posY))

    def convertLocalToMercator(self, crsLocal, wgs84):
        """
        convert telescope position from local to mercator
        """

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

        # require valid coordinate systems
        if (
            not isinstance(crsLocal, pyproj.crs.crs.CRS)
            or not isinstance(wgs84, pyproj.crs.crs.CRS)
        ):
            msg = 'crsLocal and/or wgs84 is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        # calculate lon/lat of a telescope
        self._latitude, self._longitude = pyproj.transform(
            crsLocal,
            wgs84,
            self._posX,
            self._posY
        )
        return

    def convertLocalToUtm(self, crsLocal, crsUtm):
        """
        convert telescope position from local to utm
        """
        if self._utmEast is not None and self._utmNorth is not None:
            self._logger.debug(
                'utm east and utm north are already set'
                ' - aborting convertion from local to UTM'
            )
            return

        # require valid coordinate systems
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

        # calculate utms of a telescope
        self._utmEast, self._utmNorth = pyproj.transform(
            crsLocal,
            crsUtm,
            self._posX,
            self._posY
        )
        return

    def convertUtmToMercator(self, crsUtm, wgs84):
        """
        convert telescope position from utm to mercator
        """

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

        # require valid coordinate systems
        if (
            not isinstance(crsUtm, pyproj.crs.crs.CRS)
            or not isinstance(wgs84, pyproj.crs.crs.CRS)
        ):
            msg = 'crsUtm and/or wgs84 is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        self._latitude, self._longitude = pyproj.transform(
            crsUtm,
            wgs84,
            self._utmEast,
            self._utmNorth
        )
        return

    def convertUtmToLocal(self, crsUtm, crsLocal):
        """
        convert telescope position from utm to mercator
        """

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

        # require valid coordinate systems
        if (
            not isinstance(crsUtm, pyproj.crs.crs.CRS)
            or not isinstance(crsLocal, pyproj.crs.crs.CRS)
        ):
            msg = 'crsUtm and/or crsLocal is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        self._posX, self._posY = pyproj.transform(
            crsUtm,
            crsLocal,
            self._utmEast,
            self._utmNorth
        )
        return

    @u.quantity_input(corsikaObsLevel=u.m)
    def convertAslToCorsika(self, corsikaObsLevel, corsikaSphereCenter):
        """
        convert telescope altitude to corsika
        """

        # require valid center altitude values

        if self._posZ is None and self._altitude is not None:
            self._posZ = self._altitude - corsikaObsLevel.to(u.m).value
            self._posZ += corsikaSphereCenter[self.getTelescopeType(self.name)].to(u.m).value
            return
        else:
            self._logger.debug(
                'Could not convert from asl to corsika because posZ is already '
                'set or because altitude is not set.'
            )
            return

    @u.quantity_input(corsikaObsLevel=u.m)
    def convertCorsikaToAsl(self, corsikaObsLevel, corsikaSphereCenter):
        """
        convert corsika z to altitude
        """

        if self._posZ is not None and self._altitude is None:
            self._altitude = corsikaObsLevel.to(u.m).value + self._posZ
            self._altitude -= corsikaSphereCenter[self.getTelescopeType(self.name)].to(u.m).value
            return
        else:
            self._logger.warning(
                'Could not convert from corsika to asl because posZ is not set '
                'or because altitude is already set.'
            )
            return

    @u.quantity_input(corsikaObsLevel=u.m)
    def convertAll(
        self,
        crsLocal,
        wgs84,
        crsUtm,
        corsikaObsLevel,
        corsikaSphereCenter
    ):
        """
        calculate telescope positions in missing coordinate
        systems
        """

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

        hasCorsika = self._posZ is not None
        hasAsl = self._altitude is not None

        if hasCorsika and not hasAsl:
            self.convertCorsikaToAsl(corsikaObsLevel, corsikaSphereCenter)
        elif hasAsl and not hasCorsika:
            self.convertAslToCorsika(corsikaObsLevel, corsikaSphereCenter)
        else:
            # Nothing to be converted
            pass
