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
        if name[0:1] is 'L':
            return 'LST'
        elif name[0:1] is 'M':
            return 'MST'
        elif name[0:1] is 'S':
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

    def convertLocalToMercator(self, crs_local, wgs84):
        """
        convert telescope position from local to mercator
        """

        if self._altitude is not None and self._longitude is not None:
            self._logger.warning(
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
            not isinstance(crs_local, pyproj.crs.crs.CRS)
            or not isinstance(wgs84, pyproj.crs.crs.CRS)
        ):
            msg = 'crs_local and/or wgs84 is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        # calculate lon/lat of a telescope
        self._latitude, self._longitude = pyproj.transform(
            crs_local,
            wgs84,
            self._posX,
            self._posY
        )
        return

    def convertLocalToUtm(self, crs_local, crs_utm):
        """
        convert telescope position from local to utm
        """
        if self._utmEast is not None and self._utmNorth is not None:
            self._logger.warning(
                'utm east and utm north are already set'
                ' - aborting convertion from local to UTM'
            )
            return

        # require valid coordinate systems
        if (
            not isinstance(crs_local, pyproj.crs.crs.CRS)
            or not isinstance(crs_utm, pyproj.crs.crs.CRS)
        ):
            msg = 'crs_local and/or crs_utm is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        if self._posX is None or self._posY is None:
            msg = 'posX and/or posY are not set - impossible to convert from local to mercator'
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        # calculate utms of a telescope
        self._utmEast, self._utmNorth = pyproj.transform(
            crs_local,
            crs_utm,
            self._posX,
            self._posY
        )
        return

    def convertUtmToMercator(self, crs_utm, wgs84):
        """
        convert telescope position from utm to mercator
        """

        if self._latitude is not None and self._longitude is not None:
            self._logger.warning(
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
            not isinstance(crs_utm, pyproj.crs.crs.CRS)
            or not isinstance(wgs84, pyproj.crs.crs.CRS)
        ):
            msg = 'crs_utm and/or wgs84 is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        self._latitude, self._longitude = pyproj.transform(
            crs_utm,
            wgs84,
            self._utmEast,
            self._utmNorth
        )
        return

    def convertUtmToLocal(self, crs_utm, crs_local):
        """
        convert telescope position from utm to mercator
        """

        if self._posX is not None and self._posY is not None:
            self._logger.warning(
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
            not isinstance(crs_utm, pyproj.crs.crs.CRS)
            or not isinstance(crs_local, pyproj.crs.crs.CRS)
        ):
            msg = 'crs_utm and/or crs_local is not a valid coord system'
            self._logger.error(msg)
            raise InvalidCoordSystem(msg)

        self._posX, self._posY = pyproj.transform(
            crs_utm,
            crs_local,
            self._utmEast,
            self._utmNorth
        )
        return

    # def convert_asl_to_corsika(self, corsika_obslevel, corsika_sphere_center):
    #     """
    #     convert telescope altitude to corsika
    #     """

    #     # require valid center altitude values
    #     if math.isnan(corsika_obslevel.value):
    #         return
    #     if math.isnan(self.z.value) and \
    #         not math.isnan(self.alt.value):
    #         self.z = self.alt-corsika_obslevel
    #         try:
    #             self.z += corsika_sphere_center[self.get_telescope_type(self.name)]
    #         except KeyError:
    #             logging.error('Failed finding corsika sphere center for {} (to corsika)'.format(
    #                 self.get_telescope_type(self.name)))

    # def convert_corsika_to_asl(self,corsika_obslevel, corsika_sphere_center):
    #     """
    #     convert corsika z to altitude
    #     """
    #     if not math.isnan(self.alt.value):
    #         return
    #     self.alt = corsika_obslevel+self.z
    #     try:
    #         self.alt -= corsika_sphere_center[self.get_telescope_type(self.name)]
    #     except KeyError:
    #         logging.error('Failed finding corsika sphere center for {} (to asl)'.format(
    #             self.get_telescope_type(self.name)))

    # def convert(self, crs_local, wgs84, crs_utm,
    #     center_altitude,
    #     corsika_obslevel, corsika_sphere_center):
    #     """
    #     calculate telescope positions in missing coordinate
    #     systems
    #     """
    #     self.convert_local_to_mercator(crs_local, wgs84)
    #     self.convert_local_to_utm(crs_local, crs_utm)
    #     self.convert_utm_to_mercator(crs_utm, wgs84)
    #     self.convert_utm_to_local(crs_utm, crs_local)

    #     self.convert_asl_to_corsika(corsika_obslevel, corsika_sphere_center)
    #     self.convert_corsika_to_asl(corsika_obslevel, corsika_sphere_center)
