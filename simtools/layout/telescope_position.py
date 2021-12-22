import logging
import astropy.units as u

import pyproj

import simtools.util.general as gen
import simtools.io_handler as io


class InvalidCoordSystem(Exception):
    pass


class MissingInputForConvertion(Exception):
    pass


class TelescopePosition:
    '''
    Store and perform coordinate transformations with single telescope positions.

    Configurable parameters:
        posX:
            len: 1
            unit: m
        posY:
            len: 1
            unit: m
        posZ:
            len: 1
            unit: m
        longitude:
            len: 1
            unit: m
        latitude:
            len: 1
            unit: m
        utmEast:
            len: 1
            unit: m
        utmNorth:
            len: 1
            unit: m
        altitude:
            len: 1
            unit: m
        corsikaObsLevel:
            len: 1
            unit: m
        corsikaSphereCenter:
            len: 1
            unit: m
        corsikaSphereRadius:
            len: 1
            unit: m

    Attributes
    ----------
    name: str
        Name of the telescope (e.g L-01, S-05, ...).
    config: namedtuple
        Contains the configurable parameters (zenithAngle).

    Methods
    -------
    getLocalCoordinates(self)
        Get the X and Y coordinates.
    setLocalCoordinates(self, posX, posY, posZ):
        Set the X, Y and Z coordinates.
    hasLocalCoordinates(self):
        Return True if tel has local coordinates.
    getAltitude(self)
        Get altitude.
    setAltitude(self, altitude)
        Set altitude.
    hasAltitude(self):
        Return True if tel has altitude.
    getMercatorCoordinates(self)
        Get the latitude and longitude.
    setMercatorCoordinates(self, latitude, longitude)
        Set the latitude and longitude coordinates.
    hasMercatorCoordinates(self):
        Return True if tel has Mercator coordinates.
    getUtmCoordinates(self)
        Get utm north and east.
    setUtmCoordinates(self, utmEast, utmNorth)
        Set the UTM coordinates.
    hasUtmCoordinates(self):
        Return True if tel has UTM coordinates.
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

    def __init__(
        self,
        name=None,
        prodId=dict(),
        configData=None,
        configFile=None
    ):
        '''
        TelescopePosition init.

        Parameters
        ----------
        name: str
            Name of the telescope (e.g L-01, S-05, ...)
        prodId: dict
            ...
        configData: dict.
            Dict containing the configurable parameters.
        configFile: str or Path
            Path of the yaml file containing the configurable parameters.
        '''

        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init TelescopePosition')

        self.name = name
        self._prodId = prodId

        # Loading configData
        _configDataIn = gen.collectDataFromYamlOrDict(configFile, configData, allowEmpty=True)
        _parameterFile = io.getDataFile('parameters', 'telescope-position_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_configDataIn, _parameters)

        # Making config entries into attributes
        for par, value in zip(self.config._fields, self.config):
            self.__dict__['_' + par] = value

    @classmethod
    def fromKwargs(cls, **kwargs):
        '''
        Builds a TelescopePosition object from kwargs only.
        The configurable parameters can be given as kwargs, instead of using the
        configData or configFile arguments.

        Parameters
        ----------
        kwargs
            Containing the arguments and the configurable parameters.

        Returns
        -------
        Instance of this class.
        '''
        args, configData = gen.separateArgsAndConfigData(
            expectedArgs=['name', 'prodId'],
            **kwargs
        )
        return cls(**args, configData=configData)

    def __repr__(self):
        telstr = self.name
        if self.hasLocalCoordinates():
            telstr += '\t CORSIKA x(->North): {0:0.2f} y(->West): {1:0.2f} z: {2:0.2f}'.format(
                self._posX,
                self._posY,
                self._posZ
            )
        if self.hasUtmCoordinates():
            telstr += '\t UTM East: {0:0.2f} UTM North: {1:0.2f}'.format(
                self._utmEast,
                self._utmNorth
            )
        if self.hasMercatorCoordinates():
            telstr += '\t Longitude: {0:0.5f} Latitude: {1:0.5f}'.format(
                self._longitude,
                self._latitude
            )
        if self.hasAltitude():
            telstr += '\t Alt: {:0.2f}'.format(self._altitude)

        if len(self._prodId) > 0:
            telstr += '\t', self._prodId
        return telstr

    def getTelescopeSize(self):
        # Guessing the tel size from the name
        if self.name[0] == 'L':
            return 'LST'
        elif self.name[0] == 'M':
            return 'MST'
        elif self.name[0] == 'S':
            return 'SST'
        else:
            self._logger.warning('Telescope size could not be guessed from the name')
            return None

    def getLocalCoordinates(self):
        '''
        Get the X, Y and Z coordinates.

        Returns
        -------
        (posX [u.m], posY [u.m], posZ [u.m])
        '''
        return self._posX * u.m, self._posY * u.m, self._posZ * u.m

    @u.quantity_input(posX=u.m, posY=u.m, posZ=u.m)
    def setLocalCoordinates(self, posX, posY, posZ):
        ''' Set the X, Y and Z coordinates. '''
        if None not in [self._posX, self._posY, self._posZ]:
            self._logger.warning('Local coordinates are already set and will be overwritten')

        self._posX = posX.value
        self._posY = posY.value
        self._posZ = posZ.value

    def hasLocalCoordinates(self):
        '''
        Return True if tel has local coordinates.

        Returns
        -------
        bool
        '''
        return self._posX is not None and self._posY is not None and self._posZ is not None

    def getAltitude(self):
        '''
        Get altitude.

        Returns
        -------
        altitude [u.m]
        '''
        return self._altitude * u.m

    @u.quantity_input(altitude=u.m)
    def setAltitude(self, altitude):
        ''' Set altitude. '''
        if None not in [self._altitude]:
            self._logger.warning('Altitude is already set and will be overwritten')

        self._altitude = altitude.value

    def hasAltitude(self):
        '''
        Return True if tel has altitude.

        Returns
        -------
        bool
        '''
        return self._altitude is not None

    def getMercatorCoordinates(self):
        '''
        Get the latitude and longitude.

        Returns
        -------
        (latitude [u.deg], longitude [u.deg])
        '''
        return self._latitude * u.deg, self._longitude * u.deg

    @u.quantity_input(latitude=u.deg, longitude=u.deg)
    def setMercatorCoordinates(self, latitude, longitude):
        ''' Set the latitude and longitude coordinates. '''
        if None not in [self._latitude, self._longitude]:
            self._logger.warning('Mercator coordinates are already set and will be overwritten')

        self._latitude = latitude.value
        self._longitude = longitude.value

    def hasMercatorCoordinates(self):
        '''
        Return True if tel has Mercator coordinates.

        Returns
        -------
        bool
        '''
        return self._latitude is not None and self._longitude is not None

    def getUtmCoordinates(self):
        '''
        Get utm north and east.

        Returns
        -------
        (utmNorth [u.m], utmEast [u.m])
        '''
        return self._utmNorth * u.m, self._utmEast * u.m

    @u.quantity_input(utmEast=u.m, utmNorth=u.m)
    def setUtmCoordinates(self, utmEast, utmNorth):
        ''' Set the UTM coordinates. '''
        if None not in [self._utmEast, self._utmNorth]:
            self._logger.warning('UTM coordinates are already set and will be overwritten')

        self._utmEast = utmEast.value
        self._utmNorth = utmNorth.value

    def hasUtmCoordinates(self):
        '''
        Return True if tel has UTM coordinates.

        Returns
        -------
        bool
        '''
        return self._utmEast is not None and self._utmNorth is not None

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

        if self.hasMercatorCoordinates():
            self._logger.debug(
                'altitude and longitude are already set'
                ' - aborting convertion from local to mercator'
            )
            return

        if not self.hasLocalCoordinates():
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
        if self.hasUtmCoordinates():
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

        if not self.hasLocalCoordinates():
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

        if self.hasMercatorCoordinates():
            self._logger.debug(
                'altitude and longitude are already set'
                ' - aborting convertion from utm to mercator'
            )
            return

        if not self.hasUtmCoordinates():
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
        if self.hasLocalCoordinates():
            self._logger.debug(
                'latitude and longitude are already set'
                ' - aborting convertion from utm to local'
            )
            return

        if not self.hasUtmCoordinates():
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
    def convertAslToCorsika(self, corsikaObsLevel=None, corsikaSphereCenter=None):
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
        givenPars = corsikaObsLevel is not None and corsikaSphereCenter is not None

        if not hasPars and not givenPars:
            msg = 'Cannot convert to corsika because obs level and/or sphere center were not given'
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        if not self.hasLocalCoordinates() and self.hasAltitude():

            if givenPars:
                self._posZ = (
                    self._altitude - corsikaObsLevel.to(u.m).value
                    + corsikaSphereCenter.to(u.m).value
                )
            else:  # hasPars
                self._posZ = self._altitude - self._corsikaObsLevel + self._corsikaSphereCenter

            return
        else:
            self._logger.debug(
                'Could not convert from asl to corsika because posZ is already '
                'set or because altitude is not set.'
            )
            return

    @u.quantity_input(corsikaObsLevel=u.m, corsikaSphereCenter=u.m)
    def convertCorsikaToAsl(self, corsikaObsLevel=None, corsikaSphereCenter=None):
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
        givenPars = corsikaObsLevel is not None and corsikaSphereCenter is not None

        if not hasPars and not givenPars:
            msg = 'Cannot convert to corsika because obs level and/or sphere center were not given'
            self._logger.error(msg)
            raise MissingInputForConvertion(msg)

        if self.hasLocalCoordinates() and not self.hasAltitude():

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
        crsLocal=None,
        wgs84=None,
        crsUtm=None,
        corsikaObsLevel=None,
        corsikaSphereCenter=None
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

        if crsLocal is None:
            self._logger.warning('crsLocal is None - convertions will be impacted')
        if crsUtm is None:
            self._logger.warning('crsUtm is None - convertions will be impacted')
        if wgs84 is None:
            self._logger.warning('wgs84 is None - convertions will be impacted')

        # Starting by local <-> UTM <-> Mercator

        if (
            self.hasLocalCoordinates()
            and not self.hasMercatorCoordinates()
            and crsLocal is not None
        ):
            self.convertLocalToMercator(crsLocal, wgs84)

        if self.hasLocalCoordinates() and not self.hasUtmCoordinates() and crsLocal is not None:
            self.convertLocalToUtm(crsLocal, crsUtm)

        if self.hasUtmCoordinates() and not self.hasLocalCoordinates() and crsUtm is not None:
            self.convertUtmToLocal(crsUtm, crsLocal)

        if self.hasUtmCoordinates() and not self.hasMercatorCoordinates() and crsUtm is not None:
            self.convertUtmToMercator(crsUtm, wgs84)

        # Dealing with altitude <-> posZ
        if corsikaObsLevel is None or corsikaSphereCenter is None:
            self._logger.warning(
                'CORSIKA convertions cannot be done - missing obs level and/or sphere center'
            )
        elif self.hasLocalCoordinates() and not self.hasAltitude():
            self.convertCorsikaToAsl(corsikaObsLevel, corsikaSphereCenter)
        elif self.hasAltitude() and not self.hasLocalCoordinates():
            self.convertAslToCorsika(corsikaObsLevel, corsikaSphereCenter)
        else:
            # Nothing to be converted
            pass
    # End of convertAll
