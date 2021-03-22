import logging

import pyproj
import astropy.units as u
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools.util import names
from simtools.util.general import collectArguments
from simtools.layout.telescope_data import TelescopeData


class InvalidTelescopeListFile(Exception):
    pass


class LayoutArray:
    '''
    Manage telescope positions at the array layout level.

    Attributes
    ----------
    name: str
        Name of the telescope (e.g L-01, S-05, ...).
    label: str
        Instance label.

    Methods
    -------
    fromLayoutArrayName(layoutArrayName, label=None, filesLocation=None, logger=__name__)
        Create a LayoutArray from a layout name (e.g. South-4LST, North-Prod5, ...)
    readTelescopeListFile(telescopeListFile)
        Read list of telescopes from a ecsv file.
    addTelescope(
        telescopeName,
        posX=None,
        posY=None,
        posZ=None,
        longitude=None,
        latitude=None,
        utmEast=None,
        utmNorth=None,
        altitude=None
    )
        Add an individual telescope to the telescope list.
    exportTelescopeList()
        Export a ECSV file with the telescope positions.
    getNumberOfTelescopes()
        Return the number of telescopes in the list.
    getCorsikaInputList()
        Get a string with the piece of text to be added to
        the CORSIKA input file.
    printTelescopeList()
        Print list of telescopes in current layout for inspection.
    convertCoordinates()
        Perform all the possible conversions the coordinates of the tel positions.
    '''

    ALL_INPUTS = {
        'epsg': {'default': None, 'unit': None},
        'centerLongitude': {'default': None, 'unit': u.deg},
        'centerLatitude': {'default': None, 'unit': u.deg},
        'centerNorthing': {'default': None, 'unit': u.m},
        'centerEasting': {'default': None, 'unit': u.m},
        'centerAltitude': {'default': None, 'unit': u.m},
        'corsikaObsLevel': {'default': None, 'unit': u.m},
        'corsikaSphereCenter': {'default': None, 'isDict': True, 'unit': u.m},
        'corsikaSphereRadius': {'default': None, 'isDict': True, 'unit': u.m}
    }

    def __init__(self, label=None, name=None, filesLocation=None, logger=__name__, **kwargs):
        '''
        LayoutArray init.

        Parameters
        ----------
        name: str
            Name of the telescope (e.g L-01, S-05, ...)
        label: str
            Instance label.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        logger: str
            Logger name to use in this instance
        **kwargs:
            Physical parameters with units (if applicable).
            Options:
                epsg
                centerLongitude (u.deg)
                centerLatitude (u.deg)
                centerNorthing (u.m)
                cernterEasting (u.m)
                centerAltitude (u.m)
                corsikaObsLevel (u.m)
                corsikaSphereCenter {(u.m)}
                corsikaSphereRadius {(u.m)}
        '''
        self._logger = logging.getLogger(logger)
        self._logger.debug('Init LayoutArray')

        self.label = label

        self.name = name
        self._telescopeList = []

        # Collecting arguments
        collectArguments(
            self,
            args=[*self.ALL_INPUTS],
            allInputs=self.ALL_INPUTS,
            **kwargs
        )

        self._loadArrayCenter()

        # Output directory
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)
        self._outputDirectory = io.getLayoutOutputDirectory(self._filesLocation, self.label)
        self._outputDirectory.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self._telescopeList)

    def __getitem__(self, i):
        return self._telescopeList[i]

    @classmethod
    def fromLayoutArrayName(
        cls,
        layoutArrayName,
        label=None,
        filesLocation=None,
        logger=__name__
    ):
        '''
        Create a LayoutArray from a layout name (e.g. South-4LST, North-Prod5, ...)

        Parameters
        ----------
        layoutArrayName: str
            e.g. South-4LST, North-Prod5 ...
        label: str, optional
            Instance label. Important for output file naming.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        logger: str
            Logger name to use in this instance

        Returns
        -------
        Instance of the LayoutArray class.
        '''
        spl = layoutArrayName.split('-')
        siteName = names.validateSiteName(spl[0])
        arrayName = names.validateArrayName(spl[1])
        validLayoutArrayName = siteName + '-' + arrayName

        layout = cls(
            name=validLayoutArrayName,
            label=label,
            filesLocation=filesLocation,
            logger=logger
        )

        telescopeListFile = io.getDataFile(
            'layout',
            'telescope_positions-{}.ecsv'.format(validLayoutArrayName)
        )
        layout.readTelescopeListFile(telescopeListFile)

        return layout
        # End of fromLayoutArrayName

    def _loadArrayCenter(self):
        ''' Load the array center and make convertions if needed. '''

        self._arrayCenter = TelescopeData()
        self._arrayCenter.name = 'array_center'

        self._arrayCenter.setLocalCoordinates(0 * u.m, 0 * u.m, 0 * u.m)
        if self._centerLatitude is not None and self._centerLongitude is not None:
            self._arrayCenter.setMercatorCoordinates(
                self._centerLatitude * u.deg,
                self._centerLongitude * u.deg
            )
        if self._centerEasting is not None and self._centerNorthing is not None:
            self._arrayCenter.setUtmCoordinates(
                self._centerEasting * u.m,
                self._centerNorthing * u.m
            )
        if self._centerAltitude is not None:
            self._arrayCenter.setAltitude(self._centerAltitude * u.m)

        # Converting
        wgs84 = self._getWgs84()
        crs_local = self._getCrsLocal()
        crs_utm = self._getCrsUtm()
        self._arrayCenter.convertAll(
            crsLocal=crs_local,
            wgs84=wgs84,
            crsUtm=crs_utm
        )

        # Filling in center UTM coordinates if needed
        if (
            (self._centerNorthing is None or self._centerEasting is None)
            and self._arrayCenter.hasUtmCoordinates()
        ):
            self._centerNorthing, self._centerEasting = self._arrayCenter.getUtmCoordinates()
    # End of _loadArrayCenter

    def _appendTelescope(self, row, table, prodList):
        ''' Append a new telescope from table row to list of telescopes. '''

        tel = TelescopeData()
        tel.name = row['telescope_name']

        if (
            'pos_x' in table.colnames
            and 'pos_y' in table.colnames
            and 'pos_z' in table.colnames
        ):
            tel.setLocalCoordinates(
                posX=row['pos_x'] * table['pos_x'].unit,
                posY=row['pos_y'] * table['pos_y'].unit,
                posZ=row['pos_z'] * table['pos_z'].unit
            )

        if 'utm_east' in table.colnames and 'utm_north' in table.colnames:
            tel.setUtmCoordinates(
                utmEast=row['utm_east'] * table['utm_east'].unit,
                utmNorth=row['utm_north'] * table['utm_north'].unit
            )

        if 'lat' in table.colnames and 'lon' in table.colnames:
            tel.setMercatorCoordinates(
                latitude=row['lat'] * table['lat'].unit,
                longitude=row['lon'] * table['lon'].unit
            )

        if 'alt' in table.colnames:
            tel.setAltitude(altitude=row['alt'] * table['alt'].unit)

        for prod in prodList:
            tel.prodId[prod] = row[prod]
        self._telescopeList.append(tel)

    def readTelescopeListFile(self, telescopeListFile):
        '''
        Read list of telescopes from a ecsv file.

        Parameters
        ----------
        telescopeListFile: str (or Path)
            Path to the telescope list file.

        Raises
        ------
        InvalidTelescopeListFile
            If cannot read telescope list file or the table does not contain
            telescope_name key.

        '''
        table = Table.read(telescopeListFile, format='ascii.ecsv')

        self._logger.info('Reading telescope list from {}'.format(telescopeListFile))

        # Require telescope_name in telescope lists
        if 'telescope_name' not in table.colnames:
            msg = 'Error reading telescope names from {}'.format(telescopeListFile)
            self._logger.error(msg)
            raise InvalidTelescopeListFile(msg)

        # Reference coordinate system
        if 'EPSG' in table.meta:
            self._epsg = table.meta['EPSG']
        if 'center_northing' in table.meta and 'center_easting' in table.meta:
            self._centerNorthing = u.Quantity(table.meta['center_northing']).to(u.m).value
            self._centerEasting = u.Quantity(table.meta['center_easting']).to(u.m).value
        if 'center_lon' in table.meta and 'center_lat' in table.meta:
            self._centerLongitude = u.Quantity(table.meta['center_lon']).to(u.deg).value
            self._centerLatitude = u.Quantity(table.meta['center_lat']).to(u.deg).value
        if 'center_alt' in table.meta:
            self._centerAltitude = u.Quantity(table.meta['center_alt']).to(u.m).value

        # CORSIKA parameters
        if 'corsika_obs_level' in table.meta:
            self._corsikaObsLevel = u.Quantity(table.meta['corsika_obs_level']).value
        if 'corsika_sphere_center' in table.meta:
            self._corsikaSphereCenter = dict()
            for key, value in table.meta['corsika_sphere_center'].items():
                self._corsikaSphereCenter[key] = u.Quantity(value).to(u.m).value
        if 'corsika_sphere_radius' in table.meta:
            self._corsikaSphereRadius = dict()
            for key, value in table.meta['corsika_sphere_radius'].items():
                self._corsikaSphereRadius[key] = u.Quantity(value).to(u.m).value

        # Initialise telescope lists from productions
        # (require column names include 'prod' string)
        prodList = [row_name for row_name in table.colnames if row_name.find('prod') >= 0]

        for row in table:
            self._appendTelescope(row, table, prodList)

    @u.quantity_input(
        posX=u.m,
        posY=u.m,
        posZ=u.m,
        longitude=u.deg,
        latitude=u.deg,
        utmEast=u.deg,
        utmNorth=u.deg,
        altitude=u.m
    )
    def addTelescope(
        self,
        telescopeName,
        posX=None,
        posY=None,
        posZ=None,
        longitude=None,
        latitude=None,
        utmEast=None,
        utmNorth=None,
        altitude=None
    ):
        '''
        Add an individual telescope to the telescope list.

        Parameters
        ----------
        telescopeName: str
            Name of the telescope starting with L, M or S (e.g. L-01, M-06 ...)
        posX: astropy.units.quantity.Quantity
            X coordinate in equivalent units of u.m.
        posY: astropy.units.quantity.Quantity
            Y coordinate in equivalent units of u.m.
        posZ: astropy.units.quantity.Quantity
            Z coordinate in equivalent units of u.m.
        longitude: astropy.units.quantity.Quantity
            Longitude coordinate in equivalent units of u.deg.
        latitude: astropy.units.quantity.Quantity
            Latitude coordinate in equivalent units of u.deg.
        utmEast: astropy.units.quantity.Quantity
            UTM east coordinate in equivalent units of u.deg.
        utmNorth: astropy.units.quantity.Quantity
            UTM north coordinate in equivalent units of u.deg.
        altitude: astropy.units.quantity.Quantity
            Altitude coordinate in equivalent units of u.m.
        '''

        tel = TelescopeData(
            name=telescopeName,
            posX=posX,
            posY=posY,
            posZ=posZ,
            longitude=longitude,
            latitude=latitude,
            utmEast=utmEast,
            utmNorth=utmNorth,
            altitude=altitude
        )
        self._telescopeList.append(tel)

    def exportTelescopeList(self):
        ''' Export a ECSV file with the telescope positions. '''

        fileName = names.layoutTelescopeListFileName(self.name, None)
        self.telescopeListFile = self._outputDirectory.joinpath(fileName)

        self._logger.debug('Exporting telescope list to ECSV file {}'.format(
            self.telescopeListFile)
        )

        metaData = {
            'center_lon': self._centerLongitude * u.deg,
            'center_lat': self._centerLatitude * u.deg,
            'center_alt': self._centerAltitude * u.m,
            'center_northing': self._centerNorthing * u.m,
            'center_easting': self._centerEasting * u.m,
            'corsika_obs_level': self._corsikaObsLevel * u.m,
            'corsika_sphere_center': {
                key: value * u.m for (key, value) in self._corsikaSphereCenter.items()
            },
            'corsika_sphere_radius': {
                key: value * u.m for (key, value) in self._corsikaSphereRadius.items()
            },
            'EPSG': self._epsg
        }

        table = Table(meta=metaData)

        tel_names = list()
        pos_x, pos_y, pos_z = list(), list(), list()
        utm_east, utm_north = list(), list()
        longitude, latitude = list(), list()
        altitude = list()
        for tel in self._telescopeList:
            tel_names.append(tel.name)

            if tel.hasLocalCoordinates():
                x, y, z = tel.getLocalCoordinates()
                pos_x.append(x)
                pos_y.append(y)
                pos_z.append(z)

            if tel.hasMercatorCoordinates():
                lat, lon = tel.getMercatorCoordinates()
                latitude.append(lat)
                longitude.append(lon)

            if tel.hasUtmCoordinates():
                un, ue = tel.getUtmCoordinates()
                utm_east.append(ue)
                utm_north.append(un)

            if tel.hasAltitude():
                alt = tel.getAltitude()
                altitude.append(alt)

        table['telescope_name'] = tel_names

        if len(pos_x) > 0:
            table['pos_x'] = pos_x * u.m
            table['pos_y'] = pos_x * u.m
            table['pos_z'] = pos_x * u.m

        if len(latitude) > 0:
            table['lat'] = latitude * u.deg
            table['lon'] = longitude * u.deg

        if len(utm_east) > 0:
            table['utm_east'] = utm_east * u.m
            table['utm_north'] = utm_north * u.m

        if len(altitude) > 0:
            table['alt'] = altitude * u.m

        table.write(self.telescopeListFile, format='ascii.ecsv', overwrite=True)
        # End of exportTelescopeList

    def getNumberOfTelescopes(self):
        '''
        Return the number of telescopes in the list.

        Returns
        -------
        int
            Number of telescopes.
        '''
        return len(self._telescopeList)

    def getCorsikaInputList(self):
        '''
        Get a string with the piece of text to be added to
        the CORSIKA input file.

        Returns
        -------
        str
            Piece of text to be added to the CORSIKA input file.
        '''
        corsikaList = ''
        for tel in self._telescopeList:
            posX, posY, posZ = tel.getLocalCoordinates()
            sphereRadius = self._corsikaSphereRadius[tel.getTelescopeSize()]

            corsikaList += 'TELESCOPE'
            corsikaList += '\t {:.3f}E2'.format(posX.value)
            corsikaList += '\t {:.3f}E2'.format(posY.value)
            corsikaList += '\t {:.3f}E2'.format(posZ.value)
            corsikaList += '\t {:.3f}E2'.format(sphereRadius)
            corsikaList += '\t # {}\n'.format(tel.name)

        return corsikaList

    def printTelescopeList(self):
        ''' Print list of telescopes in current layout for inspection. '''
        print('LayoutArray: {}'.format(self.name))
        print('ArrayCenter')
        print(self._arrayCenter)
        print('Telescopes')
        for tel in self._telescopeList:
            print(tel)

    def convertCoordinates(self):
        ''' Perform all the possible conversions the coordinates of the tel positions. '''

        self._logger.info('Converting telescope coordinates')

        # 1: setup reference coordinate systems

        # Mercator WGS84
        wgs84 = self._getWgs84()

        # Local transverse Mercator projection
        crs_local = self._getCrsLocal()

        # UTM system
        crs_utm = self._getCrsUtm()

        # 2. convert coordinates
        for tel in self._telescopeList:

            if self._corsikaObsLevel is not None:
                corsikaObsLevel = self._corsikaObsLevel * u.m
            else:
                corsikaObsLevel = None

            if self._corsikaSphereCenter is not None:
                corsikaSphereCenter = self._corsikaSphereCenter[tel.getTelescopeSize()] * u.m
            else:
                corsikaSphereCenter = None

            tel.convertAll(
                crsLocal=crs_local,
                wgs84=wgs84,
                crsUtm=crs_utm,
                corsikaObsLevel=corsikaObsLevel,
                corsikaSphereCenter=corsikaSphereCenter
            )
    # End of convertCoordinates

    def _getCrsLocal(self):
        ''' Get the crs_local '''
        if self._centerLongitude is not None and self._centerLatitude is not None:
            proj4_string = (
                '+proj=tmerc +ellps=WGS84 +datum=WGS84'
                + ' +lon_0={} +lat_0={}'.format(self._centerLongitude, self._centerLatitude)
                + ' +axis=nwu +units=m +k_0=1.0'
            )
            crs_local = pyproj.CRS.from_proj4(proj4_string)
            self._logger.info('Local Mercator projection: {}'.format(crs_local))
            return crs_local
        else:
            self._logger.warning('crs_local cannot be built because center lon and lat are missing')
            return None

    def _getCrsUtm(self):
        ''' Get crs_utm '''
        if self._epsg is not None:
            crs_utm = pyproj.CRS.from_user_input(self._epsg)
            self._logger.info('UTM system: {}'.format(crs_utm))
            return crs_utm
        else:
            self._logger.warning('crs_utm cannot be built because center lon and lat are missing')
            return None

    def _getWgs84(self):
        ''' Get wgs84 '''
        return pyproj.CRS('EPSG:4326')
