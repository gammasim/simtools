import astropy.units as u
import logging
from astropy.table import Table

import pyproj

import simtools.config as cfg
import simtools.io_handler as io
from simtools.util import names
from simtools.util.general import collectArguments
from simtools.layout.telescope_data import TelescopeData


class InvalidTelescopeListFile(Exception):
    pass


class LayoutArray:
    """
    layout class for
    - storage of telescope position
    - conversion of coordinate systems of positions
    """

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
        """Inits ArrayData with blah."""
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

        # Output directory
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)
        self._outputDirectory = io.getLayoutOutputDirectory(self._filesLocation, self.label)
        self._outputDirectory.mkdir(parents=True, exist_ok=True)

    def _appendTelescope(self, row, table, prodList):
        """Append a new telescope from table row
        to list of telescopes
        """

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

    def readTelescopeListFile(self, telescopeFile):
        """
        read list of telescopes from a ecsv file
        """
        table = Table.read(telescopeFile, format='ascii.ecsv')

        self._logger.info('Reading telescope list from {}'.format(telescopeFile))

        # Require telescope_name in telescope lists
        if 'telescope_name' not in table.colnames:
            msg = 'Error reading telescope names from {}'.format(telescopeFile)
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

        return True

    # def addListOfTelescopes(self, telescopes):
    #     """
    #     """
    #     if not telescopes

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
        """
        """

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

        fileName = names.layoutTelescopeListFileName(self.name, self.label)
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
            table['utm_east'] = utm_east * u.deg
            table['utm_north'] = utm_north * u.deg

        if len(altitude) > 0:
            table['alt'] = altitude * u.m

        table.write(self.telescopeListFile, format='ascii.ecsv', overwrite=True)

    def getNumberOfTelescopes(self):
        return len(self._telescopeList)

    def getCorsikaInputList(self):
        '''
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

    def printTelescopeList(self, short=False):
        """
        print list of telescopes in current layout

        Available formats (examples, column names in ecsv file):
        - telescope_name - default telescope names
        - prod3b_mst_N - North layout (with MST-NectarCam)
        """
        for tel in self._telescopeList:
            print(tel)

        return None

    def convertCoordinates(self):
        """
        conversion depends what is given in the orginal
        telescope list

        after conversion, following coordinates should
        be filled:
        - local transverse Mercator projection
        - Mercator (WGS84) projection
        - UTM coordinate system
        """

        self._logger.info('Converting telescope coordinates')

        # 1: setup reference coordinate systems

        # Mercator WGS84
        wgs84 = pyproj.CRS('EPSG:4326')

        # Local transverse Mercator projection
        crs_local = None
        if self._centerLongitude is not None and self._centerLatitude is not None:
            proj4_string = (
                '+proj=tmerc +ellps=WGS84 +datum=WGS84'
                + ' +lon_0={} +lat_0={}'.format(self._centerLongitude, self._centerLatitude)
                + ' +axis=nwu +units=m +k_0=1.0'
            )
            crs_local = pyproj.CRS.from_proj4(proj4_string)
            self._logger.info('Local Mercator projection: {}'.format(crs_local))
        else:
            self._logger.warning('crs_local cannot be built because center lon and lat are missing')

        # UTM system
        crs_utm = None
        if self._epsg is not None:
            crs_utm = pyproj.CRS.from_user_input(self._epsg)
            self._logger.info('UTM system: {}'.format(crs_utm))
        else:
            self._logger.warning('crs_utm cannot be built because center lon and lat are missing')

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
