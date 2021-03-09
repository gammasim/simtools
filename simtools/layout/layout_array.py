import astropy.units as u
import logging
from astropy.table import Table

import pyproj

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
        'centerNorthing': {'default': None, 'unit': u.deg},
        'centerEasting': {'default': None, 'unit': u.deg},
        'centerAltitude': {'default': None, 'unit': u.m},
        'corsikaObsLevel': {'default': None, 'unit': u.m},
        'corsikaSphereCenter': {'default': None, 'isDict': True, 'unit': u.m},
        'corsikaSphereRadius': {'default': None, 'isDict': True, 'unit': u.m}
    }

    TEL_SIZE = {'L': 0, 'M': 1, 'S': 2}

    def __init__(self, name=None, logger=__name__, **kwargs):
        """Inits ArrayData with blah."""
        self._logger = logging.getLogger(logger)
        self._logger.debug('Init LayoutArray')

        self.name = name
        self._telescopeList = []

        if name[0] == 'L':
            self._telSize = 'LST'
        elif name[0] == 'M':
            self._telSize = 'MST'
        elif name[0] == 'S':
            self._telSize = 'SST'
        else:
            self._logger.warning('Telescope size could not be guessed from the name')

        # Collecting arguments
        collectArguments(
            self,
            args=[*self.ALL_INPUTS],
            allInputs=self.ALL_INPUTS,
            **kwargs
        )

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
            tel.setMercadorCoordinates(
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
        pass

    # def read_layout(self, layout_list, layout_name):
    #     """
    #     read a layout from a layout yaml file
    #     """

    #     print(layout_name, layout_list)

    #     return None

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

    # def print_array_center(self):
    #     """
    #     print coordinates of array center used
    #     for coordinate transformations
    #     """
    #     print('Array center coordinates:')
    #     if not math.isnan(self.center_lon.value) and \
    #             not math.isnan(self.center_lat.value):
    #         print('\t Longitude {0:0.2f}'.format(self.center_lon))
    #         print('\t Latitude {0:0.2f}'.format(self.center_lat))
    #     if not math.isnan(self.center_northing.value) and \
    #             not math.isnan(self.center_easting.value):
    #         print('\t Northing {0:0.2f}'.format(self.center_northing))
    #         print('\t Easting {0:0.2f}'.format(self.center_easting))
    #     print('\t Altitude {0:0.2f}'.format(self.center_altitude))
    #     print('\t EGSP %s' % (self.epsg))

    # def print_corsika_parameters(self):
    #     """
    #     print CORSIKA parameters defined in header of
    #     ecsv file
    #     """
    #     print('CORSIKA parameters')
    #     print('\t observation level {0:0.2f}'.format(self.corsika_obslevel))
    #     print('\t sphere center ', self.corsika_sphere_center)
    #     print('\t sphere radius ', self.corsika_sphere_radius)

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
                corsikaSphereCenter = self._corsikaSphereCenter[self._telSize] * u.m
            else:
                corsikaSphereCenter = None

            tel.convertAll(
                crsLocal=crs_local,
                wgs84=wgs84,
                crsUtm=crs_utm,
                corsikaObsLevel=corsikaObsLevel,
                corsikaSphereCenter=corsikaSphereCenter
            )

    # def compareArrayCenter(self, layout2):
    #     """
    #     compare array center coordinates of this array
    #     with another one
    #     """
    #     print('comparing array center coordinates')
    #     print('')
    #     print('{0:12s} | {1:>16s} | {2:>16s} | {3:>16s} | '.format(
    #         '', 'layout_1', 'layout_2', 'difference'))
    #     print('{0:12s} | {1:>16s} | {2:>16s} | {3:>16s} | '.format(
    #         '-----', '-----', '-----', '-----'))
    #     print('{0:12s} | {1:12.2f} | {2:12.2f} | {3:12.2f} |'.format(
    #         'Longitude',
    #         self.center_lon, layout2.center_lon,
    #         self.center_lon-layout2.center_lon))
    #     print('{0:12s} | {1:12.2f} | {2:12.2f} | {3:12.2f} |'.format(
    #         'Latitude',
    #         self.center_lat, layout2.center_lat,
    #         self.center_lat-layout2.center_lat))
    #     print('{0:12s} | {1:14.2f} | {2:14.2f} | {3:14.2f} |'.format(
    #         'Northing',
    #         self.center_northing, layout2.center_northing,
    #         self.center_northing-layout2.center_northing))
    #     print('{0:12s} | {1:14.2f} | {2:14.2f} | {3:14.2f} |'.format(
    #         'Easting',
    #         self.center_easting, layout2.center_easting,
    #         self.center_easting-layout2.center_easting))
    #     print('{0:12s} | {1:14.2f} | {2:14.2f} | {3:14.2f} |'.format(
    #         'Altitude',
    #         self.center_altitude, layout2.center_altitude,
    #         self.center_altitude-layout2.center_altitude))

     

    # def print_differences(self, layout2, tolerance_geod=0, tolerance_alt=0):
    #     """
    #     print differences between telescope positions
    #     """

    #     geod = pyproj.Geod(ellps="WGS84")

    #     print('{0:6s} | {1:>14s} | {2:>14s} | {3:>14s} | {4:>14s} | {5:>14s} | {6:>14s} | {7:>12s} | {8:>12s}'.format(
    #         'tel', 'E(layout_1)', 'N(layout_1)', 'alt(layout_1)', \
    #         'E(layout_2)', 'N(layout_2)', 'alt(layout_2)', \
    #         'dist', 'delta_alt'))
    #     print('{0:6s} | {1:>14s} | {2:>14s} | {3:>14s} | {4:>14s} | {5:>14s} | {6:>14s} | {7:>12s} | {8:>10s} |'.format(
    #         '-----', '-----', '-----', '-----', '-----', '-----', \
    #         '-----', '-----', '-----'))
    #     for tel_1 in self.telescope_list:
    #         for tel_2 in layout2.telescope_list:
    #             if tel_1.name != tel_2.name:
    #                 continue
    #             _, _, diff = geod.inv(
    #                 tel_1.lon.value, tel_1.lat.value, 
    #                 tel_2.lon.value, tel_2.lat.value)
    #             if diff <= tolerance_geod and \
    #                     abs(tel_1.alt.value-tel_2.alt.value) <= tolerance_alt:
    #                 continue

    #             print('{0:6s} | {1:12.2f} | {2:12.2f} | {3:12.2f} | {4:12.2f} | {5:12.2f} | {6:12.2f} | {7:10.2f} | {8:10.2f}'.format(
    #                 tel_1.name,
    #                 tel_1.utm_east, tel_1.utm_north, tel_1.alt,
    #                 tel_2.utm_east, tel_2.utm_north, tel_2.alt,
    #                 diff*u.meter, abs(tel_1.alt.value-tel_2.alt.value)*u.meter))

    # def compare_telescope_positions(self, layout2, tolerance_geod=0, tolerance_alt=0):
    #     """
    #     compare telescope positions of two telescope lists
    #     """
    #     print('')
    #     print('comparing telescope positions')
    #     print('  tolerance (pos) {0:f} m)'.format(tolerance_geod))
    #     print('  tolerance (alt) {0:f} m)'.format(tolerance_alt))
    #     print('')
    #     # Step 1: make sure that lists are compatible
    #     for tel_1 in self.telescope_list:
    #         telescope_found = False
    #         for tel_2 in layout2.telescope_list:
    #             if tel_1.name == tel_2.name:
    #                 telescope_found = True

    #         if not telescope_found:
    #             print('Telescope {0:s} from list 1 not found in list 2'.format(
    #                 tel_1.name))

    #     # Step 2: compare coordinate values
    #     self.print_differences(layout2, tolerance_geod, tolerance_alt)
