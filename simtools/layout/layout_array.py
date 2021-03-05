import math
import astropy.units as u
import logging
from astropy.table import Table

import pyproj

from simtools.util.general import collectArguments
from simtools.layout.telescope_data import TelescopeData


class LayoutArray:
    """
    layout class for
    - storage of telescope position
    - conversion of coordinate systems of positions
    """

    ALL_INPUTS = {
        'epsg': {'default': None},
        'centerLongitude': {'default': None, 'unit': u.deg},
        'centerLatitude': {'default': None, 'unit': u.deg},
        'centerNorthing': {'default': None, 'unit': u.deg},
        'centerEasting': {'default': None, 'unit': u.deg},
        'centerAltitude': {'default': None, 'unit': u.m},
        'corsikaObsLevel': {'default': None, 'unit': u.m},
        'corsikaSphereCenter': {'default': None, 'isList': True, 'unit': u.m},
        'corsikaSphereRadius': {'default': None, 'isList': True, 'unit': u.m}
    }

    def __init__(self, name=None, logger=__name__):
        """Inits ArrayData with blah."""
        self._logger = logging.getLogger(logger)
        self._logger.debug('Init LayoutArray')

        self.name = name
        self._telescopeList = []

        # Collecting arguments
        collectArguments(
            self,
            args=[*self.ALL_INPUTS],
            allInputs=self.ALL_INPUTS,
            **kwargs
        )

    def _append_telescope(self, row, table, prod_list):
        """Append a new telescope from table row
        to list of telescopes
        """

        tel = layout_telescope.TelescopeData()
        tel.name = row['telescope_name']
        if 'pos_x' in table.colnames:
            tel.x = row['pos_x']*table['pos_x'].unit
        if 'pos_y' in table.colnames:
            tel.y = row['pos_y']*table['pos_y'].unit
        if 'pos_z' in table.colnames:
            tel.z = row['pos_z']*table['pos_z'].unit
        if 'utm_east' in table.colnames:
            tel.utm_east = row['utm_east']*table['utm_east'].unit
        if 'utm_north' in table.colnames:
            tel.utm_north = row['utm_north']*table['utm_north'].unit
        if 'alt' in table.colnames:
            tel.alt = row['alt']*table['alt'].unit
        if 'lon' in table.colnames:
            tel.lon = row['lon']*table['lon'].unit
        if 'lat' in table.colnames:
            tel.lat = row['lat']*table['lat'].unit

        for prod in prod_list:
            tel.prod_id[prod] = row[prod]
        return tel


    def read_telescope_list(self, telescope_file):
        """
        read list of telescopes from a ecsv file
        """
        try:
            table = Table.read(telescope_file, format='ascii.ecsv')
        except Exception as ex:
            logging.error('Error reading telescope list from {}'.format(telescope_file))
            logging.error(ex.args)
            return False
        logging.info('reading telescope list from {}'.format(telescope_file))
        
        # require telescope_name in telescope lists
        if 'telescope_name' not in table.colnames:
            logging.error('Error reading telescope names from {}'
                .format(telescope_file))
            logging.error('   required column telescope_name missing')
            logging.error(table.meta)
            return False

        # reference coordinate system
        if 'EPSG' in table.meta:
            self.epsg = table.meta['EPSG']
        if 'center_northing' in table.meta and \
                'center_easting' in table.meta:
            self.center_northing = u.Quantity(table.meta['center_northing'])
            self.center_easting = u.Quantity(table.meta['center_easting'])
        if 'center_lon' in table.meta and \
                'center_lat' in table.meta:
            self.center_lon = u.Quantity(table.meta['center_lon'])
            self.center_lat = u.Quantity(table.meta['center_lat'])
        if 'center_alt' in table.meta:
            self.center_altitude = u.Quantity(table.meta['center_alt'])
        # CORSIKA parameters
        if 'corsika_obs_level' in table.meta:
            self.corsika_obslevel = u.Quantity(table.meta['corsika_obs_level'])
        if 'corsika_sphere_center' in table.meta:
            for key, value in table.meta['corsika_sphere_center'].items():
                self.corsika_sphere_center[key] = u.Quantity(value)
        if 'corsika_sphere_radius' in table.meta:
            for key, value in table.meta['corsika_sphere_radius'].items(): 
                self.corsika_sphere_radius[key] = u.Quantity(value)

        
        # initialise telescope lists from productions
        # (require column names include 'prod' string)
        prod_list = [row_name for row_name in table.colnames if row_name.find('prod') >= 0]

        self.telescope_list = [self._append_telescope(row, table, prod_list) for row in table]

        return True

    def read_layout(self, layout_list, layout_name):
        """
        read a layout from a layout yaml file
        """

        print(layout_name, layout_list)

        return None

    def print_telescope_list(self, short_printout):
        """
        print list of telescopes in current layout

        Available formats (examples, column names in ecsv file):
        - telescope_name - default telescope names
        - prod3b_mst_N - North layout (with MST-NectarCam)
        """
        for tel in self.telescope_list:
            if short_printout:
                tel.print_short_telescope_list()
            else:
                tel.print_telescope()

        return None

    def print_array_center(self):
        """
        print coordinates of array center used
        for coordinate transformations
        """
        print('Array center coordinates:')
        if not math.isnan(self.center_lon.value) and \
                not math.isnan(self.center_lat.value):
            print('\t Longitude {0:0.2f}'.format(self.center_lon))
            print('\t Latitude {0:0.2f}'.format(self.center_lat))
        if not math.isnan(self.center_northing.value) and \
                not math.isnan(self.center_easting.value):
            print('\t Northing {0:0.2f}'.format(self.center_northing))
            print('\t Easting {0:0.2f}'.format(self.center_easting))
        print('\t Altitude {0:0.2f}'.format(self.center_altitude))
        print('\t EGSP %s' % (self.epsg))

    def print_corsika_parameters(self):
        """
        print CORSIKA parameters defined in header of
        ecsv file
        """
        print('CORSIKA parameters')
        print('\t observation level {0:0.2f}'.format(self.corsika_obslevel))
        print('\t sphere center ', self.corsika_sphere_center)
        print('\t sphere radius ', self.corsika_sphere_radius)

    def convert_coordinates(self):
        """
        conversion depends what is given in the orginal
        telescope list

        after conversion, following coordinates should
        be filled:
        - local transverse Mercator projection
        - Mercator (WGS84) projection
        - UTM coordinate system
        """

        logging.info('Converting telescope coordinates')

        # 1: setup reference coordinate systems

        # Mercator WGS84
        wgs84 = pyproj.CRS('EPSG:4326')
        
        # local transverse Mercator projection
        crs_local = None
        if self.center_lon is not None \
                and self.center_lat is not None:
            proj4_string = '+proj=tmerc +ellps=WGS84 +datum=WGS84'
            proj4_string = '%s +lon_0=%s +lat_0=%s' % \
                (proj4_string,
                 self.center_lon.value,
                 self.center_lat.value)
            proj4_string = '%s +axis=nwu +units=m +k_0=1.0' % \
                (proj4_string)
            crs_local = pyproj.CRS.from_proj4(proj4_string)
            logging.info('\t Local Mercator projection: {}'.format(crs_local))
        # UTM system
        crs_utm = None
        if not math.isnan(self.epsg):
            crs_utm = pyproj.CRS.from_user_input(self.epsg)
            logging.info('\t UTM system: {}'.format(crs_utm))
        # 2. convert coordinates
        for tel in self.telescope_list:
            tel.convert(crs_local, 
                        wgs84, 
                        crs_utm, 
                        self.center_altitude,
                        self.corsika_obslevel, 
                        self.corsika_sphere_center)

    def compare_array_center(self, layout2):
        """
        compare array center coordinates of this array
        with another one
        """
        print('comparing array center coordinates')
        print('')
        print('{0:12s} | {1:>16s} | {2:>16s} | {3:>16s} | '.format(
            '', 'layout_1', 'layout_2', 'difference'))
        print('{0:12s} | {1:>16s} | {2:>16s} | {3:>16s} | '.format(
            '-----', '-----', '-----', '-----'))
        print('{0:12s} | {1:12.2f} | {2:12.2f} | {3:12.2f} |'.format(
            'Longitude',
            self.center_lon, layout2.center_lon,
            self.center_lon-layout2.center_lon))
        print('{0:12s} | {1:12.2f} | {2:12.2f} | {3:12.2f} |'.format(
            'Latitude',
            self.center_lat, layout2.center_lat,
            self.center_lat-layout2.center_lat))
        print('{0:12s} | {1:14.2f} | {2:14.2f} | {3:14.2f} |'.format(
            'Northing',
            self.center_northing, layout2.center_northing,
            self.center_northing-layout2.center_northing))
        print('{0:12s} | {1:14.2f} | {2:14.2f} | {3:14.2f} |'.format(
            'Easting',
            self.center_easting, layout2.center_easting,
            self.center_easting-layout2.center_easting))
        print('{0:12s} | {1:14.2f} | {2:14.2f} | {3:14.2f} |'.format(
            'Altitude',
            self.center_altitude, layout2.center_altitude,
            self.center_altitude-layout2.center_altitude))

     

    def print_differences(self, layout2, tolerance_geod=0, tolerance_alt=0):
        """
        print differences between telescope positions
        """

        geod = pyproj.Geod(ellps="WGS84")

        print('{0:6s} | {1:>14s} | {2:>14s} | {3:>14s} | {4:>14s} | {5:>14s} | {6:>14s} | {7:>12s} | {8:>12s}'.format(
            'tel', 'E(layout_1)', 'N(layout_1)', 'alt(layout_1)', \
            'E(layout_2)', 'N(layout_2)', 'alt(layout_2)', \
            'dist', 'delta_alt'))
        print('{0:6s} | {1:>14s} | {2:>14s} | {3:>14s} | {4:>14s} | {5:>14s} | {6:>14s} | {7:>12s} | {8:>10s} |'.format(
            '-----', '-----', '-----', '-----', '-----', '-----', \
            '-----', '-----', '-----'))
        for tel_1 in self.telescope_list:
            for tel_2 in layout2.telescope_list:
                if tel_1.name != tel_2.name:
                    continue
                _, _, diff = geod.inv(
                    tel_1.lon.value, tel_1.lat.value, 
                    tel_2.lon.value, tel_2.lat.value)
                if diff <= tolerance_geod and \
                        abs(tel_1.alt.value-tel_2.alt.value) <= tolerance_alt:
                    continue

                print('{0:6s} | {1:12.2f} | {2:12.2f} | {3:12.2f} | {4:12.2f} | {5:12.2f} | {6:12.2f} | {7:10.2f} | {8:10.2f}'.format(
                    tel_1.name,
                    tel_1.utm_east, tel_1.utm_north, tel_1.alt,
                    tel_2.utm_east, tel_2.utm_north, tel_2.alt,
                    diff*u.meter, abs(tel_1.alt.value-tel_2.alt.value)*u.meter))

    def compare_telescope_positions(self, layout2, tolerance_geod=0, tolerance_alt=0):
        """
        compare telescope positions of two telescope lists
        """
        print('')
        print('comparing telescope positions')
        print('  tolerance (pos) {0:f} m)'.format(tolerance_geod))
        print('  tolerance (alt) {0:f} m)'.format(tolerance_alt))
        print('')
        # Step 1: make sure that lists are compatible
        for tel_1 in self.telescope_list:
            telescope_found = False
            for tel_2 in layout2.telescope_list:
                if tel_1.name == tel_2.name:
                    telescope_found = True

            if not telescope_found:
                print('Telescope {0:s} from list 1 not found in list 2'.format(
                    tel_1.name))

        # Step 2: compare coordinate values
        self.print_differences(layout2, tolerance_geod, tolerance_alt)
