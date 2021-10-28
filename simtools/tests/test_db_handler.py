#!/usr/bin/python3

import logging
import subprocess
import unittest
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler


class TestDBHandler(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.db = db_handler.DatabaseHandler()
        self.testDataDirectory = io.getTestOutputDirectory()
        self.DB_CTA_SIMULATION_MODEL = 'CTA-Simulation-Model'

    def test_reading_db_lst(self):

        self.logger.info('----Testing reading LST-----')
        pars = self.db.getModelParameters('north', 'lst-1', 'Current')
        if cfg.get('useMongoDB'):
            assert(pars['parabolic_dish']['Value'] == 1)
            assert(pars['camera_pixels']['Value'] == 1855)
        else:
            assert(pars['parabolic_dish'] == 1)
            assert(pars['camera_pixels'] == 1855)

        self.db.exportModelFiles(pars, self.testDataDirectory)
        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

    def test_reading_db_mst_nc(self):

        self.logger.info('----Testing reading MST-NectarCam-----')
        pars = self.db.getModelParameters(
            'north',
            'mst-NectarCam-D',
            'Current'
        )
        if cfg.get('useMongoDB'):
            assert(pars['camera_pixels']['Value'] == 1855)
        else:
            assert(pars['camera_pixels'] == 1855)

        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

        self.logger.info('Removing the files written in {}'.format(self.testDataDirectory))
        subprocess.call(['rm -f {}/*'.format(self.testDataDirectory)], shell=True)

    def test_reading_db_mst_fc(self):

        self.logger.info('----Testing reading MST-FlashCam-----')
        pars = self.db.getModelParameters(
            'north',
            'mst-FlashCam-D',
            'Current'
        )
        if cfg.get('useMongoDB'):
            assert(pars['camera_pixels']['Value'] == 1764)
        else:
            assert(pars['camera_pixels'] == 1764)

        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

        self.logger.info('Removing the files written in {}'.format(self.testDataDirectory))
        subprocess.call(['rm -f {}/*'.format(self.testDataDirectory)], shell=True)

    def test_reading_db_sst(self):

        self.logger.info('----Testing reading SST-----')
        pars = self.db.getModelParameters('south', 'sst-D', 'Current')
        if cfg.get('useMongoDB'):
            assert(pars['camera_pixels']['Value'] == 2048)
        else:
            assert(pars['camera_pixels'] == 2048)

        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

        self.logger.info('Removing the files written in {}'.format(self.testDataDirectory))
        subprocess.call(['rm -f {}/*'.format(self.testDataDirectory)], shell=True)

    def test_modify_db(self):

        # This test is only relevant for the MongoDB
        if not cfg.get('useMongoDB'):
            return

        self.logger.info('----Testing copying a whole telescope-----')
        self.db.copyTelescope(
            self.DB_CTA_SIMULATION_MODEL,
            'North-LST-1',
            'Current',
            'North-LST-Test',
            'sandbox'
        )
        self.db.copyDocuments(
            self.DB_CTA_SIMULATION_MODEL,
            'metadata',
            {'Entry': 'Simulation-Model-Tags'},
            'sandbox'
        )
        pars = self.db.readMongoDB(
            'sandbox',
            'North-LST-Test',
            'Current',
            self.testDataDirectory,
            False
        )
        assert(pars['camera_pixels']['Value'] == 1855)

        self.logger.info('----Testing adding a parameter-----')
        self.db.addParameter(
            'sandbox',
            'North-LST-Test',
            'camera_config_version',
            'test',
            42
        )
        pars = self.db.readMongoDB(
            'sandbox',
            'North-LST-Test',
            'test',
            self.testDataDirectory,
            False
        )
        assert(pars['camera_config_version']['Value'] == 42)

        self.logger.info('----Testing updating a parameter-----')
        self.db.updateParameter(
            'sandbox',
            'North-LST-Test',
            'test',
            'camera_config_version',
            999
        )
        pars = self.db.readMongoDB(
            'sandbox',
            'North-LST-Test',
            'test',
            self.testDataDirectory,
            False
        )
        assert(pars['camera_config_version']['Value'] == 999)

        self.logger.info('----Testing adding a new parameter-----')
        self.db.addNewParameter(
            'sandbox',
            'North-LST-Test',
            'test',
            'camera_config_version_test',
            999
        )
        pars = self.db.readMongoDB(
            'sandbox',
            'North-LST-Test',
            'test',
            self.testDataDirectory,
            False
        )
        assert(pars['camera_config_version_test']['Value'] == 999)

        self.logger.info('Testing deleting a query (a whole telescope in this case and metadata)')
        query = {'Telescope': 'North-LST-Test'}
        self.db.deleteQuery('sandbox', 'telescopes', query)
        query = {'Entry': 'Simulation-Model-Tags'}
        self.db.deleteQuery('sandbox', 'metadata', query)

    def test_reading_db_sites(self):

        # This test is only relevant for the MongoDB
        if not cfg.get('useMongoDB'):
            return

        self.logger.info('----Testing reading La Palma parameters-----')
        pars = self.db.getSiteParameters('North', 'Current')
        if cfg.get('useMongoDB'):
            assert(pars['altitude']['Value'] == 2158)
        else:
            assert(pars['altitude'] == 2158)

        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

        self.logger.info('Removing the files written in {}'.format(self.testDataDirectory))
        subprocess.call(['rm -f {}/*'.format(self.testDataDirectory)], shell=True)

        self.logger.info('----Testing reading Paranal parameters-----')
        pars = self.db.getSiteParameters('South', 'Current')
        if cfg.get('useMongoDB'):
            assert(pars['altitude']['Value'] == 2147)
        else:
            assert(pars['altitude'] == 2147)

        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

        self.logger.info('Removing the files written in {}'.format(self.testDataDirectory))
        subprocess.call(['rm -f {}/*'.format(self.testDataDirectory)], shell=True)

        return

    def test_separating_get_and_write(self):
        pars = self.db.getModelParameters('north', 'lst-1', 'prod4')

        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

        self.db.exportModelFiles(pars, self.testDataDirectory)

        self.logger.info('Listing files written in {}'.format(self.testDataDirectory))
        subprocess.call(['ls -lh {}'.format(self.testDataDirectory)], shell=True)

    def test_insert_files_db(self):

        # This test is only relevant for the MongoDB
        if not cfg.get('useMongoDB'):
            return

        self.logger.info('----Testing inserting files to the DB-----')
        self.logger.info('Creating a temporary file in {}'.format(self.testDataDirectory))
        fileName = Path(self.testDataDirectory).joinpath('test_file.dat')
        with open(fileName, 'w') as f:
            f.write('# This is a test file')

        file_id = self.db.insertFileToDB(fileName, 'sandbox')
        assert(file_id == self.db._getFileMongoDB('sandbox', 'test_file.dat'))

        subprocess.call(['rm -f {}'.format(fileName)], shell=True)


if __name__ == '__main__':
    unittest.main()

    # tt = TestDBHandler()
    # tt.setUp()
    # tt.test_separating_get_and_write()

    # tt.test_reading_db_lst()
    # test_reading_db_mst_nc()
    # test_reading_db_mst_fc()
    # test_reading_db_sst()
    # test_modify_db()
    # test_reading_db_sites()
