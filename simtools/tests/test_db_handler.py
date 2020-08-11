#!/usr/bin/python3

import logging
import subprocess
from pathlib import Path

from simtools import db_handler
import simtools.config as cfg
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

testDataDirectory = './data/test-output'
DB_CTA_SIMULATION_MODEL = 'CTA-Simulation-Model'


def test_reading_db_lst():

    logger.info('----Testing reading LST-----')
    db = db_handler.DatabaseHandler(logger.name)
    pars = db.getModelParameters('north-lst-1', 'prod4', testDataDirectory)
    assert(pars['parabolic_dish']['Value'] == 1)
    assert(pars['camera_pixels']['Value'] == 1855)

    logger.info('Listing files written in {}'.format(testDataDirectory))
    subprocess.call(['ls -lh {}'.format(testDataDirectory)], shell=True)

    return


def test_reading_db_mst_nc():

    logger.info('----Testing reading MST-NectarCam-----')
    db = db_handler.DatabaseHandler(logger.name)
    pars = db.getModelParameters('north-mst-NectarCam-D', 'prod4', testDataDirectory)
    assert(pars['camera_pixels']['Value'] == 1855)

    logger.info('Listing files written in {}'.format(testDataDirectory))
    subprocess.call(['ls -lh {}'.format(testDataDirectory)], shell=True)

    logger.info('Removing the files written in {}'.format(testDataDirectory))
    subprocess.call(['rm -f {}/*'.format(testDataDirectory)], shell=True)

    return


def test_reading_db_mst_fc():

    logger.info('----Testing reading MST-FlashCam-----')
    db = db_handler.DatabaseHandler(logger.name)
    pars = db.getModelParameters('north-mst-FlashCam-D', 'prod4', testDataDirectory)
    assert(pars['camera_pixels']['Value'] == 1764)

    logger.info('Listing files written in {}'.format(testDataDirectory))
    subprocess.call(['ls -lh {}'.format(testDataDirectory)], shell=True)

    logger.info('Removing the files written in {}'.format(testDataDirectory))
    subprocess.call(['rm -f {}/*'.format(testDataDirectory)], shell=True)

    return


def test_reading_db_sst():

    logger.info('----Testing reading SST-----')
    db = db_handler.DatabaseHandler(logger.name)
    pars = db.getModelParameters('south-sst-D', 'prod4', testDataDirectory)
    assert(pars['camera_pixels']['Value'] == 2048)

    logger.info('Listing files written in {}'.format(testDataDirectory))
    subprocess.call(['ls -lh {}'.format(testDataDirectory)], shell=True)

    logger.info('Removing the files written in {}'.format(testDataDirectory))
    subprocess.call(['rm -f {}/*'.format(testDataDirectory)], shell=True)

    return


def test_modify_db():

    logger.info('----Testing copying a whole telescope-----')
    db = db_handler.DatabaseHandler(logger.name)
    db.copyTelescope(
        DB_CTA_SIMULATION_MODEL,
        'North-LST-1',
        'prod4',
        'North-LST-Test',
        'sandbox'
    )
    pars = db.readMongoDB('sandbox', 'North-LST-Test', 'prod4', testDataDirectory, False)
    assert(pars['camera_pixels']['Value'] == 1855)

    logger.info('----Testing adding a parameter-----')
    db.addParameter(
        'sandbox',
        'North-LST-Test',
        'camera_config_version',
        'test',
        42
    )
    pars = db.readMongoDB('sandbox', 'North-LST-Test', 'test', testDataDirectory, False)
    assert(pars['camera_config_version']['Value'] == 42)

    logger.info('----Testing updating a parameter-----')
    db.updateParameter(
        'sandbox',
        'North-LST-Test',
        'test',
        'camera_config_version',
        999
    )
    pars = db.readMongoDB('sandbox', 'North-LST-Test', 'test', testDataDirectory, False)
    assert(pars['camera_config_version']['Value'] == 999)

    logger.info('----Testing adding a new parameter-----')
    db.addNewParameter(
        'sandbox',
        'North-LST-Test',
        'test',
        'camera_config_version_test',
        999
    )
    pars = db.readMongoDB('sandbox', 'North-LST-Test', 'test', testDataDirectory, False)
    assert(pars['camera_config_version_test']['Value'] == 999)

    logger.info('----Testing deleting a query (a whole telescope in this case)-----')
    query = {'Telescope': 'North-LST-Test'}
    db.deleteQuery('sandbox', query)

    return


if __name__ == '__main__':

    # test_get_model_file()
    test_reading_db_lst()
    test_reading_db_mst_nc()
    test_reading_db_mst_fc()
    test_reading_db_sst()
    test_modify_db()
    pass
