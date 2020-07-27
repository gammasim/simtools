#!/usr/bin/python3

import logging
import subprocess
from pathlib import Path

import simtools.db_handler as db
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

testDataDirectory = './data/test-output'


def test_reading_db_lst():

    logger.info('----LST-----')
    pars = db.getModelParameters('north-lst-1', 'prod4', testDataDirectory)
    assert(pars['parabolic_dish']['Value'] == 1)
    assert(pars['camera_pixels']['Value'] == 1855)

    logger.info('Listing files written in {}'.format(testDataDirectory))
    subprocess.call(['ls -lh {}'.format(testDataDirectory)], shell=True)

    return


def test_reading_db_mst_nc():

    logger.info('----MST-NectarCam-----')
    pars = db.getModelParameters('north-mst-nectarcam-d', 'prod4', testDataDirectory)
    assert(pars['camera_pixels']['Value'] == 1855)

    return


def test_reading_db_mst_fc():

    logger.info('----MST-FlashCam-----')
    pars = db.getModelParameters('north-mst-flashcam-d', 'prod4', testDataDirectory)
    assert(pars['camera_pixels']['Value'] == 1764)

    return


def test_reading_db_sst():

    logger.info('----SST-----')
    pars = db.getModelParameters('south-sst-d', 'prod4', testDataDirectory)
    assert(pars['camera_pixels']['Value'] == 2048)

    return


# def test_get_model_file():

#     tel = TelescopeModel(
#         telescopeType='lst',
#         site='south',
#         version='prod4',
#         label='test-lst'
#     )

#     fileName = tel.getParameter('camera_config_file')

#     logger.info('FileName = {}'.format(fileName))

#     file = db.getModelFile(fileName)

#     logger.info('FilePath = {}'.format(file))

#     destDir = Path.cwd()
#     db.writeModelFile(fileName, destDir)

#     return


if __name__ == '__main__':

    # test_get_model_file()
    test_reading_db_lst()
    test_reading_db_mst_nc()
    test_reading_db_mst_fc()
    test_reading_db_sst()
    pass
