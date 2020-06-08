#!/usr/bin/python3

import logging
from pathlib import Path

import simtools.db_handler as db
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_model_parameters():

    pars = db.getModelParameters('lst', 'prod4')
    logger.info('----LST-----')
    logger.info(pars)

    pars = db.getModelParameters('sst', 'prod4')
    logger.info('----SST-----')
    logger.info(pars)

    pars = db.getModelParameters('mst-flashcam', 'prod4')
    logger.info('----MST-FlashCam-----')
    logger.info(pars)

    return


def test_get_model_file():

    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-lst'
    )

    fileName = tel.getParameter('camera_config_file')

    logger.info('FileName = {}'.format(fileName))

    file = db.getModelFile(fileName)

    logger.info('FilePath = {}'.format(file))

    destDir = Path.cwd()
    db.writeModelFile(fileName, destDir)

    return


if __name__ == '__main__':

    # test_get_model_parameters()
    test_get_model_file()
    pass
