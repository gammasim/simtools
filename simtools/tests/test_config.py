#!/usr/bin/python3

import logging

from simtools.util import config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_get():
    yamlDBPath = cfg.get('yamlDBPath')
    logger.info(yamlDBPath)
    # nek = config.get('NonExistingKey')


def test_input_options():
    print('yamlDBPath: {}'.format(cfg.get('yamlDBPath')))
    cfg.setConfigFileName('config.yml')
    print('cfg.CONFIG_FILE_NAME: {}'.format(cfg.CONFIG_FILE_NAME))
    print('yamlDBPath: {}'.format(cfg.get('yamlDBPath')))


if __name__ == '__main__':

    # test_get()
    test_input_options()
