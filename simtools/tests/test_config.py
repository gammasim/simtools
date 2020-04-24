#!/usr/bin/python3

import logging

from simtools.util import config as cfg

logging.getLogger().setLevel(logging.DEBUG)


def test_get():
    modelFilesLocations = cfg.get('modelFilesLocations')
    logging.info(modelFilesLocations)
    # nek = config.get('NonExistingKey')


def test_input_options():
    print('modelFilesLocations: {}'.format(cfg.get('modelFilesLocations')))
    cfg.setConfigFileName('config.yml')
    print('cfg.CONFIG_FILE_NAME: {}'.format(cfg.CONFIG_FILE_NAME))
    print('modelFilesLocations: {}'.format(cfg.get('modelFilesLocations')))


def test_find_file():
    f1 = cfg.findFile('mirror_MST_D80.dat')
    print(f1)
    f2 = cfg.findFile('parValues-LST.yml')
    print(f2)


if __name__ == '__main__':

    # test_get()
    # test_input_options()
    # test_find_file()
    pass
