#!/usr/bin/python3

import logging
import yaml

__all__ = ['loadConfig', 'get']


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setConfigFileName(fileName):
    logger.debug('Setting the config file name to {}'.format(fileName))
    global CONFIG_FILE_NAME
    CONFIG_FILE_NAME = fileName


def loadConfig(fileName=None):
    """ Load config file and return it as a dict
        3 possible options for the config fileName:
        1st - fileName parameter is given (not None)
        2nd - CONFIG_FILE_NAME exists (set by setConfigFileName)
        3rd - ./config.yml
    """

    if fileName is not None:
        thisFileName = fileName
    elif 'CONFIG_FILE_NAME' in globals():
        thisFileName = CONFIG_FILE_NAME
    else:
        thisFileName = 'config.yml'

    with open(thisFileName, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def get(par, fileName=None):
    """ Return a single value from the config file """
    config = loadConfig(fileName)
    if par not in config.keys():
        logger.error('Config does not contain {}'.format(par))
        raise KeyError()
    else:
        return config[par]
