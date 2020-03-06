#!/usr/bin/python3

import logging
import yaml

__all__ = ['loadConfig']


logger = logging.getLogger(__name__)


def loadConfig(fileName='config.yml'):
    """ Load config file and return it as a dict """
    with open('config.yml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def get(par, fileName='config.yml'):
    """ Return a single value from the config file """
    config = loadConfig(fileName)
    print(config)
    if par not in config.keys():
        logger.error('{} does not contain {}'.format(fileName, par))
        raise KeyError()
    else:
        return config[par]
