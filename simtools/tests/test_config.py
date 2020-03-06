#!/usr/bin/python3

import logging

from simtools.util import config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_get():
    yamlDBPath = cfg.get('yamlDBPath')
    logging.info(yamlDBPath)
    # nek = config.get('NonExistingKey')


if __name__ == '__main__':

    test_get()
