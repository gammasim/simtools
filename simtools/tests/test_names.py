#!/usr/bin/python3

import logging
from simtools.util import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_names():
    newName = names.validateTelescopeName('north-sst-D')
    print(newName)


def test_validate_other_names():
    modelVersion = names.validateModelVersionName('p4')
    print(modelVersion)


if __name__ == '__main__':

    test_validate_telescope_names()
    test_validate_other_names()
    pass
