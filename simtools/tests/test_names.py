#!/usr/bin/python3

import logging
from simtools.util import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_names():

    def validate(name):
        print('Validating {}'.format(name))
        newName = names.validateTelescopeName(name)
        print('New name {}'.format(newName))

    for n in ['north-sst-d', 'south-mst-flashcam-d', 'north-sct-d']:
        validate(n)


def test_validate_other_names():
    modelVersion = names.validateModelVersionName('p4')
    print(modelVersion)


if __name__ == '__main__':

    test_validate_telescope_names()
    test_validate_other_names()
    pass
