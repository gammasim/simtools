#!/usr/bin/python3

import logging
from simtools.util import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_names():

    def validate(name):
        print('Validating {}'.format(name))
        newName = names.validateTelescopeModelName(name)
        print('New name {}'.format(newName))

    for n in ['sst-d', 'mst-flashcam-d', 'sct-d']:
        validate(n)


def test_validate_other_names():
    modelVersion = names.validateModelVersionName('p4')
    print(modelVersion)


if __name__ == '__main__':

    test_validate_telescope_names()
    test_validate_other_names()
