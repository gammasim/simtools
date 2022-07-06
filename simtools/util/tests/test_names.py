#!/usr/bin/python3

import logging
import pytest

from simtools.util import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_names():
    def validate(name):
        print("Validating {}".format(name))
        newName = names.validateTelescopeModelName(name)
        print("New name {}".format(newName))

    for n in ["sst-d", "mst-flashcam-d", "sct-d"]:
        validate(n)


def test_validate_other_names():
    modelVersion = names.validateModelVersionName("p4")
    print(modelVersion)


def test_simtoolsInstrumentName():

    assert (
        names.simtoolsInstrumentName(
            'South', 'MST', 'FlashCam', 'D')
        == 'South-MST-FlashCam-D')
    assert (
        names.simtoolsInstrumentName(
            'North', 'MST', 'NectarCam', '7')
        == 'North-MST-NectarCam-7')

    with pytest.raises(ValueError):
        names.simtoolsInstrumentName(
            'West', 'MST', 'FlashCam', 'D')


if __name__ == "__main__":

    test_validate_telescope_names()
    test_validate_other_names()
