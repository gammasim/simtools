#!/usr/bin/python3

import logging

import pytest

from simtools.psf_analysis import PSFImage

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_reading_simtel_file(args_dict, io_handler):

    testFile = io_handler.get_input_data_file(
        fileName="photons-LST-d12.0-za20.0-off0.000_lst_integral.lis",
        test=True,
    )
    image = PSFImage(focalLength=2800.0)
    image.read_photon_list_from_simtel_file(testFile)
    logger.info(image.get_psf(0.8, "cm"))

    assert 3.2248259134010397 == pytest.approx(image.get_psf(0.8, "cm"))
