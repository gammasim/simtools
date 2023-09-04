#!/usr/bin/python3

import logging

import pytest

from simtools.psf_analysis import PSFImage

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_reading_simtel_file(args_dict, io_handler):
    test_file = io_handler.get_input_data_file(
        file_name="photons-North-LST-1-d10.0-za20.0-off0.000_validate_optics.lis.gz",
        test=True,
    )
    image = PSFImage(focal_length=2800.0)
    image.read_photon_list_from_simtel_file(test_file)
    logger.info(image.get_psf(0.8, "cm"))

    assert 3.343415291615846 == pytest.approx(image.get_psf(0.8, "cm"))
