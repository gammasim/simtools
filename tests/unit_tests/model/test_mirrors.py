#!/usr/bin/python3

import logging

import pytest

from simtools.model.mirrors import Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_list(db, io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated_simtel.dat",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    assert 198 == mirrors.number_of_mirrors
    assert 151.0 == pytest.approx(mirrors.mirror_diameter)
    assert 3 == mirrors.shape_type


def test_read_mirror_list_from_ecsv(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    assert 198 == mirrors.number_of_mirrors
    assert 151.0 == pytest.approx(mirrors.mirror_diameter)
    assert 3 == mirrors.shape_type


def test_get_single_mirror_parameters(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    (
        mirror_x,
        mirror_y,
        mirror_diameter,
        focal_length,
        shape_type,
    ) = mirrors.get_single_mirror_parameters(198)
    assert 1022.49 == mirror_x
    assert -462.0 == mirror_y
    assert 151.0 == mirror_diameter
    assert 2920.0 == focal_length
    assert 3 == shape_type
