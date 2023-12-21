#!/usr/bin/python3

import logging

import pytest
from astropy.utils.diff import report_diff_values

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
    assert 151.0 == pytest.approx(mirrors.mirror_diameter.value)
    assert 3 == mirrors.shape_type
    mirror_list_file = io_handler.get_input_data_file(
        file_name="MLTdata-preproduction.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    assert 1590.35 == pytest.approx(mirrors.mirror_table["focal_length"][0])


def test_get_mirror_table(io_handler):
    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv",
        test=True,
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)

    report = report_diff_values(mirrors.mirror_table, mirrors.get_mirror_table())
    assert report is True


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

    def assert_mirror_parameters():
        assert 1022.49 == mirror_x
        assert -462.0 == mirror_y
        assert 151.0 == mirror_diameter
        assert 2920.0 == focal_length
        assert 3 == shape_type

    assert_mirror_parameters()

    mirror_list_file = io_handler.get_input_data_file(
        file_name="mirror_list_CTA-N-LST1_v2019-03-31_rotated_simtel.dat",
        test=True,
    )
    logger.info(f"Using mirror list with simtel format {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    (
        mirror_x,
        mirror_y,
        mirror_diameter,
        focal_length,
        shape_type,
    ) = mirrors.get_single_mirror_parameters(198)
    assert_mirror_parameters()
    logger.info("Wrong mirror id returns the first mirror table row")
    mirrors = Mirrors(mirror_list_file)
    (
        mirror_x,
        mirror_y,
        mirror_diameter,
        focal_length,
        shape_type,
    ) = mirrors.get_single_mirror_parameters(9999)
    assert_mirror_parameters()
