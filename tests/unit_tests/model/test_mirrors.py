#!/usr/bin/python3

import logging

import pytest

import simtools.utils.general as gen
from simtools.model.mirrors import Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_list(db, io_handler):
    test_file_name = "mirror_CTA-LST-flen_grouped.dat"
    db.export_file_db(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        dest=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        file_name=test_file_name,
    )
    mirror_list_file = gen.find_file(
        test_file_name, io_handler.get_output_directory(sub_dir="model", dir_type="test")
    )
    logger.info(f"Using mirror list {mirror_list_file}")
    mirrors = Mirrors(mirror_list_file)
    assert 198 == mirrors.number_of_mirrors
    assert 151.0 == pytest.approx(mirrors.diameter)
    assert 3 == mirrors.shape
