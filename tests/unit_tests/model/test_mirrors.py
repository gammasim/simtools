#!/usr/bin/python3

import logging

import pytest

import simtools.util.general as gen
from simtools.model.mirrors import Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_list(db, io_handler):

    testFileName = "mirror_CTA-LST-flen_grouped.dat"
    db.export_file_db(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        dest=io_handler.get_output_directory(dirType="model", test=True),
        fileName=testFileName,
    )
    mirrorListFile = gen.find_file(
        testFileName, io_handler.get_output_directory(dirType="model", test=True)
    )
    logger.info("Using mirror list {}".format(mirrorListFile))
    mirrors = Mirrors(mirrorListFile)
    assert 198 == mirrors.numberOfMirrors
    assert 151.0 == pytest.approx(mirrors.diameter)
    assert 3 == mirrors.shape
