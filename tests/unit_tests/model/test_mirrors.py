#!/usr/bin/python3

import logging

import pytest

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.model.mirrors import Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


def test_read_list(db):

    testFileName = "mirror_CTA-LST-flen_grouped.dat"
    db.exportFileDB(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        dest=io.getOutputDirectory(dirType="model", test=True),
        fileName=testFileName,
    )
    mirrorListFile = cfg.findFile(testFileName, io.getOutputDirectory(dirType="model", test=True))
    logger.info("Using mirror list {}".format(mirrorListFile))
    mirrors = Mirrors(mirrorListFile)
    assert 198 == mirrors.numberOfMirrors
    assert 151.0 == pytest.approx(mirrors.diameter)
    assert 3 == mirrors.shape
