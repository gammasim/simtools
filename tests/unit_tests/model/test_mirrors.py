#!/usr/bin/python3

import logging
import pytest

import simtools.io_handler as io
from simtools.model.mirrors import Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_list(cfg_setup):

    mirrorListFile = io.getTestDataFile("mirror_CTA-LST-flen_grouped.dat")
    logger.info("Using mirror list {}".format(mirrorListFile))
    mirrors = Mirrors(mirrorListFile)
    assert 198 == mirrors.numberOfMirrors
    assert 151.0 == pytest.approx(mirrors.diameter)
    assert 3 == mirrors.shape
