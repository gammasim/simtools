#!/usr/bin/python3

import logging

import simtools.io_handler as io
from simtools.model.mirrors import Mirrors

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_list():
    mirrorListFile = io.getTestDataFile("mirror_CTA-LST-flen_grouped.dat")
    logger.info("Using mirror list {}".format(mirrorListFile))
    mirrors = Mirrors(mirrorListFile)
    logger.info("Number of Mirrors = {}".format(mirrors.numberOfMirrors))
    logger.info("Mirrors Diameter in cm = {}".format(mirrors.diameter))
    logger.info("Mirrors Shape = {}".format(mirrors.shape))
    return


if __name__ == "__main__":

    test_read_list()
