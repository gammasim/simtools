#!/usr/bin/python3

import logging

import simtools.io_handler as io
from simtools.layout.layout_array import LayoutArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_tel_list():
    layout = LayoutArray(name='testLayout')
    telFile = io.getTestDataFile('telescope_positions_prod5_north.ecsv')
    layout.readTelescopeList(telFile)
    layout.convertCoordinates()


if __name__ == '__main__':

    test_read_tel_list()
    pass
