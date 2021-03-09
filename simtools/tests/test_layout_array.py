#!/usr/bin/python3

import logging

import astropy.units as u

import simtools.io_handler as io
from simtools.layout.layout_array import LayoutArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_read_tel_list():
    layout = LayoutArray(name='testLayout')
    telFile = io.getTestDataFile('telescope_positions_prod5_north.ecsv')
    layout.readTelescopeListFile(telFile)
    layout.convertCoordinates()


def test_dict_input():
    layout = LayoutArray(
        name='testLayout',
        corsikaSphereRadius={'LST': 1 * u.m, 'MST': 1 * u.m, 'SST': 1 * u.m}
    )
    print(layout.__dict__)
    # telFile = io.getTestDataFile('telescope_positions_prod5_north.ecsv')
    # layout.readTelescopeList(telFile)
    # layout.convertCoordinates()


def test_add_tel():
    layout = LayoutArray(name='testLayout')
    telFile = io.getTestDataFile('telescope_positions_prod5_north.ecsv')
    layout.readTelescopeListFile(telFile)
    # layout.addTelescope(telescopeName='bla')

    layout.printTelescopeList()

    # layout.convertCoordinates()


# def test_export_tel_list_file():
#     layout = LayoutArray(name='testLayout')
#     telFile = io.getTestDataFile('telescope_positions_prod5_north.ecsv')
#     layout.readTelescopeListFile(telFile)
#     layout.convertCoordinates()


if __name__ == '__main__':

    # test_read_tel_list()
    test_add_tel()
    # test_dict_input()
    pass
