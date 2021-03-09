#!/usr/bin/python3

import logging

import astropy.units as u
import pyproj

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
    layout.addTelescope(telescopeName='L-05', posX=100 * u.m, posY=100 * u.m, posZ=100 * u.m)

    layout.printTelescopeList()

    # layout.convertCoordinates()


def test_build_layout():
    layout = LayoutArray(
        name='LST4',
        centerLongitude=-17.8920302 * u.deg,
        centerLatitude=28.7621661 * u.deg,
        epsg=32628,
        centerAltitude=2177 * u.m
    )
    layout.addTelescope(telescopeName='L-01', posX=57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName='L-02', posX=-57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName='L-03', posX=57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName='L-04', posX=-57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m)

    layout.convertCoordinates()
    layout.printTelescopeList()

    layout.exportTelescopeList()

# def test_export_tel_list_file():
#     layout = LayoutArray(name='testLayout')
#     telFile = io.getTestDataFile('telescope_positions_prod5_north.ecsv')
#     layout.readTelescopeListFile(telFile)
#     layout.convertCoordinates()


if __name__ == '__main__':

    # test_read_tel_list()
    # test_add_tel()
    # test_dict_input()
    test_build_layout()
    pass
