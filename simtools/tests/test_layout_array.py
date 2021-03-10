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
    layout.printTelescopeList()


def test_add_tel():
    layout = LayoutArray(name='testLayout')
    telFile = io.getTestDataFile('telescope_positions_prod5_north.ecsv')
    layout.readTelescopeListFile(telFile)
    layout.addTelescope(telescopeName='L-05', posX=100 * u.m, posY=100 * u.m, posZ=100 * u.m)

    layout.printTelescopeList()


def test_build_layout():
    parameters = {
        'centerLongitude': -17.8920302 * u.deg,
        'centerLatitude': 28.7621661 * u.deg,
        'epsg': 32628,
        'centerAltitude': 2177 * u.m,
        'centerNorthing': 3185066.0 * u.m,
        'centerEasting': 217611.0 * u.m,
        'corsikaSphereRadius': {'LST': 12.5 * u.m, 'MST': 9.6 * u.m, 'SST': 3 * u.m},
        'corsikaSphereCenter': {'LST': 16 * u.m, 'MST': 9 * u.m, 'SST': 3.25 * u.m},
        'corsikaObsLevel': 2158 * u.m
    }

    layout = LayoutArray(label='test_layout', name='LST4', **parameters)

    # Adding 4 LST on a regular grid
    layout.addTelescope(telescopeName='L-01', posX=57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName='L-02', posX=-57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName='L-03', posX=57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName='L-04', posX=-57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m)

    layout.convertCoordinates()
    layout.printTelescopeList()
    layout.exportTelescopeList()

    # Building a second layout from the file exported by the first one
    layout_copy = LayoutArray(label='test_layout', name='LST4-copy')
    layout_copy.readTelescopeListFile(layout.telescopeListFile)
    layout_copy.printTelescopeList()
    layout_copy.exportTelescopeList()

    # Comparing both layouts
    for par in ['_centerEasting', '_centerLatitude', '_epsg', '_corsikaObsLevel']:
        assert layout.__dict__[par] == layout_copy.__dict__[par]

    assert layout.getNumberOfTelescopes() == layout_copy.getNumberOfTelescopes()

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
