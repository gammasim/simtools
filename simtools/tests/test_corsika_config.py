#!/usr/bin/python3

import logging
from astropy import units

from simtools.corsika_config import CorsikaConfig
from simtools.util import config as cfg

logging.getLogger().setLevel(logging.DEBUG)


def test_general():
    cc = CorsikaConfig(
        site='Paranal',
        arrayName='4LST',
        nshow=100,
        nrun=10,
        wrong_par=200,
        zenith=20 * units.deg,
        viewcone=5 * units.deg,
        erange=[0.01 * units.GeV, 10 * units.GeV],
        eslope=2,
        phi=0 * units.deg,
        cscat=[10, 1500 * units.m, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc.exportFile()

    cc2 = CorsikaConfig(
        site='LaPalma',
        arrayName='1SST',
        nshow=1000,
        nrun=11,
        zenith=[0 * units.deg, 60 * units.deg],
        viewcone=[0 * units.deg, 10 * units.deg],
        erange=[0.01 * units.TeV, 10 * units.TeV],
        eslope=2,
        phi=0 * units.deg,
        cscat=[10, 1500 * units.m, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc2.exportFile()

    cc3 = CorsikaConfig(
        site='LaPalma',
        arrayName='1MST',
        nshow=10000,
        zenith=[0 * units.deg, 60 * units.deg],
        viewcone=[0 * units.deg, 0 * units.deg],
        erange=[0.01 * units.TeV, 10 * units.TeV],
        eslope=2,
        phi=0 * units.deg,
        cscat=[10, 1500  * units.m, 0],
        primary='electron',
        label='test-corsika-config'
    )
    # Testing default parameters
    assert cc3._parameters['RUNNR'] == [1]
    assert cc3._parameters['EVTNR'] == [1]
    cc3.exportFile()


def test_units():
    cc = CorsikaConfig(
        site='Paranal',
        arrayName='4LST',
        nshow=100,
        nrun=10,
        zenith=0.1 * units.rad,
        viewcone=5 * units.deg,
        erange=[0.01 * units.TeV, 10 * units.TeV],
        eslope=2,
        phi=0 * units.deg,
        cscat=[10, 1500 * units.m, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc.exportFile()


if __name__ == '__main__':

    # test_general()
    # test_units()
    pass
