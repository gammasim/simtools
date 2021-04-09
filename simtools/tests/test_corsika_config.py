#!/usr/bin/python3

import logging
from astropy import units as u

from simtools.corsika.corsika_config import CorsikaConfig

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_general():
    cc = CorsikaConfig(
        site='Paranal',
        layoutName='4LST',
        nshow=100,
        nrun=10,
        wrong_par=200,
        zenith=20 * u.deg,
        viewcone=5 * u.deg,
        erange=[0.01 * u.GeV, 10 * u.GeV],
        eslope=-2,
        phi=0 * u.deg,
        cscat=[10, 1500 * u.m, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc.printParameters()
    cc.exportInputFile()

    cc2 = CorsikaConfig(
        site='LaPalma',
        layoutName='1SST',
        nshow=1000,
        nrun=11,
        zenith=[0 * u.deg, 60 * u.deg],
        viewcone=[0 * u.deg, 10 * u.deg],
        erange=[0.01 * u.TeV, 10 * u.TeV],
        eslope=-2,
        phi=0 * u.deg,
        cscat=[10, 1500 * u.m, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc2.exportInputFile()

    cc3 = CorsikaConfig(
        site='LaPalma',
        layoutName='1MST',
        nshow=10000,
        zenith=[0 * u.deg, 60 * u.deg],
        viewcone=[0 * u.deg, 0 * u.deg],
        erange=[0.01 * u.TeV, 10 * u.TeV],
        eslope=-2,
        phi=0 * u.deg,
        cscat=[10, 1500 * u.m, 0],
        primary='electron',
        label='test-corsika-config'
    )
    # Testing default parameters
    assert cc3._userParameters['RUNNR'] == [1]
    assert cc3._userParameters['EVTNR'] == [1]
    cc3.exportInputFile()


def test_units():
    cc = CorsikaConfig(
        site='Paranal',
        layoutName='4LST',
        nshow=100,
        nrun=10,
        zenith=0.1 * u.rad,
        viewcone=5 * u.deg,
        erange=[0.01 * u.TeV, 10 * u.TeV],
        eslope=2,
        phi=0 * u.deg,
        cscat=[10, 1500 * u.m, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc.exportInputFile()


def test_running_corsika_externally():
    cc = CorsikaConfig(
        site='South',
        layoutName='4LST',
        nshow=10,
        nrun=10,
        zenith=20 * u.deg,
        viewcone=0 * u.deg,
        erange=[0.01 * u.TeV, 10 * u.TeV],
        eslope=-2,
        phi=0 * u.deg,
        cscat=[1, 400 * u.m, 0],
        primary='gamma',
        label='test-corsika-config'
    )
    cc.exportInputFile()


if __name__ == '__main__':

    # test_general()
    # test_units()
    test_running_corsika_externally()
    pass
