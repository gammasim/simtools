#!/usr/bin/python3

import logging

from simtools.corsika_config import CorsikaConfig
from simtools.util import config as cfg

logging.getLogger().setLevel(logging.DEBUG)

config = cfg.loadConfig()  # config dict


def test_general():
    cc = CorsikaConfig(
        site='Paranal',
        arrayName='4LST',
        databaseLocation=config['databaseLocation'],
        nshow=100,
        nrun=10,
        wrong_par=200,
        zenith=20,
        viewcone=5,
        erange=[0.01, 10],
        eslope=2,
        phi=0,
        cscat=[10, 1500e2, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc.exportFile()

    cc2 = CorsikaConfig(
        site='LaPalma',
        arrayName='1SST',
        databaseLocation=config['databaseLocation'],
        nshow=1000,
        nrun=11,
        zenith=[0, 60],
        viewcone=[0, 10],
        erange=[0.01, 10],
        eslope=2,
        phi=0,
        cscat=[10, 1500e2, 0],
        primary='proton',
        label='test-corsika-config'
    )
    cc2.exportFile()

    cc3 = CorsikaConfig(
        site='LaPalma',
        arrayName='1MST',
        databaseLocation=config['databaseLocation'],
        nshow=10000,
        zenith=[0, 60],
        viewcone=[0, 0],
        erange=[0.01, 10],
        eslope=2,
        phi=0,
        cscat=[10, 1500e2, 0],
        primary='electron',
        label='test-corsika-config'
    )
    # Testing default parameters
    assert cc3._parameters['RUNNR'] == [1]
    assert cc3._parameters['EVTNR'] == [1]
    cc3.exportFile()


if __name__ == '__main__':

    test_general()