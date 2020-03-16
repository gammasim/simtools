#!/usr/bin/python3

import logging

from simtools.corsika_config import CorsikaConfig
from simtools.util import config as cfg

logging.getLogger().setLevel(logging.DEBUG)

config = cfg.loadConfig()  # config dict

if __name__ == '__main__':

    cc = CorsikaConfig(
        site='Paranal',
        arrayName='4LST',
        databaseLocation=config['databaseLocation'],
        nshow=100,
        wrong_par=200,
        zenith=20,
        viewcone=5,
        label='test-corsika-config'
    )
    cc.exportFile()

    cc2 = CorsikaConfig(
        site='LaPalma',
        arrayName='1SST',
        databaseLocation=config['databaseLocation'],
        nshow=1000,
        zenith=[0, 60],
        viewcone=[0, 10],
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
        label='test-corsika-config'
    )
    cc3.exportFile()
