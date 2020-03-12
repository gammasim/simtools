#!/usr/bin/python3

import logging

from simtools.corsika_config import CorsikaConfig

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':

    cc = CorsikaConfig(site='Paranal', array='4LST', nshow=100, wrong_par=200, zenith=[0, 20])
    cc.exportFile()
