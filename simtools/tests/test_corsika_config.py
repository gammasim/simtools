#!/usr/bin/python3

import logging
from astropy import units as u

import simtools.config as cfg
# from simtools.corsika_config import CorsikaConfig

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# def test_general():
#     cc = CorsikaConfig(
#         site='Paranal',
#         arrayName='4LST',
#         nshow=100,
#         nrun=10,
#         wrong_par=200,
#         zenith=20 * u.deg,
#         viewcone=5 * u.deg,
#         erange=[0.01 * u.GeV, 10 * u.GeV],
#         eslope=2,
#         phi=0 * u.deg,
#         cscat=[10, 1500 * u.m, 0],
#         primary='proton',
#         label='test-corsika-config',
#         logger=logger.name
#     )
#     cc.exportFile()

#     cc2 = CorsikaConfig(
#         site='LaPalma',
#         arrayName='1SST',
#         nshow=1000,
#         nrun=11,
#         zenith=[0 * u.deg, 60 * u.deg],
#         viewcone=[0 * u.deg, 10 * u.deg],
#         erange=[0.01 * u.TeV, 10 * u.TeV],
#         eslope=2,
#         phi=0 * u.deg,
#         cscat=[10, 1500 * u.m, 0],
#         primary='proton',
#         label='test-corsika-config',
#         logger=logger.name
#     )
#     cc2.exportFile()

#     cc3 = CorsikaConfig(
#         site='LaPalma',
#         arrayName='1MST',
#         nshow=10000,
#         zenith=[0 * u.deg, 60 * u.deg],
#         viewcone=[0 * u.deg, 0 * u.deg],
#         erange=[0.01 * u.TeV, 10 * u.TeV],
#         eslope=2,
#         phi=0 * u.deg,
#         cscat=[10, 1500 * u.m, 0],
#         primary='electron',
#         label='test-corsika-config',
#         logger=logger.name
#     )
#     # Testing default parameters
#     assert cc3._parameters['RUNNR'] == [1]
#     assert cc3._parameters['EVTNR'] == [1]
#     cc3.exportFile()


# def test_units():
#     cc = CorsikaConfig(
#         site='Paranal',
#         arrayName='4LST',
#         nshow=100,
#         nrun=10,
#         zenith=0.1 * u.rad,
#         viewcone=5 * u.deg,
#         erange=[0.01 * u.TeV, 10 * u.TeV],
#         eslope=2,
#         phi=0 * u.deg,
#         cscat=[10, 1500 * u.m, 0],
#         primary='proton',
#         label='test-corsika-config',
#         logger=logger.name
#     )
#     cc.exportFile()


if __name__ == '__main__':

    test_general()
    test_units()
    pass
