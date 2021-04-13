#!/usr/bin/python3

import logging

import astropy.units as u

from simtools.shower_simulator import ShowerSimulator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_general():
    showerConfigData = {
        'corsikaDataDirectory': './corsika-data',
        'site': 'South',
        'layoutName': 'Prod5',
        'runs': [3, 4],
        'nshow': 10,
        'primary': 'gamma',
        'erange': [100 * u.GeV, 1 * u.TeV],
        'eslope': -2,
        'zenith': 20 * u.deg,
        'azimuth': 0 * u.deg,
        'viewcone': 0 * u.deg,
        'cscat': [10, 1500 * u.m, 0]
    }

    ss = ShowerSimulator(
        label='test-corsika-runner',
        showerConfigData=showerConfigData
    )

    # ss.submit()

    ss.printListOfOutputFiles(runs=[2, 10, 35])

    # runNumber = 3
    # script = cr.getRunScriptFile(runNumber=runNumber)
    # logger.debug('Script file written in {}'.format(script))

    # logger.debug('Run log file in {}'.format(cr.getRunLogFile(runNumber)))
    # logger.debug('CORSIKA log file in {}'.format(cr.getCorsikaLogFile(runNumber)))
    # logger.debug('CORSIKA output file in {}'.format(cr.getCorsikaOutputFile(runNumber)))


if __name__ == '__main__':
    test_general()
