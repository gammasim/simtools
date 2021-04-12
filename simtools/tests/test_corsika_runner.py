#!/usr/bin/python3

import logging

import astropy.units as u

from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_arguments_and_script():
    showerConfigData = {
        'corsikaDataDirectory': './corsika-data',
        'nshow': 10,
        'primary': 'gamma',
        'erange': [100 * u.GeV, 1 * u.TeV],
        'eslope': -2,
        'zenith': 20 * u.deg,
        'azimuth': 0 * u.deg,
        'viewcone': 0 * u.deg,
        'cscat': [10, 1500 * u.m, 0]
    }

    cr = CorsikaRunner(
        site='south',
        layoutName='Prod5',
        label='test-corsika-runner',
        showerConfigData=showerConfigData
    )

    runNumber = 3
    script = cr.getRunScriptFile(runNumber=runNumber)
    logger.debug('Script file written in {}'.format(script))

    logger.debug('Run log file in {}'.format(cr.getRunLogFile(runNumber)))
    logger.debug('CORSIKA log file in {}'.format(cr.getCorsikaLogFile(runNumber)))
    logger.debug('CORSIKA output file in {}'.format(cr.getCorsikaOutputFile(runNumber)))


if __name__ == '__main__':
    test_arguments_and_script()
