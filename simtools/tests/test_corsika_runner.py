#!/usr/bin/python3

import logging

import astropy.units as u

from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_arguments_and_script():
    showerConfigData = {
        'corsikaDataDirectory': './corsika-data',
        'nshow': 100,
        'primary': 'gamma',
        'erange': [100 * u.GeV, 10 * u.TeV],
        'eslope': -2,
        'zenith': 20 * u.deg,
        'azimuth': 0 * u.deg,
        'viewcone': 0 * u.deg,
        'cscat': [10, 1000 * u.m, 0]
    }

    cr = CorsikaRunner(
        site='south',
        layoutName='Prod5',
        label='test-corsika-runner',
        showerConfigData=showerConfigData
    )

    script = cr.getRunScriptFile(run=1)
    print(script)


if __name__ == '__main__':
    test_arguments_and_script()
