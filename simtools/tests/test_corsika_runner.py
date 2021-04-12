#!/usr/bin/python3

import logging

from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_arguments_and_script():
    showerConfigData = {
        'corsikaDataDirectory': 'corsika-data',
        'nshow': 100,
        'primary': 'gamma',
        'erange': 10
    }

    cr = CorsikaRunner(
        site='south',
        layoutName='Prod5',
        label='test-corsika-runner',
        showerConfigData=showerConfigData
    )

    cr.getRunScript(run=1)


if __name__ == '__main__':
    test_arguments_and_script()
