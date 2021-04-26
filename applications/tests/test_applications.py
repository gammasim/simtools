#!/usr/bin/python3

import logging
import os
import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


APP_LIST = {
    # Optics
    'compare_cumulative_psf': [
        ['-s', 'North', '-t', 'LST-1', '--model_version', 'prod4', '--data', 'PSFcurve_data_v2.txt',
         '--zenith', '20', '--test']
    ]


    # 'validate_camera_efficiency': [
    #     ['-s', 'North', '-t', 'MST-NectarCam-D']
    # ],
    # 'validate_camera_fov': ['--bla']

}


@pytest.mark.parametrize('application', APP_LIST.keys())
def test_applications(application):
    logger.info('Testing {}'.format(application))

    def makeCommand(app, args):
        cmd = 'python applications/' + app + '.py'
        for aa in args:
            cmd += ' ' + aa
        return cmd

    for args in APP_LIST[application]:
        logger.info('Running with args: {}'.format(args))
        cmd = makeCommand(application, args)
        out = os.system(cmd)
        assert out == 0


if __name__ == '__main__':
    # unittest.main()

    test_applications('test')
