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
    ],
    'derive_mirror_rnda': [
        ['-s', 'North', '-t', 'MST-FlashCam-D', '--model_version', 'prod4', '--mean_d80', '1.4',
         '--sig_d80', '0.16', '--mirror_list', 'mirror_MST_focal_lengths.dat', '--d80_list',
         'mirror_MST_D80.dat', '--rnda', '0.0075', ' --test']
    ],
    'validate_optics': [
        ['-s', 'North', '-t', 'LST-1', '--max_offset', '1.0', '--src_distance', '11', '--zenith',
         '20', '--test']
    ],
    # Camera
    'validate_camera_efficiency': [
        ['-s', 'North', '-t', 'MST-NectarCam-D', '--model_version', 'prod4']
    ],
    'validate_camera_fov': [
        ['-s', 'North', '-t', 'MST-NectarCam-D', '--model_version', 'prod4']
    ],
    # Layout
    'make_regular_arrays': [
        []
    ]

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
        isOutputValid = (out == 0)
        assert isOutputValid


if __name__ == '__main__':
    # unittest.main()

    test_applications('test')
