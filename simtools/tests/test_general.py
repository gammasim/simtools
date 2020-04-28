#!/usr/bin/python3

import logging
from astropy import units as u

from simtools.util import config as cfg
from simtools.util.general import collectArguments


logging.getLogger().setLevel(logging.DEBUG)


def test_collect_args():
    ''' Test collectArguments function from util.general '''

    class Dummy():
        ALL_INPUTS = {
            'zenithAngle': {'default': 20, 'unit': u.deg},
            'offAxisAngle': {'default': 0, 'unit': u.deg, 'isList': True}
        }

        def __init__(self, **kwargs):
            collectArguments(
                self,
                args=['zenithAngle', 'offAxisAngle'],
                allInputs=self.ALL_INPUTS,
                **kwargs
            )
            print(self.__dict__)

    d = Dummy(zenithAngle=20 * u.deg, offAxisAngle=[0 * u.deg, 10 * u.deg])
    d = Dummy(zenithAngle=20 * u.deg)


if __name__ == '__main__':

    test_collect_args()
    pass
