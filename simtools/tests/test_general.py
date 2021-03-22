#!/usr/bin/python3

import logging
import astropy.units as u

import simtools.io_handler as io
from simtools.util.general import collectArguments, collectDataFromYamlOrDict


logging.getLogger().setLevel(logging.DEBUG)


def test_collect_args():
    ''' Test collectArguments function from util.general '''

    class Dummy():
        ALL_INPUTS = {
            'zenithAngle': {'default': 20, 'unit': u.deg},
            'offAxisAngle': {'default': 0, 'unit': u.deg, 'isList': True},
            'testDict': {'default': None, 'unit': u.deg, 'isDict': True}
        }

        def __init__(self, **kwargs):
            collectArguments(
                self,
                args=['zenithAngle', 'offAxisAngle', 'testDict'],
                allInputs=self.ALL_INPUTS,
                **kwargs
            )
            print(self.__dict__)

    d = Dummy(zenithAngle=20 * u.deg, offAxisAngle=[0 * u.deg, 10 * u.deg])
    d = Dummy(zenithAngle=20 * u.deg, testDict={'test1': 2 * u.deg, 'test2': 3 * u.deg})
    print(d)


def test_collect_dict_data():
    inDict = {
        'k1': 2,
        'k2': 'bla'
    }
    inYaml = io.getTestDataFile('test_collect_dict_data.yml')

    d1 = collectDataFromYamlOrDict(None, inDict)
    assert 'k2' in d1.keys()
    assert d1['k1'] == 2

    d2 = collectDataFromYamlOrDict(inYaml, None)
    assert 'k3' in d2.keys()
    assert d2['k4'] == ['bla', 2]

    d3 = collectDataFromYamlOrDict(inYaml, inDict)
    assert d3 == d2


if __name__ == '__main__':

    test_collect_dict_data()
    # test_collect_args()
    pass
