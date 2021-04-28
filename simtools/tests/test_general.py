#!/usr/bin/python3

import logging
import astropy.units as u

import simtools.io_handler as io
import simtools.util.general as gen


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
            gen.collectArguments(
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

    d1 = gen.collectDataFromYamlOrDict(None, inDict)
    assert 'k2' in d1.keys()
    assert d1['k1'] == 2

    d2 = gen.collectDataFromYamlOrDict(inYaml, None)
    assert 'k3' in d2.keys()
    assert d2['k4'] == ['bla', 2]

    d3 = gen.collectDataFromYamlOrDict(inYaml, inDict)
    assert d3 == d2

def test_validate_config_data():

    parameterFile = io.getTestDataFile('test_parameters.yml')

    parameters = gen.collectDataFromYamlOrDict(parameterFile, None)

    configData = {
        'zenith': 0 * u.deg,
        'offaxis': [0 * u.deg, 0.2 * u.rad],
        'cscat': [0, 10 * u.m, 3 * u.km],
        'testName': 10
    }

    validatedData = gen.validateConfigData(configData=configData, parameters=parameters)

    assert 'validatedName' in validatedData.keys()

    print(validatedData)

    # configData1 = {
    #     'zenith': 0 * u.deg,
    #     'offaxis': [0, 20] * u.deg
    # }

    # validatedData1 = gen.validateConfigData(configData=configData1, parameters=parameters)

    # print(validatedData1)


if __name__ == '__main__':

    # test_collect_dict_data()
    # test_collect_args()
    test_validate_config_data()
