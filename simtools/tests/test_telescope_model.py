#!/usr/bin/python3

import logging
import unittest

from simtools.model.telescope_model import TelescopeModel, InvalidParameter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestTelescopeModel(unittest.TestCase):

    def setUp(self):
        self.label = 'test-telescope-model'
        self.telModel = TelescopeModel(
            telescopeName='North-LST-1',
            modelVersion='Current',
            label='test-telescope-model'
        )

    def test_handling_parameters(self):
        logger.info(
            'Old mirror_reflection_random_angle:{}'.format(
                self.telModel.getParameterValue('mirror_reflection_random_angle')
            )
        )
        logger.info('Changing mirror_reflection_random_angle')
        new_mrra = '0.0080 0 0'
        self.telModel.changeParameter('mirror_reflection_random_angle', new_mrra)
        self.assertEqual(
            self.telModel.getParameterValue('mirror_reflection_random_angle'),
            new_mrra
        )

        logging.info('Adding new_parameter')
        new_par = '23'
        self.telModel.addParameter('new_parameter', new_par)
        self.assertEqual(self.telModel.getParameterValue('new_parameter'), new_par)

        with self.assertRaises(InvalidParameter):
            self.telModel.getParameter('bla_bla')

    def test_flen_type(self):
        flenInfo = self.telModel.getParameter('focal_length')
        logger.info('Focal Length = {}, type = {}'.format(
            flenInfo['Value'],
            flenInfo['Type'])
        )
        self.assertIsInstance(flenInfo['Value'], float)

    def test_cfg_file(self):
        # Exporting
        self.telModel.exportConfigFile()

        logger.info('Config file: {}'.format(self.telModel.getConfigFile()))

        # Importing
        cfgFile = self.telModel.getConfigFile()
        tel = TelescopeModel.fromConfigFile(
            telescopeName='south-sst-d',
            label='test-sst',
            configFileName=cfgFile
        )
        tel.exportConfigFile()


if __name__ == '__main__':
    unittest.main()
