#!/usr/bin/python3

import logging

from ctamclib.telescope_model import TelescopeModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_input_validation():
    yamlDBPath = ('/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/'
                  'configReports')

    telType = 'lst'
    site = 'south'
    logger.info('Input telType: {}'.format(telType))
    logger.info('Input site: {}'.format(site))

    tel = TelescopeModel(yamlDBPath=yamlDBPath, telescopeType=telType, site=site, version='prod4',
                         label='test-lst')

    logger.info('Validated telType: {}'.format(tel.telescopeType))
    logger.info('Validated site: {}'.format(tel.site))


def test_handling_parameters():

    yamlDBPath = ('/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/'
                  'configReports')

    tel = TelescopeModel(yamlDBPath=yamlDBPath, telescopeType='lst', site='south', version='prod4',
                         label='test-lst')

    logger.info('Old mirror_reflection_random_angle:{}'.format(tel.getParameter('mirror_reflection_'
                                                                                'random_angle')))
    logger.info('Changing mirror_reflection_random_angle')
    new_mrra = '0.0080 0 0'
    tel.changeParameters(mirror_reflection_random_angle=new_mrra)
    assert tel.getParameter('mirror_reflection_random_angle') == new_mrra

    logger.info('Adding new_parameter')
    new_par = '23'
    tel.addParameters(new_parameter=new_par)
    assert tel.getParameter('new_parameter') == new_par


def test_flen_type():
    yamlDBPath = ('/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/'
                  'configReports')

    tel = TelescopeModel(yamlDBPath=yamlDBPath, telescopeType='lst', site='south', version='prod4',
                         label='test-lst')

    flen = tel.getParameter('focal_length')
    logger.info('Focal Length = {}, type = {}'.format(flen, type(flen)))
    assert type(flen) == float


def test_cfg_file():
    # Exporting
    yamlDBPath = ('/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/'
                  'configReports')

    tel = TelescopeModel(yamlDBPath=yamlDBPath, telescopeType='lst', site='south', version='prod4',
                         label='test-lst')
    # tel.exportConfigFile(loc='/home/prado/Work/Projects/CTA_MC/MCLib')
    tel.exportConfigFile()

    logger.info('Config file: ', tel.getConfigFile())

    # Importing

    cfgFile = tel.getConfigFile()
    tel = TelescopeModel.fromConfigFile(telescopeType='astri', site='south', label='test-astri',
                                        configFileName=cfgFile)

    tel.exportConfigFile()


if __name__ == '__main__':

    test_handling_parameters()
    test_input_validation()
    test_flen_type()
    test_cfg_file()
