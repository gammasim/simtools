#!/usr/bin/python3

import yaml
import logging

from simtools.util import config as cfg
from simtools.telescope_model import TelescopeModel

logging.getLogger().setLevel(logging.DEBUG)


def test_input_validation():
    telType = 'lst'
    site = 'south'
    logging.info('Input telType: {}'.format(telType))
    logging.info('Input site: {}'.format(site))

    tel = TelescopeModel(
        telescopeType=telType,
        site=site,
        version='prod4',
        label='test-lst'
    )

    logging.info('Validated telType: {}'.format(tel.telescopeType))
    logging.info('Validated site: {}'.format(tel.site))
    return


def test_handling_parameters():
    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-lst'
    )

    logging.info(
        'Old mirror_reflection_random_angle:{}'.format(
            tel.getParameter('mirror_reflection_random_angle')
        )
    )
    logging.info('Changing mirror_reflection_random_angle')
    new_mrra = '0.0080 0 0'
    tel.changeParameters(mirror_reflection_random_angle=new_mrra)
    assert tel.getParameter('mirror_reflection_random_angle') == new_mrra

    logging.info('Adding new_parameter')
    new_par = '23'
    tel.addParameters(new_parameter=new_par)
    assert tel.getParameter('new_parameter') == new_par
    return


def test_flen_type():
    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-lst'
    )
    flen = tel.getParameter('focal_length')
    logging.info('Focal Length = {}, type = {}'.format(flen, type(flen)))
    assert type(flen) == float
    return


def test_cfg_file():
    # Exporting
    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-lst'
    )
    # tel.exportConfigFile(loc='/home/prado/Work/Projects/CTA_MC/MCLib')
    tel.exportConfigFile()

    logging.info('Config file: {}'.format(tel.getConfigFile()))

    # Importing
    cfgFile = tel.getConfigFile()
    tel = TelescopeModel.fromConfigFile(
        telescopeType='astri',
        site='south',
        label='test-astri',
        configFileName=cfgFile
    )
    tel.exportConfigFile()
    return


def test_cfg_input():
    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-input'
    )
    return


if __name__ == '__main__':

    # test_handling_parameters()
    # test_input_validation()
    # test_flen_type()
    # test_cfg_file()
    # test_cfg_input()
    pass
