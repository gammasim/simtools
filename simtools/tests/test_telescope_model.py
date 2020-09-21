#!/usr/bin/python3

import yaml
import logging

from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input_validation():
    telName = 'north-lst-1'
    logger.info('Input telName: {}'.format(telName))

    tel = TelescopeModel(
        telescopeName=telName,
        version='current',
        label='test-lst',
        logger=logger.name
    )

    logger.info('Validated telName: {}'.format(tel.telescopeName))
    return


def test_handling_parameters():
    tel = TelescopeModel(
        telescopeName='north-lst-1',
        version='current',
        label='test-lst',
        logger=logger.name
    )

    logger.info(
        'Old mirror_reflection_random_angle:{}'.format(
            tel.getParameter('mirror_reflection_random_angle')
        )
    )
    logger.info('Changing mirror_reflection_random_angle')
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
        telescopeName='north-lst-1',
        version='current',
        label='test-lst',
        logger=logger.name
    )
    flen = tel.getParameter('focal_length')
    logger.info('Focal Length = {}, type = {}'.format(flen, type(flen)))
    assert type(flen) == float
    return


def test_cfg_file():
    # Exporting
    tel = TelescopeModel(
        telescopeName='south-sst-d',
        version='current',
        label='test-sst',
        logger=logger.name
    )
    # tel.exportConfigFile(loc='/home/prado/Work/Projects/CTA_MC/MCLib')
    tel.exportConfigFile()

    logger.info('Config file: {}'.format(tel.getConfigFile()))

    # Importing
    cfgFile = tel.getConfigFile()
    tel = TelescopeModel.fromConfigFile(
        telescopeName='south-sst-d',
        label='test-sst',
        configFileName=cfgFile,
        logger=logger.name
    )
    tel.exportConfigFile()
    return


def test_cfg_input():
    tel = TelescopeModel(
        telescopeName='north-lst-1',
        version='current',
        label='test-sst-2',
        logger=logger.name
    )
    return


if __name__ == '__main__':

    # test_handling_parameters()
    # test_input_validation()
    # test_flen_type()
    # test_cfg_file()
    # test_cfg_input()
    test_pars_from_db_handler()
    pass
