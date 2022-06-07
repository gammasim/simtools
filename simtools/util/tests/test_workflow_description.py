#!/usr/bin/python3

import logging
import pytest

import simtools.util.workflow_description as workflow

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_fill_product_association_identifier():

    workflow_1 = workflow.WorkflowDescription()
    workflow_1.toplevel_meta = get_generic_toplevel_meta()
    workflow_1.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION'] = \
        get_generic_user_meta()['PRODUCT']['ASSOCIATION']
    workflow_1._fill_product_association_identifier()

    workflow_1.toplevel_meta['CTA']['PRODUCT'].pop('ASSOCIATION')

    with pytest.raises(KeyError):
        workflow_1._fill_product_association_identifier()


def test_read_instrument_name():

    _workflow = workflow.WorkflowDescription()
    _workflow.toplevel_meta = get_generic_toplevel_meta()

    _association_1 = \
        get_generic_user_meta()['PRODUCT']['ASSOCIATION'][0]
    _association_2 = \
        get_generic_user_meta()['PRODUCT']['ASSOCIATION'][1]

    assert (
        _workflow._read_instrument_name(_association_1)
        == 'South-MST-FlashCam-D')
    assert (
        _workflow._read_instrument_name(_association_2)
        == 'North-MST-NectarCam-7')

    _association_3 = \
        get_generic_user_meta()['PRODUCT']['ASSOCIATION'][0]
    _association_3['SITE'] = 'Moon'

    with pytest.raises(ValueError):
        _workflow._read_instrument_name(_association_3)


def test_merge_config_dicts():

    d_low_priority = {
        'REFERENCE': {'VERSION': '0.1.0'},
        'ACTIVITY': {
            'NAME': 'SetParameterFromExternal',
            'DESCRIPTION': 'Set data columns'
        },
        'DATAMODEL': 'model-A',
        'PRODUCT': None
    }

    d_high_priority = {
        'REFERENCE': {'VERSION': '0.2.0'},
        'ACTIVITY': {'NAME': None},
        'PRODUCT': {'DIRECTORY': './'},
        'DATAMODEL': 'model-B'
    }

    _workflow = workflow.WorkflowDescription()
    _workflow._merge_config_dicts(d_low_priority, d_high_priority)

    d_merged = {
        'REFERENCE': {'VERSION': '0.2.0'},
        'ACTIVITY': {
            'NAME': 'SetParameterFromExternal',
            'DESCRIPTION': 'Set data columns'
        },
        'PRODUCT': {'DIRECTORY': './'},
        'DATAMODEL': 'model-B',
    }

    assert d_merged == d_high_priority


def test_fill_activity_meta():

    file_writer_1 = workflow.WorkflowDescription()
    file_writer_1.toplevel_meta = get_generic_toplevel_meta()
    file_writer_1._fill_activity_meta()

    file_writer_2 = workflow.WorkflowDescription()
    file_writer_2.toplevel_meta = get_generic_toplevel_meta()

    del file_writer_2.workflow_config['ACTIVITY']['NAME']
    file_writer_2.workflow_config['ACTIVITY']['NONAME'] = 'workflow_name'

    with pytest.raises(KeyError):
        file_writer_2._fill_activity_meta()


def get_generic_toplevel_meta():
    """
    Return toplevel data model template
    """

    return {
        'CTA': {
            'REFERENCE': {
                'VERSION': '1.0.0'},
            'PRODUCT': {
                'DESCRIPTION': None,
                'CONTEXT': None,
                'CREATION_TIME': None,
                'ID': None,
                'DATA': {
                    'CATEGORY': 'SIM',
                    'LEVEL': 'R0',
                    'ASSOCIATION': None,
                    'TYPE': 'service',
                    'MODEL': {
                        'NAME': 'simpipe-table',
                        'VERSION': '0.1.0',
                        'URL': None},
                },
                'FORMAT': None,
                'ASSOCIATION': [
                    {
                        'SITE': None,
                        'CLASS': None,
                        'TYPE': None,
                        'SUBTYPE': None,
                        'ID': None
                    }
                ]
            },
            'INSTRUMENT': {
                'SITE': None,
                'CLASS': None,
                'TYPE': None,
                'SUBTYPE': None,
                'ID': None
            },
            'PROCESS': {
                'TYPE': None,
                'SUBTYPE': None,
                'ID': None
            },
            'CONTACT': {
                'ORGANIZATION': None,
                'NAME': None,
                'EMAIL': None
            },
            'ACTIVITY': {
                'NAME': None,
                'TYPE': 'software',
                'ID': None,
                'START': None,
                'END': None,
                'SOFTWARE': {
                    'NAME': 'gammasim-tools',
                    'VERSION': None}
            }
        }
    }


def get_generic_user_meta():

    return {
        'CONTACT': 'my_name',
        'INSTRUMENT': 'my_instrument',
        'PRODUCT': {
            'DESCRIPTION': 'my_product',
            'CREATION_TIME': '2050-01-01',
            'ASSOCIATION': [
                {
                    'SITE': 'South',
                    'CLASS': 'MST',
                    'TYPE': 'FlashCam',
                    'SUBTYPE': 'D',
                    'ID:': None
                },
                {
                    'SITE': 'North',
                    'CLASS': 'MST',
                    'TYPE': 'NectarCam',
                    'SUBTYPE': '7',
                    'ID:': None
                }
            ]

        },
        'PROCESS': 'process_description'
    }
