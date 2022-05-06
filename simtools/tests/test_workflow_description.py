#!/usr/bin/python3

import logging
import pytest

import simtools.util.workflow_description as workflow

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def test_read_instrument_name():

    workflow_1 = workflow.WorkflowDescription(
        get_generic_workflow_config(),
        get_generic_toplevel_meta())

    user_meta_1 = get_generic_user_meta()
    workflow_1._user_meta = user_meta_1
    workflow_1._fill_user_meta()

    for association in user_meta_1['PRODUCT']['ASSOCIATION']:
        print(association)
    assert (
        file_workflow._read_instrument_name(user_meta_1['PRODUCT']['ASSOCIATION'][0])
        == "South-MST-FlashCam-D")
    assert (
        file_workflow._read_instrument_name(user_meta_1['PRODUCT']['ASSOCIATION'][1])
        == "North-MST-NectarCam-7")

    user_meta_2 = get_generic_user_meta()
    user_meta_2["PRODUCT"]["ASSOCIATION"][0]["TYPE"] = "Structure"
    file_workflow._user_meta = user_meta_2
    file_workflow._fill_user_meta()

    assert (
        file_workflow._read_instrument_name(user_meta_2["PRODUCT"]["ASSOCIATION"][0])
        == "South-MST-Structure-D")

    user_meta_3 = get_generic_user_meta()
    user_meta_3["PRODUCT"]["ASSOCIATION"][0]["SITE"] = "Neptun"
    file_workflow._user_meta = user_meta_3
    file_workflow._fill_user_meta()

    with pytest.raises(ValueError):
        file_workflow._read_instrument_name(user_meta_3["PRODUCT"]["ASSOCIATION"][0])


def test_fill_user_meta():

    user_meta_1 = get_generic_user_meta()

    file_writer = workflow.ModelData(
        get_generic_workflow_config(),
        get_generic_toplevel_meta())
    file_workflow._user_meta = user_meta_1
    file_workflow._fill_user_meta()

    user_meta_2 = {
        'CONTACT': 'my_name'
    }
    file_workflow._user_meta = user_meta_2

    with pytest.raises(KeyError):
        file_workflow._fill_user_meta()


def test_fill_activity_meta():

    file_writer_1 = workflow.ModelData(
        get_generic_workflow_config(),
        get_generic_toplevel_meta())
    file_writer_1._fill_activity_meta()

    workflow_config_2 = get_generic_workflow_config()
    del workflow_config_2['CTASIMPIPE']['ACTIVITY']['NAME']
    workflow_config_2['CTASIMPIPE']['ACTIVITY']['NONAME'] = 'workflow_name'
    file_writer_2 = workflow.ModelData(
        workflow_config_2,
        get_generic_toplevel_meta())

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


def get_generic_workflow_config():

    return {
        'CTASIMPIPE': {
            'ACTIVITY': {
                'NAME': 'workflow_name'
            },
            'DATAMODEL': {
                'USERINPUTSCHEMA': 'schema',
                'TOPLEVELMODEL': 'model',
                'SCHEMADIRECTORY': 'directory'
            },
            'PRODUCT': {
                'DIRECTORY': None
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
