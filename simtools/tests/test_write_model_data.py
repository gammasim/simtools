#!/usr/bin/python3

import logging
import pytest

import simtools.util.write_model_data as writer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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
            }
        }
    }

def ffftest_fill_user_meta():

    user_meta_1 = {
        'CONTACT': 'my_name',
        'INSTRUMENT': 'my_instrument',
        'PRODUCT': {
            'DESCRIPTION': 'my_product',
            'CREATION_TIME': '2050-01-01'
        },
        'PROCESS': 'process_description'
    }

    file_writer = writer.ModelData()
    file_writer._user_meta = user_meta_1
    file_writer._fill_user_meta()

    user_meta_2 = {
        'CONTACT': 'my_name'
    }
    file_writer._user_meta = user_meta_2

    with pytest.raises(KeyError):
        file_writer._fill_user_meta()


def test_fill_activity_meta():

    workflow_config_1 = get_generic_workflow_config()
    file_writer_1 = writer.ModelData(workflow_config_1)
    file_writer_1._fill_activity_meta()

    workflow_config_2 = get_generic_workflow_config()
    workflow_config_2['CTASIMPIPE']['ACTIVITY']['NONAME'] = 'workflow_name'
    file_writer_2 = writer.ModelData(workflow_config_2)

    with pytest.raises(KeyError):
        file_writer_2._fill_activity_meta()
