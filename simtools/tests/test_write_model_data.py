#!/usr/bin/python3

import logging
import pytest

import simtools.util.write_model_data as writer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def test_fill_user_meta():

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

    workflow_config_1 = {
        'CTASIMPIPE': {
            'ACTIVITY': {
                'NAME': 'workflow_name'
            }
        }
    }

    file_writer = writer.ModelData()
    file_writer._workflow_config = workflow_config_1
    file_writer._fill_activity_meta()

    workflow_config_2 = {
        'CTASIMPIPE': {
            'ACTIVITY': {
                'NONAME': 'workflow_name'
            }
        }
    }
    file_writer._workflow_config = workflow_config_2

    with pytest.raises(KeyError):
        file_writer._fill_activity_meta()
