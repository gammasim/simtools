#!/usr/bin/python3

import logging
import pytest

import simtools.util.write_model_data as writer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def test_prepare_metadata():

    user_meta_1 = {
        'CONTACT': 'my_name',
        'INSTRUMENT': 'my_instrument'
    }

    file_writer = writer.ModelData()
    file_writer._prepare_metadata(user_meta_1)

    user_meta_2 = {
        'CONTACT': 'my_name'
    }

    with pytest.raises(KeyError):
        file_writer._prepare_metadata(user_meta_2)



