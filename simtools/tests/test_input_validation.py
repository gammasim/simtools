#!/usr/bin/python3

import pytest
import logging

import simtools.util.input_validation as validator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test__validate_data_type_datetime():

    date_validator = validator.SchemaValidator(None, None)

    date_key = 'CREATION_TIME'
    date_schema1 = {'type': 'datetime'}

    # tests should succeed
    date_validator._validate_data_type(
        date_schema1, date_key, '2018-03-01 12:00:00')

    date_data2 = '2018-15-01 12:00:00'
    with pytest.raises(
        ValueError,
        match=r"invalid date format. Expected %Y-%m-%d %H:%M:%S; Found 2018-15-01 12:00"):
            date_validator._validate_data_type(date_schema1, date_key, date_data2)
    # tests should fail
    date_data1 = '2018-03-01 12:00'
    with pytest.raises(
        ValueError,
        match=r"invalid date format. Expected %Y-%m-%d %H:%M:%S; Found 2018-03-01 12:00"):
            date_validator._validate_data_type(date_schema1, date_key, date_data1)


def test__validate_data_type_email():

    date_validator = validator.SchemaValidator(None, None)

    email_key = 'EMAIL'
    email_schema1 = {'type': 'email'}

    # tests should suceed
    date_validator._validate_data_type(
        email_schema1, email_key, 'me@blabla.de')

    # tests should fail
    with pytest.raises(
        ValueError,
        match=r"invalid email format in field EMAIL: me-blabla.de"):
            date_validator._validate_data_type(email_schema1, email_key, 'me-blabla.de')

# TODO
# several missing tests
