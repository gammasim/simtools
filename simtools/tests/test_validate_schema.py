#!/usr/bin/python3

import pytest
import logging

import simtools.util.validate_schema as validator

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

def test__validate_data_type_schema_str():

    date_validator = validator.SchemaValidator(None, None)
    test_key = 'SUBTYPE'
    test_schema_1 = {'type': 'str'}
    date_validator._validate_data_type(
        test_schema_1, test_key, 'test_string')
    date_validator._validate_data_type(
        test_schema_1, test_key, 25)

def test__validate_data_type_schema_float():

    date_validator = validator.SchemaValidator(None, None)
    test_key = 'SUBTYPE'
    test_schema_2 = {'type': 'float'}

    date_validator._validate_data_type(
        test_schema_2, test_key, 25.0)
    date_validator._validate_data_type(
        test_schema_2, test_key, 25)

    with pytest.raises(
        ValueError,
        match=r"invalid type for key SUBTYPE. Expected: float, Found: str"):
            date_validator._validate_data_type(test_schema_2, test_key, 'abc')

def test__validate_data_type_schema_bool():

    date_validator = validator.SchemaValidator(None, None)
    test_key = 'SUBTYPE'
    test_schema_4 = {'type': 'bool'}

    date_validator._validate_data_type(
        test_schema_4, test_key, False)
    date_validator._validate_data_type(
        test_schema_4, test_key, True)

def test__validate_data_type_schema_int():

    date_validator = validator.SchemaValidator(None, None)
    test_key = 'SUBTYPE'
    test_schema_3 = {'type': 'int'}

    date_validator._validate_data_type(
        test_schema_3, test_key, 25)

    with pytest.raises(
        ValueError,
        match=r"invalid type for key SUBTYPE. Expected: int, Found: str"):
            date_validator._validate_data_type(test_schema_3, test_key, 'abc')
    with pytest.raises(
        ValueError,
        match=r"invalid type for key SUBTYPE. Expected: int, Found: float"):
            date_validator._validate_data_type(test_schema_3, test_key, 25.5)

def test__check_if_field_is_optional():

    date_validator = validator.SchemaValidator(None, None)

    test_key = 'test_key'
    test_value_1 = {'required': False}
    test_value_2 = {'required': True}

    date_validator._check_if_field_is_optional(test_key, test_value_1)

    # tests should fail
    with pytest.raises(
        ValueError,
        match=r"required data field test_key not found"):
            date_validator._check_if_field_is_optional(test_key, test_value_2)

# TODO
# tests for _validate_schema
