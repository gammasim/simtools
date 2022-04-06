#!/usr/bin/python3

import pytest
import logging

import simtools.util.validate_schema as validator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_validate_data_type_datetime():

    date_validator = validator.SchemaValidator(None, None)

    date_key = 'CREATION_TIME'
    date_schema1 = {
        'type': 'datetime',
        'required': True
    }

    # tests should succeed
    date_validator._validate_data_type(
        date_schema1, date_key, '2018-03-01 12:00:00')

    # tests should fail
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


def test_validate_data_type_email():

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

def test_validate_data_type_schema_str():

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

def test_validate_instrument_list():

    date_validator = validator.SchemaValidator(None, None)
    date_validator._reference_schema = get_generic_instrument_reference_schema()

    instrument_1 = {
        'INSTRUMENT': {
            'SITE': 'South',
            'CLASS': 'MST',
            'TYPE': 'FlashCam',
            'SUBTYPE': 'D'
        }
    }
    instrument_list = [instrument_1['INSTRUMENT']]
    date_validator._validate_instrument_list(instrument_list)

    instrument_2 = instrument_1
    del instrument_1['INSTRUMENT']['CLASS']
    instrument_list.append(instrument_2)
    with pytest.raises(
        UnboundLocalError,
        match=r"No data for `SITE` key"):
            date_validator._validate_instrument_list(instrument_list)

def test_check_if_field_is_optional():

    date_validator = validator.SchemaValidator(None, None)

    test_value_1 = {'required': False}
    test_value_2 = {'required': True}

    assert date_validator._field_is_optional(test_value_1) == True
    assert date_validator._field_is_optional(test_value_2) == False

def test_field_is_optional():

    date_validator = validator.SchemaValidator(None, None)

    value_1 = {'required': True}
    value_2 = {'required': False}
    value_3 = {'required': None}

    assert date_validator._field_is_optional(value_1) == False
    assert date_validator._field_is_optional(value_2) == True
    assert date_validator._field_is_optional(value_3) == True
    # TODO understand why no KeyError is raised
    # value_4 = {'wrong_key': True}
    # with pytest.raises(KeyError):
    #    date_validator._field_is_optional(value_4)

def test_get_reference_schema_file():

    _workflow_config = get_generic_workflow_config()

    date_validator = validator.SchemaValidator(None, None)

    test_string = date_validator._get_reference_schema_file(
        _workflow_config)

    assert test_string == "directory/schema"

    del _workflow_config["CTASIMPIPE"]["DATAMODEL"]["USERINPUTSCHEMA"]
    with pytest.raises(KeyError, match=r"USERINPUTSCHEMA"):
        date_validator._get_reference_schema_file(
            _workflow_config)


def test_remove_line_feed():

    test_string_1 = "ABCK sdlkfjs sdlkf jsd "
    test_string_2 = "ABCK\nsdlkfjs sdlkf\njsd "
    test_string_3 = "ABCK\rsdlkfjs sdlkf\njsd "
    test_string_4 = "ABCK\tsdlkfjs sdlkf\njsd "

    date_validator = validator.SchemaValidator(None, None)

    string_1 = date_validator._remove_line_feed(test_string_1)
    string_2 = date_validator._remove_line_feed(test_string_2)
    string_3 = date_validator._remove_line_feed(test_string_3)
    string_4 = date_validator._remove_line_feed(test_string_4)

    assert string_1 == test_string_1
    assert string_2 == test_string_1
    assert string_3 == "ABCKsdlkfjs sdlkf jsd "
    assert string_4 == "ABCK\tsdlkfjs sdlkf jsd "


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

def get_generic_instrument_reference_schema():

    return {
        'INSTRUMENT': {
            'SITE': {
                'type': 'str',
                'required': True
            },
            'CLASS': {
                'type': 'str',
                'required': True
            },
            'TYPE': {
                'type': 'str',
                'required': True
            },
            'SUBTYPE': {
                'type': 'str',
                'required': False
            },
            'ID': {
                'type': 'str',
                'required': True
            }
        }
    }

