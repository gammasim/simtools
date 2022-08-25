#!/usr/bin/python3

import logging

import pytest

import simtools.util.validate_schema as validator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_validate_data_type_datetime():

    date_validator = validator.SchemaValidator()

    date_key = "CREATION_TIME"
    date_schema1 = {"type": "datetime", "required": True}

    # tests should succeed
    date_validator._validate_data_type(date_schema1, date_key, "2018-03-01 12:00:00")

    # tests should fail
    date_data2 = "2018-15-01 12:00:00"
    with pytest.raises(
        ValueError, match=r"invalid date format. Expected %Y-%m-%d %H:%M:%S; Found 2018-15-01 12:00"
    ):
        date_validator._validate_data_type(date_schema1, date_key, date_data2)
    # tests should fail
    date_data1 = "2018-03-01 12:00"
    with pytest.raises(
        ValueError, match=r"invalid date format. Expected %Y-%m-%d %H:%M:%S; Found 2018-03-01 12:00"
    ):
        date_validator._validate_data_type(date_schema1, date_key, date_data1)


def test_validate_data_type_email():

    date_validator = validator.SchemaValidator()

    email_key = "EMAIL"
    email_schema1 = {"type": "email"}

    # tests should suceed
    date_validator._validate_data_type(email_schema1, email_key, "me@blabla.de")

    # tests should fail
    with pytest.raises(ValueError, match=r"invalid email format in field EMAIL: me-blabla.de"):
        date_validator._validate_data_type(email_schema1, email_key, "me-blabla.de")


def test_validate_data_type_schema_str():

    date_validator = validator.SchemaValidator()
    test_key = "SUBTYPE"
    test_schema_1 = {"type": "str"}
    date_validator._validate_data_type(test_schema_1, test_key, "test_string")
    date_validator._validate_data_type(test_schema_1, test_key, 25)


def test_validate_data_type_schema_float():

    date_validator = validator.SchemaValidator()
    test_key = "SUBTYPE"
    test_schema_2 = {"type": "float"}

    date_validator._validate_data_type(test_schema_2, test_key, 25.0)
    date_validator._validate_data_type(test_schema_2, test_key, 25)

    with pytest.raises(
        ValueError, match=r"invalid type for key SUBTYPE. Expected: float, Found: str"
    ):
        date_validator._validate_data_type(test_schema_2, test_key, "abc")


def test_validate_data_type_schema_bool():

    date_validator = validator.SchemaValidator()
    test_key = "SUBTYPE"
    test_schema_4 = {"type": "bool"}

    date_validator._validate_data_type(test_schema_4, test_key, False)
    date_validator._validate_data_type(test_schema_4, test_key, True)


def test_validate_data_type_schema_int():

    date_validator = validator.SchemaValidator()
    test_key = "SUBTYPE"
    test_schema_3 = {"type": "int"}

    date_validator._validate_data_type(test_schema_3, test_key, 25)

    with pytest.raises(
        ValueError, match=r"invalid type for key SUBTYPE. Expected: int, Found: str"
    ):
        date_validator._validate_data_type(test_schema_3, test_key, "abc")
    with pytest.raises(
        ValueError, match=r"invalid type for key SUBTYPE. Expected: int, Found: float"
    ):
        date_validator._validate_data_type(test_schema_3, test_key, 25.5)


def test_validate_schema():

    date_validator = validator.SchemaValidator()
    reference_schema = get_generic_instrument_reference_schema()
    test_schema_1 = get_instrument_test_schema()
    date_validator._validate_schema(reference_schema, test_schema_1)

    test_schema_2 = get_instrument_test_schema()
    test_schema_2["INSTRUMENT"].pop("CLASS")
    with pytest.raises(ValueError, match=r"Missing required field CLASS"):
        date_validator._validate_schema(reference_schema, test_schema_2)

    reference_schema_2 = get_generic_instrument_reference_schema()
    _telid = {"type": int, "required": False}
    reference_schema_2["INSTRUMENT"]["TELID"] = _telid
    test_schema_3 = get_instrument_test_schema()
    test_schema_3["INSTRUMENT"]["TELID"] = 5.5
    with pytest.raises(ValueError):
        date_validator._validate_schema(reference_schema_2, test_schema_3)


def test_validate_instrument_list():

    date_validator = validator.SchemaValidator()
    date_validator._reference_schema = get_generic_instrument_reference_schema()

    instrument_1 = {
        "INSTRUMENT": {
            "SITE": "South",
            "CLASS": "MST",
            "TYPE": "FlashCam",
            "SUBTYPE": "D",
            "ID": "A",
        }
    }
    instrument_list = [instrument_1["INSTRUMENT"]]
    date_validator._validate_instrument_list(instrument_list)

    del instrument_1["INSTRUMENT"]["CLASS"]
    instrument_list.append(instrument_1)
    with pytest.raises(ValueError, match=r"Missing required field CLASS"):
        date_validator._validate_instrument_list(instrument_list)


def test_check_if_field_is_optional():

    date_validator = validator.SchemaValidator()

    test_value_1 = {"required": False}
    test_value_2 = {"required": True}
    test_value_3 = {"required": None}

    assert date_validator._field_is_optional(test_value_1) == True
    assert date_validator._field_is_optional(test_value_2) == False
    assert date_validator._field_is_optional(test_value_3) == True


def test_remove_line_feed():

    test_string_1 = "ABCK sdlkfjs sdlkf jsd "
    test_string_2 = "ABCK\nsdlkfjs sdlkf\njsd "
    test_string_3 = "ABCK\rsdlkfjs sdlkf\njsd "
    test_string_4 = "ABCK\tsdlkfjs sdlkf\njsd "

    date_validator = validator.SchemaValidator()

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
        "CTASIMPIPE": {
            "ACTIVITY": {"NAME": "workflow_name"},
            "DATAMODEL": {
                "USERINPUTSCHEMA": "schema",
                "TOPLEVELMODEL": "model",
                "SCHEMADIRECTORY": "directory",
            },
            "PRODUCT": {"DIRECTORY": None},
        }
    }


def get_instrument_test_schema():

    return {
        "INSTRUMENT": {
            "SITE": "north",
            "CLASS": "camera",
            "TYPE": "lst",
            "SUBTYPE": "subtype",
            "ID": "id",
            "TELID": 5.5,
        }
    }


def get_generic_instrument_reference_schema():

    return {
        "INSTRUMENT": {
            "SITE": {"type": "str", "required": True},
            "CLASS": {"type": "str", "required": True},
            "TYPE": {"type": "str", "required": True},
            "SUBTYPE": {"type": "str", "required": False},
            "ID": {"type": "str", "required": True},
        }
    }
