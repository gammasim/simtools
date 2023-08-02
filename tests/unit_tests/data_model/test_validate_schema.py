#!/usr/bin/python3

import logging

import pytest

import simtools.data_model.validate_schema as validator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_validate_data_type_datetime():

    date_validator = validator.SchemaValidator()

    date_key = "creation_time"
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

    email_key = "email"
    email_schema1 = {"type": "email"}

    # tests should suceed
    date_validator._validate_data_type(email_schema1, email_key, "me@blabla.de")

    # tests should fail
    with pytest.raises(ValueError, match=r"invalid email format in field email: me-blabla.de"):
        date_validator._validate_data_type(email_schema1, email_key, "me-blabla.de")


def test_validate_data_type_schema_str():

    date_validator = validator.SchemaValidator()
    test_key = "subtype"
    test_schema_1 = {"type": "str"}
    date_validator._validate_data_type(test_schema_1, test_key, "test_string")
    date_validator._validate_data_type(test_schema_1, test_key, 25)


def test_validate_data_type_schema_float():

    date_validator = validator.SchemaValidator()
    test_key = "subtype"
    test_schema_2 = {"type": "float"}

    date_validator._validate_data_type(test_schema_2, test_key, 25.0)
    date_validator._validate_data_type(test_schema_2, test_key, 25)

    with pytest.raises(
        ValueError, match=r"invalid type for key subtype. Expected: float, Found: str"
    ):
        date_validator._validate_data_type(test_schema_2, test_key, "abc")


def test_validate_data_type_schema_bool():

    date_validator = validator.SchemaValidator()
    test_key = "subtype"
    test_schema_4 = {"type": "bool"}

    date_validator._validate_data_type(test_schema_4, test_key, False)
    date_validator._validate_data_type(test_schema_4, test_key, True)


def test_validate_data_type_schema_int():

    date_validator = validator.SchemaValidator()
    test_key = "subtype"
    test_schema_3 = {"type": "int"}

    date_validator._validate_data_type(test_schema_3, test_key, 25)

    with pytest.raises(
        ValueError, match=r"invalid type for key subtype. Expected: int, Found: str"
    ):
        date_validator._validate_data_type(test_schema_3, test_key, "abc")
    with pytest.raises(
        ValueError, match=r"invalid type for key subtype. Expected: int, Found: float"
    ):
        date_validator._validate_data_type(test_schema_3, test_key, 25.5)


def test_process_schema():

    data_validator = validator.SchemaValidator()

    with pytest.raises(TypeError):
        data_validator._process_schema()

    data_validator.data_dict = {}
    data_validator._process_schema()

    data_validator.data_dict["product"] = { "description": "test" }
    data_validator._process_schema()
    assert data_validator.data_dict["product"]["description"] == "test"

    data_validator.data_dict["product"] = { "description": "test\ntest" }
    data_validator._process_schema()
    assert data_validator.data_dict["product"]["description"] == "test test"


def test_validate_and_transform():

    date_validator = validator.SchemaValidator()
    date_validator.validate_and_transform()

    with pytest.raises(FileNotFoundError):
        date_validator.validate_and_transform(meta_file_name="this_file_is_not_there.yml")
    
    date_validator.validate_and_transform(meta_file_name="tests/resources/MLTdata-preproduction.meta.yml")
    assert date_validator.data_dict["activity"]["name"] == "mirror_2f_measurement"


def test_validate_schema():

    date_validator = validator.SchemaValidator()
    reference_schema = get_generic_instrument_reference_schema()
    test_schema_1 = get_instrument_test_schema()
    date_validator._validate_schema(reference_schema, test_schema_1)

    test_schema_2 = get_instrument_test_schema()
    test_schema_2["instrument"].pop("class")
    with pytest.raises(ValueError, match=r"Missing required field 'class'"):
        date_validator._validate_schema(reference_schema, test_schema_2)

    reference_schema_2 = get_generic_instrument_reference_schema()
    _telid = {"type": "int", "required": False}
    reference_schema_2["instrument"]["telid"] = _telid
    test_schema_3 = get_instrument_test_schema()
    test_schema_3["instrument"]["telid"] = 5.5
    with pytest.raises(ValueError):
        date_validator._validate_schema(reference_schema_2, test_schema_3)


def test_validate_list():

    date_validator = validator.SchemaValidator()
    date_validator._reference_schema = get_generic_instrument_reference_schema()

    instrument_1 = {
        "instrument": {
            "site": "South",
            "class": "MST",
            "type": "FlashCam",
            "subtype": "D",
            "id": "A",
        }
    }
    instrument_list = [instrument_1["instrument"]]
    date_validator._validate_list("instrumentlist", instrument_list)

    del instrument_1["instrument"]["class"]
    instrument_list.append(instrument_1)
    with pytest.raises(ValueError, match=r"Missing required field 'class'"):
        date_validator._validate_list("instrumentlist", instrument_list)


def test_check_if_field_is_optional():

    date_validator = validator.SchemaValidator()

    test_value_1 = {"required": False}
    test_value_2 = {"required": True}
    test_value_3 = {"required": None}
    # dictionaries with required fields
    test_value_4 = {"context": {"document": {"required": False}}}
    test_value_5 = {"context": {"document": {"required": True}}}
    test_value_6 = {"context": {"document": {"required": False}, "item": {"required": True}}}

    assert date_validator._field_is_optional(test_value_1)
    assert not date_validator._field_is_optional(test_value_2)
    assert date_validator._field_is_optional(test_value_3)
    assert date_validator._field_is_optional(test_value_4)
    assert not date_validator._field_is_optional(test_value_5)
    assert not date_validator._field_is_optional(test_value_6)


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
        "instrument": {
            "site": "north",
            "class": "camera",
            "type": "lst",
            "subtype": "subtype",
            "id": "id",
            "telid": 5.5,
        }
    }


def get_generic_instrument_reference_schema():

    return {
        "instrument": {
            "site": {"type": "str", "required": True},
            "class": {"type": "str", "required": True},
            "type": {"type": "str", "required": True},
            "subtype": {"type": "str", "required": False},
            "id": {"type": "str", "required": True},
        }
    }
