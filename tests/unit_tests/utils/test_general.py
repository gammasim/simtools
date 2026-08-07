#!/usr/bin/python3

import logging
import time
from copy import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

import simtools.utils.general as gen
from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH, TEST_RESOURCES_GENERATED

FAILED_TO_READ_FILE_ERROR = r"^Failed to read file"
KEY2_ADDED = "['key2']: added in second object"
KEY1_REMOVED = "['key1']: removed in second object"
KEY2_REMOVED = "['key2']: removed in second object"

url_desy = "https://www.desy.de"
url_simtools_main = "https://github.com/gammasim/simtools/"
url_simtools = "https://raw.githubusercontent.com/gammasim/simtools/main/"

test_data = "Test data"


def test_get_file_age(tmp_test_directory) -> None:
    # Create a temporary file and wait for 1 seconds before accessing it
    with open(tmp_test_directory / "test_file.txt", "w", encoding="utf-8") as file:
        file.write(test_data)

    time.sleep(0.2)

    try:
        age_in_minutes = gen.get_file_age(tmp_test_directory / "test_file.txt")
        # Age should be within an acceptable range (0 to 0.05 minutes or 3 seconds)
        assert 0 <= age_in_minutes <= 0.05
    except FileNotFoundError:
        pytest.fail("get_file_age raised FileNotFoundError for an existing file.")

    # Ensure that the function raises FileNotFoundError for a non-existent file
    with pytest.raises(FileNotFoundError):
        gen.get_file_age(tmp_test_directory / "nonexistent_file.txt")


def test_get_log_excerpt(tmp_test_directory) -> None:
    log_file = tmp_test_directory / "log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("This is a log file.\n")
        f.write("This is the second line of the log file.\n")

    assert gen.get_log_excerpt(log_file) == (
        "\n\nRuntime error - See below the relevant part of the log/err file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        "This is a log file."
        "This is the second line of the log file.\n\n"
        "====================================================================\n"
    )


def test_log_level_from_user() -> None:
    assert gen.get_log_level_from_user("info") == logging.INFO
    assert gen.get_log_level_from_user("debug") == logging.DEBUG
    assert gen.get_log_level_from_user("warning") == logging.WARNING
    assert gen.get_log_level_from_user("error") == logging.ERROR

    with pytest.raises(ValueError, match=r"^'invalid' is not a logging level"):
        gen.get_log_level_from_user("invalid")
    with pytest.raises(ValueError, match=r"^'1' is not a logging level"):
        gen.get_log_level_from_user(1)
    with pytest.raises(ValueError, match=r"^'None' is not a logging level"):
        gen.get_log_level_from_user(None)
    with pytest.raises(ValueError, match=r"^'True' is not a logging level"):
        gen.get_log_level_from_user(True)


def test_find_file_in_current_directory(tmp_test_directory) -> None:
    file_name = tmp_test_directory / "test.txt"
    with open(file_name, "w") as _file:
        _file.write(test_data)
    file_path = gen.find_file(file_name, tmp_test_directory)
    assert file_path == file_name


def test_find_file_in_non_existing_directory(tmp_test_directory) -> None:
    file_name = tmp_test_directory / "test.txt"

    loc = Path("non_existing_directory")
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_find_file_recursively(tmp_test_directory) -> None:
    file_name = "test_1.txt"
    test_directory_sub_dir = tmp_test_directory / "test"
    Path(test_directory_sub_dir).mkdir(parents=True, exist_ok=True)
    with open(test_directory_sub_dir / file_name, "w", encoding="utf-8") as _file:
        _file.write(test_data)
    loc = tmp_test_directory
    file_path = gen.find_file(file_name, loc)
    assert file_path == Path(loc).joinpath("test").joinpath(file_name)

    # Test also the case in which we recursively find unrelated files.
    file_name = "test_2.txt"
    Path(test_directory_sub_dir / "unrelated_sub_dir").mkdir(parents=True, exist_ok=True)
    with open(
        test_directory_sub_dir / "unrelated_sub_dir" / "unrelated_file.txt", "w", encoding="utf-8"
    ) as file:
        file.write(test_data)
    loc = tmp_test_directory
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_url_exists(caplog, mocker):
    import urllib.error

    def mock_urlopen(url, timeout=5):
        if url == url_simtools_main:
            mock_ctx = mocker.MagicMock()
            mock_ctx.__enter__.return_value.status = 200
            mock_ctx.__exit__.return_value = False
            return mock_ctx
        if url is None:
            raise AttributeError("'NoneType' object has no attribute")
        raise urllib.error.URLError("not found")

    mocker.patch("simtools.utils.general.urllib.request.urlopen", side_effect=mock_urlopen)

    assert gen.url_exists(url_simtools_main)
    with caplog.at_level(logging.ERROR):
        assert not gen.url_exists(url_simtools)  # raw URL does not exist
    assert "does not exist" in caplog.text
    with caplog.at_level(logging.ERROR):
        assert not gen.url_exists(None)
    assert "URL None" in caplog.text


def test_change_dict_keys_case(caplog) -> None:
    # note that entries in DATA_COLUMNS:ATTRIBUTE should not be changed (not keys)
    _upper_dict = {
        "REFERENCE": {"VERSION": "0.1.0"},
        "ACTIVITY": {"NAME": "submit", "ID": "84890304", "DESCRIPTION": "Set data"},
        "DATA_COLUMNS": {"ATTRIBUTE": ["remove_duplicates", "SORT"]},
        "DICT_IN_LIST": {
            "KEY_OF_FIRST_DICT": ["FIRST_ITEM", {"KEY_OF_NESTED_DICT": "VALUE_OF_SECOND_DICT"}]
        },
    }
    _lower_dict = {
        "reference": {"version": "0.1.0"},
        "activity": {"name": "submit", "id": "84890304", "description": "Set data"},
        "data_columns": {"attribute": ["remove_duplicates", "SORT"]},
        "dict_in_list": {
            "key_of_first_dict": ["FIRST_ITEM", {"key_of_nested_dict": "VALUE_OF_SECOND_DICT"}]
        },
    }
    _no_change_dict_upper = gen.change_dict_keys_case(copy(_upper_dict), False)
    assert _no_change_dict_upper == _upper_dict

    _no_change_dict_lower = gen.change_dict_keys_case(copy(_lower_dict), True)
    assert _no_change_dict_lower == _lower_dict

    _changed_to_lower = gen.change_dict_keys_case(copy(_upper_dict), True)
    assert _changed_to_lower == _lower_dict

    _changed_to_upper = gen.change_dict_keys_case(copy(_lower_dict), False)
    assert _changed_to_upper == _upper_dict
    with caplog.at_level("ERROR"):
        with pytest.raises(AttributeError):
            gen.change_dict_keys_case([2], False)
    assert "Input is not a proper dictionary" in caplog.text


@patch("builtins.input", side_effect=["Y", "y"])
def test_user_confirm_yes(mock_input):
    assert gen.user_confirm()


def test_validate_data_type():
    test_cases = [
        # Test exact data type match
        ("int", 5, None, False, True),
        ("int", 5.5, None, False, False),
        ("float", 3.14, None, False, True),
        ("str", "hello", None, False, True),
        ("bool", True, None, False, True),
        ("bool", 1, None, False, False),
        ("int", None, int, False, True),
        ("float", None, float, False, True),
        ("str", None, str, False, True),
        ("bool", None, bool, False, True),
        ("bool", None, bool, False, True),
        # Test allow_subtypes=True
        ("float", 5, None, True, True),
        ("float", [1, 2, 3], None, True, True),
        ("int", [1, 2, 3], None, True, True),
        ("int", np.array([1, 2, 3]), None, True, True),
        ("float", np.array([1.0, 2.0, 3.0]), None, True, True),
        ("file", "hello", None, True, True),
        ("string", "hello", None, True, True),
        ("file", None, "object", True, True),  # 'file' type with None value
        ("boolean", True, None, True, True),
        ("boolean", 1, None, True, True),
        ("boolean", 0, None, True, True),
        ("int", None, np.uint8, True, True),  # Subtype of 'int'
        ("float", None, int, True, True),  # 'int' can be converted to 'float'
        ("list", [1, 2, 3], None, True, True),
        ("dict", {"a": 1}, None, True, True),
    ]

    for reference_dtype, value, dtype, allow_subtypes, expected_result in test_cases:
        gen._logger.debug(f"{reference_dtype} {value} {dtype} {allow_subtypes} {expected_result}")
        assert (
            gen.validate_data_type(
                reference_dtype=reference_dtype,
                value=value,
                dtype=dtype,
                allow_subtypes=allow_subtypes,
            )
            is expected_result
        )

    with pytest.raises(ValueError, match=r"^Either value or dtype must be given"):
        gen.validate_data_type("int", None, None, False)

    assert gen.validate_data_type("int", 5.0) is False
    assert gen.validate_data_type("bool", 5) is False  # allow 0/1 to be booleans


def test_convert_list_to_string():
    assert gen.convert_list_to_string(None) is None
    assert gen.convert_list_to_string("a") == "a"
    assert gen.convert_list_to_string(5) == 5
    assert gen.convert_list_to_string([1, 2, 3]) == "1 2 3"
    assert gen.convert_list_to_string(np.array([1, 2, 3])) == "1 2 3"
    assert gen.convert_list_to_string(np.array([1, 2, 3]), comma_separated=True) == "1, 2, 3"
    assert (
        gen.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], comma_separated=False, shorten_list=True
        )
        == "all: 1"
    )
    assert (
        gen.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], comma_separated=False, shorten_list=False
        )
        == "1 1 1 1 1 1 1 1 1 1 1"
    )
    assert (
        gen.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            comma_separated=False,
            shorten_list=False,
            collapse_list=True,
        )
        == "1"
    )
    assert gen.convert_list_to_string([1, 2, 3], collapse_list=True) == "1 2 3"
    assert gen.convert_list_to_string([1, 2, 3], shorten_list=True) == "1 2 3"


def test_convert_string_to_list():
    t_1 = gen.convert_string_to_list("1 2 3 4")
    assert len(t_1) == 4
    assert t_1[1] == pytest.approx(2.0)
    assert isinstance(t_1[1], float)

    t_int = gen.convert_string_to_list("1 2 3 4", False)
    assert len(t_int) == 4
    assert t_int[1] == 2
    assert isinstance(t_int[1], int)

    t_2 = gen.convert_string_to_list("0.1 0.2 0.3 0.4")
    assert len(t_2) == 4
    assert t_2[1] == pytest.approx(0.2)

    t_3 = gen.convert_string_to_list("0.1")
    assert t_3[0] == pytest.approx(0.1)

    bla_bla = "bla bla"
    assert gen.convert_string_to_list("bla_bla") == "bla_bla"
    assert gen.convert_string_to_list(bla_bla) == ["bla", "bla"]
    assert gen.convert_string_to_list("bla,bla") == ["bla", "bla"]
    assert gen.convert_string_to_list(bla_bla, force_comma_separation=True) == bla_bla
    assert gen.convert_string_to_list("bla bla, bla blaa", force_comma_separation=True) == [
        bla_bla,
        "bla blaa",
    ]
    # import for list of dimensionless entries in database
    assert gen.convert_string_to_list(",") == ["", ""]
    assert gen.convert_string_to_list(" , , ") == ["", "", ""]


def test_get_structure_array_from_table():
    table = Table(
        {
            "col1": [1, 2, 3],
            "col2": [4.0, 5.0, 6.0],
            "col3": ["a", "b", "c"],
        }
    )

    # Test with all columns
    column_names = ["col1", "col2", "col3"]
    structured_array = gen.get_structure_array_from_table(table, column_names)
    assert structured_array.dtype.names == ("col1", "col2", "col3")
    assert structured_array["col1"].tolist() == [1, 2, 3]
    assert structured_array["col2"].tolist() == [4.0, 5.0, 6.0]
    assert structured_array["col3"].tolist() == ["a", "b", "c"]

    # Test with a subset of columns
    column_names = ["col1", "col3"]
    structured_array = gen.get_structure_array_from_table(table, column_names)
    assert structured_array.dtype.names == ("col1", "col3")
    assert structured_array["col1"].tolist() == [1, 2, 3]
    assert structured_array["col3"].tolist() == ["a", "b", "c"]

    # Test with a single column
    column_names = ["col2"]
    structured_array = gen.get_structure_array_from_table(table, column_names)
    assert structured_array.dtype.names == ("col2",)
    assert structured_array["col2"].tolist() == [4.0, 5.0, 6.0]

    # Test with an empty list of columns
    column_names = []
    assert gen.get_structure_array_from_table(table, column_names).size == 0

    # Test with a non-existent column (no error thrown)
    column_names = ["col1", "non_existent_col"]
    structured_array = gen.get_structure_array_from_table(table, column_names)
    assert structured_array.dtype.names == ("col1",)


def test_convert_keys_in_dict_to_lowercase():

    # Test with a simple dictionary.
    input_data = {"Key1": "value1", "Key2": "value2"}
    expected_output = {"key1": "value1", "key2": "value2"}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a nested dictionary.
    input_data = {"Key1": {"NestedKey1": "value1"}, "Key2": "value2"}
    expected_output = {"key1": {"nestedkey1": "value1"}, "key2": "value2"}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a dictionary containing a list.
    input_data = {"Key1": ["Value1", {"NestedKey1": "value1"}], "Key2": "value2"}
    expected_output = {"key1": ["Value1", {"nestedkey1": "value1"}], "key2": "value2"}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a list of dictionaries.
    input_data = [{"Key1": "value1"}, {"Key2": "value2"}]
    expected_output = [{"key1": "value1"}, {"key2": "value2"}]
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a non-dictionary input.
    input_data = "String"
    expected_output = "String"
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with an empty dictionary.
    input_data = {}
    expected_output = {}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a dictionary containing mixed types.
    input_data = {"Key1": 123, "Key2": [1, 2, 3], "Key3": {"NestedKey1": "value1"}}
    expected_output = {"key1": 123, "key2": [1, 2, 3], "key3": {"nestedkey1": "value1"}}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output


def test_get_list_of_files_from_command_line(tmp_test_directory) -> None:
    # Test with a list of file names with valid suffixes.
    file_1 = tmp_test_directory / "file1.txt"
    file_2 = tmp_test_directory / "file2.txt"
    with open(file_1, "w", encoding="utf-8") as f:
        f.write("Content of file 1")
    with open(file_2, "w", encoding="utf-8") as f:
        f.write("Content of file 2")
    file_names = [file_1, file_2]
    suffix_list = [".txt"]
    result = gen.get_list_of_files_from_command_line(file_names, suffix_list)
    assert result == [str(file_1), str(file_2)]

    # Test with a list of file names with invalid suffixes.
    suffix_list = [".json"]
    result = gen.get_list_of_files_from_command_line(file_names, suffix_list)
    assert result == []

    # Test with a text file containing a list of file names.
    list_file = tmp_test_directory / "list_file.list"
    with open(list_file, "w", encoding="utf-8") as f:
        f.write(f"{file_1}\n{file_2}\n")
    file_names = [list_file]
    suffix_list = []
    result = gen.get_list_of_files_from_command_line(file_names, suffix_list)
    assert result == [str(file_1), str(file_2)]

    # Test with a non-existent file.
    non_existent_file = tmp_test_directory / "non_existent_file.list"
    file_names = [non_existent_file]
    suffix_list = [".txt"]
    with pytest.raises(FileNotFoundError):
        gen.get_list_of_files_from_command_line(file_names, suffix_list)


def test_resolve_file_patterns():
    with pytest.raises(ValueError, match=r"^No file list provided"):
        gen.resolve_file_patterns(None)

    assert gen.resolve_file_patterns("LICENSE") == [Path("LICENSE")]
    yml_list = gen.resolve_file_patterns(f"{MODEL_PARAMETER_SCHEMA_PATH}/*.yml")
    assert len(yml_list) > 0
    yml_and_ecvs_list = gen.resolve_file_patterns(
        [
            f"{MODEL_PARAMETER_SCHEMA_PATH}/*.yml",
            f"{TEST_RESOURCES_GENERATED}/camera_efficiency/*.ecsv",
        ]
    )
    assert len(yml_and_ecvs_list) > len(yml_list)

    with pytest.raises(FileNotFoundError, match=r"^No files found"):
        gen.resolve_file_patterns(f"{TEST_RESOURCES_GENERATED}/*.non_existent")


def test_remove_key_from_dict():
    # Test with a simple dictionary
    input_data = {"key1": "value1", "key2": "value2", "key_to_remove": "value3"}
    expected_output = {"key1": "value1", "key2": "value2"}
    assert gen.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with a nested dictionary
    input_data = {
        "key1": {"nested_key1": "value1", "key_to_remove": "value2"},
        "key2": "value3",
        "key_to_remove": "value4",
    }
    expected_output = {"key1": {"nested_key1": "value1"}, "key2": "value3"}
    assert gen.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with a list of dictionaries
    input_data = [
        {"key1": "value1", "key_to_remove": "value2"},
        {"key2": "value3", "key_to_remove": "value4"},
    ]
    expected_output = [{"key1": "value1"}, {"key2": "value3"}]
    assert gen.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with a deeply nested structure
    input_data = {
        "key1": [
            {"nested_key1": {"key_to_remove": "value1", "key3": "value2"}},
            {"key_to_remove": "value3"},
        ],
        "key2": {"key_to_remove": "value4", "key4": "value5"},
    }
    expected_output = {
        "key1": [{"nested_key1": {"key3": "value2"}}, {}],
        "key2": {"key4": "value5"},
    }
    assert gen.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with no matching keys
    input_data = {"key1": "value1", "key2": "value2"}
    expected_output = {"key1": "value1", "key2": "value2"}
    assert gen.remove_key_from_dict(input_data, "non_existent_key") == expected_output

    # Test with an empty dictionary
    input_data = {}
    expected_output = {}
    assert gen.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with an empty list
    input_data = []
    expected_output = []
    assert gen.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with a list containing non-dictionary elements
    input_data = ["value1", {"key_to_remove": "value2"}, "value3"]
    expected_output = ["value1", {}, "value3"]
    assert gen.remove_key_from_dict(input_data, "key_to_remove") == expected_output


def test_find_differences_dict():
    # Test with two identical dictionaries
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1", "key2": "value2"}
    diffs = []
    gen._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == []

    # Test with a key added in obj2
    obj1 = {"key1": "value1"}
    obj2 = {"key1": "value1", "key2": "value2"}
    diffs = []
    gen._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == [KEY2_ADDED]

    # Test with a key removed in obj2
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1"}
    diffs = []
    gen._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == [KEY2_REMOVED]

    # Test with nested dictionaries
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value2"}}
    diffs = []
    gen._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == ["['key1']['nested_key1']: value changed from value1 to value2"]

    # Test with a key added in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    diffs = []
    gen._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == ["['key1']['nested_key2']: added in second object"]

    # Test with a key removed in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    obj2 = {"key1": {"nested_key1": "value1"}}
    diffs = []
    gen._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == ["['key1']['nested_key2']: removed in second object"]

    # Test with completely different dictionaries
    obj1 = {"key1": "value1"}
    obj2 = {"key2": "value2"}
    diffs = []
    gen._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == [KEY1_REMOVED, KEY2_ADDED]


def test_find_differences_in_json_objects():
    # Test with identical dictionaries
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1", "key2": "value2"}
    assert gen.find_differences_in_json_objects(obj1, obj2) == []

    # Test with different types
    obj1 = {"key1": "value1"}
    obj2 = ["value1"]
    assert gen.find_differences_in_json_objects(obj1, obj2) == [": type changed from dict to list"]

    # Test with a key added in obj2
    obj1 = {"key1": "value1"}
    obj2 = {"key1": "value1", "key2": "value2"}
    assert gen.find_differences_in_json_objects(obj1, obj2) == [KEY2_ADDED]

    # Test with a key removed in obj2
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1"}
    assert gen.find_differences_in_json_objects(obj1, obj2) == [KEY2_REMOVED]

    # Test with nested dictionaries
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value2"}}
    assert gen.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key1']: value changed from value1 to value2"
    ]

    # Test with a key added in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    assert gen.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key2']: added in second object"
    ]

    # Test with a key removed in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    obj2 = {"key1": {"nested_key1": "value1"}}
    assert gen.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key2']: removed in second object"
    ]

    # Test with lists of different lengths
    obj1 = [1, 2, 3]
    obj2 = [1, 2]
    assert gen.find_differences_in_json_objects(obj1, obj2) == [": list length changed from 3 to 2"]

    # Test with lists containing different values
    obj1 = [1, 2, 3]
    obj2 = [1, 4, 3]
    assert gen.find_differences_in_json_objects(obj1, obj2) == ["[1]: value changed from 2 to 4"]

    # Test with completely different structures
    obj1 = {"key1": "value1"}
    obj2 = {"key2": "value2"}
    assert gen.find_differences_in_json_objects(obj1, obj2) == [KEY1_REMOVED, KEY2_ADDED]

    # Test with deeply nested structures
    obj1 = {"key1": {"nested_key1": {"deep_key": "value1"}}}
    obj2 = {"key1": {"nested_key1": {"deep_key": "value2"}}}
    assert gen.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key1']['deep_key']: value changed from value1 to value2"
    ]


def test_ensure_list():
    assert gen.ensure_list(None) == []
    assert gen.ensure_list([1, 2, 3]) == [1, 2, 3]
    assert gen.ensure_list((1, 2, 3)) == [1, 2, 3]
    assert gen.ensure_list(5) == [5]
    # Test falsy values are correctly wrapped (not treated as None)
    assert gen.ensure_list(0) == [0]
    assert gen.ensure_list(0.0) == [0.0]
    assert gen.ensure_list("") == [""]
    assert gen.ensure_list(False) == [False]
    assert gen.ensure_list("abc") == ["abc"]
    # json list
    raw_value = '["alice", "bob", "charlie"]'
    assert gen.ensure_list(raw_value) == ["alice", "bob", "charlie"]


def test_parse_typed_sequence():
    assert gen.parse_typed_sequence(None) == []
    assert gen.parse_typed_sequence(["1", "2"], int) == [1, 2]
    assert gen.parse_typed_sequence(("1", "2"), int) == [1, 2]
    assert gen.parse_typed_sequence("1, 2,3", int) == [1, 2, 3]
    assert gen.parse_typed_sequence(["1e6", "2e6"], int) == [1000000, 2000000]
    assert gen.parse_typed_sequence(5, int) == [5]
    assert gen.parse_typed_sequence("2.5") == [2.5]

    with pytest.raises(ValueError, match="Cannot safely cast non-integer value"):
        gen.parse_typed_sequence("1.5", int)


@patch("tarfile.open")  # NOSONAR
def test_pack_tar_file_mocked_tarfile(mock_tarfile_open, tmp_test_directory):
    tar_file_name = tmp_test_directory / "test_archive.tar.gz"
    # Do not actually create directories or files; mock filesystem interactions
    base_dir = tmp_test_directory / "base"

    # Paths for files (not created on disk)
    file1 = base_dir / "file1.txt"
    file2 = base_dir / "file2.txt"

    # Patch Path.is_file and Path.resolve to avoid touching the filesystem
    orig_is_file = Path.is_file
    orig_resolve = Path.resolve

    def is_file_side(self):
        # Only report our two test files as files
        if str(self) in (str(file1), str(file2)):
            return True
        return orig_is_file(self)

    def resolve_side(self, *args, **kwargs):
        # For our test files, return the path itself (tmp_test_directory paths are absolute)
        if str(self) in (str(file1), str(file2)):
            return self
        return orig_resolve(self, *args, **kwargs)

    from unittest.mock import patch as _patch

    patch_is_file = _patch.object(Path, "is_file", new=is_file_side)
    patch_resolve = _patch.object(Path, "resolve", new=resolve_side)

    mock_tar = MagicMock()
    mock_tarfile_open.return_value.__enter__.return_value = mock_tar

    # Call the function with Path methods patched
    with patch_is_file, patch_resolve:
        gen.pack_tar_file(tar_file_name, [file1, file2])

    # Verify tarfile.open was called correctly
    mock_tarfile_open.assert_called_once_with(tar_file_name, "w:gz")
    mock_tar.add.assert_any_call(file1, arcname="file1.txt")
    mock_tar.add.assert_any_call(file2, arcname="file2.txt")

    # Test sub_dir option
    mock_tarfile_open.reset_mock()
    mock_tar.reset_mock()
    with patch_is_file, patch_resolve:
        gen.pack_tar_file(tar_file_name, [file1, file2], sub_dir="subdir")
    mock_tarfile_open.assert_called_once_with(tar_file_name, "w:gz")
    mock_tar.add.assert_any_call(file1, arcname="subdir/file1.txt")
    mock_tar.add.assert_any_call(file2, arcname="subdir/file2.txt")

    with pytest.raises(ValueError, match="Unsafe file path"):
        gen.pack_tar_file(tar_file_name, ["unsafe_file"])


def test_load_environment_variables(tmp_test_directory, monkeypatch):
    env_file = tmp_test_directory / ".env"

    # Create a test .env file
    with open(env_file, "w", encoding="utf-8") as f:
        f.write('SIMTOOLS_VAR1="value1"\n')
        f.write("SIMTOOLS_VAR2=value2\n")
        f.write("SIMTOOLS_VAR3=value3 # comment\n")
        f.write("SIMTOOLS_VAR4='value4'\n")

    # Test loading all variables
    result = gen.load_environment_variables(str(env_file))
    assert result == {
        "var1": "value1",
        "var2": "value2",
        "var3": "value3",
        "var4": "value4",
    }

    # Test loading specific variables
    result = gen.load_environment_variables(str(env_file), ["var1", "var3", "var5"])
    assert "var1" in result
    assert "var3" in result
    assert result["var1"] == "value1"
    assert result["var3"] == "value3"
    assert "var5" not in result

    # Test with non-existent env file
    result = gen.load_environment_variables(str(tmp_test_directory / "non_existent.env"))
    assert result == {}

    # Test with environment variable set but not in file
    monkeypatch.setenv("SIMTOOLS_EXTERNAL_VAR", "external_value")
    result = gen.load_environment_variables(str(env_file), ["external_var"])
    assert result.get("external_var") == "external_value"


def test_find_executable_in_dir(tmp_test_directory) -> None:
    executable_file = tmp_test_directory / "test_executable"
    with open(executable_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\necho 'test'")
    executable_file.chmod(0o755)

    result = gen.find_executable_in_dir("test_executable", tmp_test_directory)
    assert result == executable_file


def test_find_executable_in_dir_not_found(tmp_test_directory) -> None:
    with pytest.raises(FileNotFoundError, match=r"^Executable not found"):
        gen.find_executable_in_dir("non_existent", tmp_test_directory)


def test_find_executable_in_dir_not_executable(tmp_test_directory) -> None:
    non_executable_file = tmp_test_directory / "test_file"
    with open(non_executable_file, "w", encoding="utf-8") as f:
        f.write("test content")
    non_executable_file.chmod(0o644)

    with pytest.raises(PermissionError, match=r"^Not executable"):
        gen.find_executable_in_dir("test_file", tmp_test_directory)

    with pytest.raises(ValueError, match=r"Both name and directory must be provided."):
        gen.find_executable_in_dir(None, None)


def test_is_safe_tar_member_unsafe_absolute_path() -> None:
    assert not gen.is_safe_tar_member("/etc/passwd")
    assert not gen.is_safe_tar_member("/path/to/file.log")


def test_is_safe_tar_member_unsafe_traversal() -> None:
    assert not gen.is_safe_tar_member("logs/../../../etc/passwd")
    assert not gen.is_safe_tar_member("../logs/test.log")
    assert not gen.is_safe_tar_member("path/../../dangerous/file")


def test_is_safe_tar_member_unsafe_null_byte() -> None:
    assert not gen.is_safe_tar_member("path\x00withNull")
    assert not gen.is_safe_tar_member("file.log\x00.exe")


def test_get_simtools_log_file_with_file_handler(tmp_test_directory) -> None:
    log_file = tmp_test_directory / "test.log"

    file_handler = logging.FileHandler(log_file)
    gen._logger.addHandler(file_handler)

    try:
        result = gen.get_simtools_log_file()
        assert result == str(log_file)
    finally:
        gen._logger.removeHandler(file_handler)
        file_handler.close()


def test_get_simtools_log_file_without_file_handler() -> None:
    # Save all handlers from the entire logger hierarchy
    saved_handlers = {}
    current_logger = gen._logger
    level = 0

    while current_logger:
        saved_handlers[level] = (current_logger, current_logger.handlers[:])
        current_logger.handlers = []
        current_logger = current_logger.parent
        level += 1

    try:
        result = gen.get_simtools_log_file()
        assert result is None
    finally:
        # Restore all handlers
        for level, (logger, handlers) in saved_handlers.items():
            logger.handlers = handlers


def test_ensure_string_lists():
    assert gen.ensure_string_lists(None) is None
    assert gen.ensure_string_lists("hello") == ["hello"]
    assert gen.ensure_string_lists(["a", "b", "c"]) == ["a", "b", "c"]
    assert gen.ensure_string_lists(("x", "y")) == ["x", "y"]
    # Tests that integers (non-iterable) raise a TypeError, as per list() behavior
    with pytest.raises(TypeError):
        gen.ensure_string_lists(123)
