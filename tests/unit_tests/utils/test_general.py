#!/usr/bin/python3

import datetime
import logging
import os
import time
from copy import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

import simtools.utils.general as gen

FAILED_TO_READ_FILE_ERROR = r"^Failed to read file"
KEY2_ADDED = "['key2']: added in second object"
KEY1_REMOVED = "['key1']: removed in second object"
KEY2_REMOVED = "['key2']: removed in second object"


url_desy = "https://www.desy.de"
url_simtools_main = "https://github.com/gammasim/simtools/"
url_simtools = "https://raw.githubusercontent.com/gammasim/simtools/main/"


test_data = "Test data"


def test_program_is_executable(caplog) -> None:
    # (assume 'ls' exist on any system the test is running)
    assert gen.program_is_executable("/bin/ls") is not None  # The actual path should not matter
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None
    with caplog.at_level("WARNING"):
        os.environ.pop("PATH", None)
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None
    assert "PATH environment variable is not set." in caplog.text


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


def test_file_has_text(tmp_test_directory, caplog, file_has_text) -> None:
    """Test the file_has_text function."""

    # Test with file that has text.
    file = tmp_test_directory / "test_file_has_text.txt"
    text = "test"
    with open(file, "w") as f:
        f.write(text)
    assert file_has_text(file, text)
    assert not file_has_text(file, "test2")

    # Test with empty file.
    file = tmp_test_directory / "test_file_is_empty.txt"
    with open(file, "w") as f:
        f.write("")
    assert not file_has_text(file, text)

    # Test with file that does not exist.
    file = tmp_test_directory / "test_file_does_not_exist.txt"
    assert not file_has_text(file, text)


def test_collect_kwargs() -> None:
    """
    Test the collect_kwargs function.
    """

    # Test with no kwargs.
    kwargs = {}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {}

    # Test with one kwargs.
    kwargs = {"label_a": 1}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"a": 1}

    # Test with multiple kwargs.
    kwargs = {"label_a": 1, "label_b": 2, "label_c": 3}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"a": 1, "b": 2, "c": 3}

    # Test with kwargs where only one starts with label_.
    kwargs = {"a": 1, "b": 2, "label_c": 3, "d": 4}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"c": 3}

    # Test with kwargs that do not start with label_.
    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {}


def test_set_default_kwargs() -> None:
    """
    Test the set_default_kwargs function.
    """

    in_kwargs = {"a": 1, "b": 2}
    out_kwargs = gen.set_default_kwargs(in_kwargs, c=3, d=4)
    assert out_kwargs == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_log_level_from_user() -> None:
    """
    Test get_log_level_from_user() function.
    """
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
    """
    Test finding a file in the temp test directory directory.
    """
    file_name = tmp_test_directory / "test.txt"
    with open(file_name, "w") as _file:
        _file.write(test_data)
    file_path = gen.find_file(file_name, tmp_test_directory)
    assert file_path == file_name


def test_find_file_in_non_existing_directory(tmp_test_directory) -> None:
    """
    Test finding a file in a non-existing directory.
    """
    file_name = tmp_test_directory / "test.txt"

    loc = Path("non_existing_directory")
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_find_file_recursively(tmp_test_directory) -> None:
    """
    Test finding a file recursively.
    """
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


def test_find_file_not_found(tmp_test_directory) -> None:
    """
    Test finding a file that does not exist.
    """
    file_name = "not_existing_file.txt"
    loc = Path(tmp_test_directory)
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_is_url():
    url = url_desy
    assert gen.is_url(url) is True

    url = "sftp://www.desy.de"
    assert gen.is_url(url) is True

    url = ""
    assert gen.is_url(url) is False

    url = "https://"
    assert gen.is_url(url) is False

    url = "desy.de"
    assert gen.is_url(url) is False

    assert gen.is_url(5.0) is False


@pytest.mark.xfail(reason="No network connection")
def test_url_exists(caplog):
    assert gen.url_exists(url_simtools_main)
    with caplog.at_level(logging.ERROR):
        assert not gen.url_exists(url_simtools)  # raw ULR does not exist
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


@pytest.mark.parametrize(
    ("input_data", "expected_output"),
    [
        (
            {
                "key1": "This is a string\n with a newline",
                "key2": ["List item 1\n", "List item 2\n"],
                "key3": {"nested_key": "Nested string\n with a newline"},
                "key4": [{"nested_dict": "string2\n"}, {"nested_dict2": "string3\n"}],
            },
            {
                "key1": "This is a string with a newline",
                "key2": ["List item 1", "List item 2"],
                "key3": {"nested_key": "Nested string with a newline"},
                "key4": [{"nested_dict": "string2"}, {"nested_dict2": "string3"}],
            },
        ),
    ],
)
def test_remove_substring_recursively_from_dict(input_data, expected_output, caplog):
    result = gen.remove_substring_recursively_from_dict(input_data, "\n")
    assert result == expected_output

    # no error should be raised for None input, but a debug message should be printed
    gen._logger.setLevel(logging.DEBUG)
    gen.remove_substring_recursively_from_dict([2])
    assert any(
        record.levelname == "DEBUG" and "Input is not a dictionary: [2]" in record.message
        for record in caplog.records
    )


@patch("builtins.input", side_effect=["Y", "y"])
def test_user_confirm_yes(mock_input):
    assert gen.user_confirm()


@patch("builtins.input", side_effect=["N", "n", EOFError, "not_Y_or_N"])
def test_user_confirm_no(mock_input):
    assert not gen.user_confirm()


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
    assert pytest.approx(t_1[1]) == 2.0
    assert isinstance(t_1[1], float)

    t_int = gen.convert_string_to_list("1 2 3 4", False)
    assert len(t_int) == 4
    assert t_int[1] == 2
    assert isinstance(t_int[1], int)

    t_2 = gen.convert_string_to_list("0.1 0.2 0.3 0.4")
    assert len(t_2) == 4
    assert pytest.approx(t_2[1]) == 0.2

    t_3 = gen.convert_string_to_list("0.1")
    assert pytest.approx(t_3[0]) == 0.1

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
    """
    Test the convert_keys_in_dict_to_lowercase function.
    """

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


def test_clear_default_sim_telarray_cfg_directories():
    """
    Test the clear_default_sim_telarray_cfg_directories function.
    """

    # Test with a simple command.
    command = "run_simulation"
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' run_simulation"
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output

    # Test with a command containing spaces.
    command = "run_simulation --config config_file"
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' run_simulation --config config_file"
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output

    # Test with an empty command.
    command = ""
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' "
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output

    # Test with a command containing special characters.
    command = "run_simulation && echo 'done'"
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' run_simulation && echo 'done'"
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output


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
    yml_list = gen.resolve_file_patterns("tests/resources/*.yml")
    assert len(yml_list) > 0
    yml_and_ecvs_list = gen.resolve_file_patterns(
        ["tests/resources/*.yml", "tests/resources/*.ecsv"]
    )
    assert len(yml_and_ecvs_list) > len(yml_list)

    with pytest.raises(FileNotFoundError, match=r"^No files found"):
        gen.resolve_file_patterns("tests/resources/*.non_existent")


def test_now_date_time_in_isoformat():
    now = gen.now_date_time_in_isoformat()
    assert now is not None
    assert isinstance(now, str)
    assert len(now) == 25
    assert now[4] == "-"
    assert now[7] == "-"
    assert now[10] == "T"
    assert now[13] == ":"
    assert now[16] == ":"
    assert datetime.datetime.fromisoformat(now) is not None


def test_is_valid_numeric_type():
    """Test _is_valid_numeric_type function."""
    # Test integer dtypes
    assert gen._is_valid_numeric_type(np.int32, np.int64)
    assert gen._is_valid_numeric_type(np.uint8, np.integer)
    assert gen._is_valid_numeric_type(np.int64, np.floating)
    assert not gen._is_valid_numeric_type(np.int32, np.str_)
    assert not gen._is_valid_numeric_type(np.int32, np.bool_)

    # Test float dtypes
    assert gen._is_valid_numeric_type(np.float32, np.float64)
    assert gen._is_valid_numeric_type(np.float64, np.floating)
    assert not gen._is_valid_numeric_type(np.float32, np.integer)
    assert not gen._is_valid_numeric_type(np.float32, np.str_)
    assert not gen._is_valid_numeric_type(np.float32, np.bool_)

    # Test non-numeric dtypes
    assert not gen._is_valid_numeric_type(np.str_, np.integer)
    assert not gen._is_valid_numeric_type(np.bool_, np.floating)
    assert not gen._is_valid_numeric_type(np.object_, np.integer)


def test_is_valid_boolean_type():
    """Test _is_valid_boolean_type function."""
    # Test values 0 and 1
    assert gen._is_valid_boolean_type(np.int32, 0)
    assert gen._is_valid_boolean_type(np.int32, 1)
    assert gen._is_valid_boolean_type(np.float32, 0)
    assert gen._is_valid_boolean_type(np.float32, 1)

    # Test boolean dtype
    assert gen._is_valid_boolean_type(np.bool_, None)
    assert gen._is_valid_boolean_type(bool, None)

    # Test non-boolean dtypes with values other than 0/1
    assert not gen._is_valid_boolean_type(np.int32, 2)
    assert not gen._is_valid_boolean_type(np.float32, 0.5)
    assert not gen._is_valid_boolean_type(np.str_, "True")

    # Test non-boolean dtypes with None value
    assert not gen._is_valid_boolean_type(np.int32, None)
    assert not gen._is_valid_boolean_type(np.float32, None)
    assert not gen._is_valid_boolean_type(np.str_, None)


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


def test_ensure_iterable():
    assert gen.ensure_iterable(None) == []
    assert gen.ensure_iterable([1, 2, 3]) == [1, 2, 3]
    assert gen.ensure_iterable(5) == [5]
    assert gen.ensure_iterable((1, 2, 3)) == (1, 2, 3)


@patch("tarfile.open")
def test_pack_tar_file_mocked_tarfile(mock_tarfile_open, tmp_test_directory):
    tar_file_name = tmp_test_directory / "test_archive.tar.gz"
    base_dir = tmp_test_directory / "base"
    base_dir.mkdir()

    # Create test files
    file1 = base_dir / "file1.txt"
    file2 = base_dir / "file2.txt"
    file1.write_text("This is file 1.", encoding="utf-8")
    file2.write_text("This is file 2.", encoding="utf-8")

    mock_tar = MagicMock()
    mock_tarfile_open.return_value.__enter__.return_value = mock_tar

    # Call the function
    gen.pack_tar_file(tar_file_name, [file1, file2], base=base_dir)

    # Verify tarfile.open was called correctly
    mock_tarfile_open.assert_called_once_with(tar_file_name, "w:gz")
    mock_tar.add.assert_any_call(file1, arcname="file1.txt")
    mock_tar.add.assert_any_call(file2, arcname="file2.txt")

    with pytest.raises(ValueError, match="Unsafe file path"):
        gen.pack_tar_file(tar_file_name, ["unsafe_file"], base=base_dir)
