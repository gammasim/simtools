#!/usr/bin/python3

from copy import copy
from pathlib import Path

import numpy as np
import pytest

import simtools.applications.maintain_simulation_model_compare_production_tables as compare_productions
import simtools.job_execution.log_inspector as log_inspector
import simtools.utils.general as gen
from simtools.utils import value_conversion

FAILED_TO_READ_FILE_ERROR = r"^Failed to read file"
KEY2_ADDED = "['key2']: added in second object"
KEY1_REMOVED = "['key1']: removed in second object"
KEY2_REMOVED = "['key2']: removed in second object"

url_desy = "https://www.desy.de"
url_simtools_main = "https://github.com/gammasim/simtools/"
url_simtools = "https://raw.githubusercontent.com/gammasim/simtools/main/"

test_data = "Test data"


def test_cleanup_intermediate_files(tmp_test_directory):
    root = Path(tmp_test_directory)
    for name in ("run.log", "run.lis.gz", "input.dat", "result.ecsv"):
        (root / name).write_text("data", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "keep.log").write_text("data", encoding="utf-8")
    empty = root / "empty" / "child"
    empty.mkdir(parents=True)
    assert (
        gen.cleanup_intermediate_files(
            root,
            patterns=("*.log", "*.lis*", "*.dat"),
            files=(root / "run.log", root / "missing"),
            directories=(empty.parent, empty, nested),
        )
        == 3
    )
    assert (root / "result.ecsv").exists()
    assert (nested / "keep.log").exists()
    assert not empty.parent.exists()
    assert gen.cleanup_intermediate_files(root) == 0


def test_cleanup_explicit_files_and_symlinks(tmp_test_directory):
    root = Path(tmp_test_directory)
    target = root / "retained"
    target.mkdir()
    (target / "keep.dat").write_text("data", encoding="utf-8")
    link = root / "link"
    link.symlink_to(target, target_is_directory=True)
    gen.cleanup_intermediate_files(root, directories=(link,))
    assert link.is_symlink()
    assert gen.cleanup_intermediate_files(root, files=(link,)) == 1
    assert (target / "keep.dat").exists()


def test_cleanup_preserves_excluded_paths(tmp_test_directory):
    root = Path(tmp_test_directory)
    keep = root / "keep.dat"
    keep.write_text("existing", encoding="utf-8")
    nested = root / "generated" / "nested"
    nested.mkdir(parents=True)
    (nested / "remove.dat").write_text("generated", encoding="utf-8")
    assert gen.cleanup_intermediate_files(root, patterns=("**/*",), exclude=(keep,)) == 1
    assert keep.read_text(encoding="utf-8") == "existing"
    assert not nested.parent.exists()


def test_get_log_excerpt(tmp_test_directory) -> None:
    log_file = tmp_test_directory / "log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("This is a log file.\n")
        f.write("This is the second line of the log file.\n")

    assert log_inspector.get_log_excerpt(log_file) == (
        "\n\nRuntime error - See below the relevant part of the log/err file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        "This is a log file."
        "This is the second line of the log file.\n\n"
        "====================================================================\n"
    )


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
    _no_change_dict_upper = value_conversion.change_dict_keys_case(copy(_upper_dict), False)
    assert _no_change_dict_upper == _upper_dict

    _no_change_dict_lower = value_conversion.change_dict_keys_case(copy(_lower_dict), True)
    assert _no_change_dict_lower == _lower_dict

    _changed_to_lower = value_conversion.change_dict_keys_case(copy(_upper_dict), True)
    assert _changed_to_lower == _lower_dict

    _changed_to_upper = value_conversion.change_dict_keys_case(copy(_lower_dict), False)
    assert _changed_to_upper == _upper_dict
    with caplog.at_level("ERROR"):
        with pytest.raises(AttributeError):
            value_conversion.change_dict_keys_case([2], False)
    assert "Input is not a proper dictionary" in caplog.text


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
            value_conversion.validate_data_type(
                reference_dtype=reference_dtype,
                value=value,
                dtype=dtype,
                allow_subtypes=allow_subtypes,
            )
            is expected_result
        )

    with pytest.raises(ValueError, match=r"^Either value or dtype must be given"):
        value_conversion.validate_data_type("int", None, None, False)

    assert value_conversion.validate_data_type("int", 5.0) is False
    assert value_conversion.validate_data_type("bool", 5) is False  # allow 0/1 to be booleans


def test_convert_list_to_string():
    assert value_conversion.convert_list_to_string(None) is None
    assert value_conversion.convert_list_to_string("a") == "a"
    assert value_conversion.convert_list_to_string(5) == 5
    assert value_conversion.convert_list_to_string([1, 2, 3]) == "1 2 3"
    assert value_conversion.convert_list_to_string(np.array([1, 2, 3])) == "1 2 3"
    assert (
        value_conversion.convert_list_to_string(np.array([1, 2, 3]), comma_separated=True)
        == "1, 2, 3"
    )
    assert (
        value_conversion.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], comma_separated=False, shorten_list=True
        )
        == "all: 1"
    )
    assert (
        value_conversion.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], comma_separated=False, shorten_list=False
        )
        == "1 1 1 1 1 1 1 1 1 1 1"
    )
    assert (
        value_conversion.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            comma_separated=False,
            shorten_list=False,
            collapse_list=True,
        )
        == "1"
    )
    assert value_conversion.convert_list_to_string([1, 2, 3], collapse_list=True) == "1 2 3"
    assert value_conversion.convert_list_to_string([1, 2, 3], shorten_list=True) == "1 2 3"


def test_convert_string_to_list():
    t_1 = value_conversion.convert_string_to_list("1 2 3 4")
    assert len(t_1) == 4
    assert t_1[1] == pytest.approx(2.0)
    assert isinstance(t_1[1], float)

    t_int = value_conversion.convert_string_to_list("1 2 3 4", False)
    assert len(t_int) == 4
    assert t_int[1] == 2
    assert isinstance(t_int[1], int)

    t_2 = value_conversion.convert_string_to_list("0.1 0.2 0.3 0.4")
    assert len(t_2) == 4
    assert t_2[1] == pytest.approx(0.2)

    t_3 = value_conversion.convert_string_to_list("0.1")
    assert t_3[0] == pytest.approx(0.1)

    bla_bla = "bla bla"
    assert value_conversion.convert_string_to_list("bla_bla") == "bla_bla"
    assert value_conversion.convert_string_to_list(bla_bla) == ["bla", "bla"]
    assert value_conversion.convert_string_to_list("bla,bla") == ["bla", "bla"]
    assert value_conversion.convert_string_to_list(bla_bla, force_comma_separation=True) == bla_bla
    assert value_conversion.convert_string_to_list(
        "bla bla, bla blaa", force_comma_separation=True
    ) == [
        bla_bla,
        "bla blaa",
    ]
    # import for list of dimensionless entries in model repository
    assert value_conversion.convert_string_to_list(",") == ["", ""]
    assert value_conversion.convert_string_to_list(" , , ") == ["", "", ""]


def test_convert_keys_in_dict_to_lowercase():

    # Test with a simple dictionary.
    input_data = {"Key1": "value1", "Key2": "value2"}
    expected_output = {"key1": "value1", "key2": "value2"}
    assert value_conversion.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a nested dictionary.
    input_data = {"Key1": {"NestedKey1": "value1"}, "Key2": "value2"}
    expected_output = {"key1": {"nestedkey1": "value1"}, "key2": "value2"}
    assert value_conversion.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a dictionary containing a list.
    input_data = {"Key1": ["Value1", {"NestedKey1": "value1"}], "Key2": "value2"}
    expected_output = {"key1": ["Value1", {"nestedkey1": "value1"}], "key2": "value2"}
    assert value_conversion.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a list of dictionaries.
    input_data = [{"Key1": "value1"}, {"Key2": "value2"}]
    expected_output = [{"key1": "value1"}, {"key2": "value2"}]
    assert value_conversion.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a non-dictionary input.
    input_data = "String"
    expected_output = "String"
    assert value_conversion.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with an empty dictionary.
    input_data = {}
    expected_output = {}
    assert value_conversion.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a dictionary containing mixed types.
    input_data = {"Key1": 123, "Key2": [1, 2, 3], "Key3": {"NestedKey1": "value1"}}
    expected_output = {"key1": 123, "key2": [1, 2, 3], "key3": {"nestedkey1": "value1"}}
    assert value_conversion.convert_keys_in_dict_to_lowercase(input_data) == expected_output


def test_resolve_file_patterns(tmp_test_directory):
    with pytest.raises(ValueError, match=r"^No file list provided"):
        gen.resolve_file_patterns(None)

    assert gen.resolve_file_patterns("LICENSE") == [Path("LICENSE")]
    schema_directory = tmp_test_directory / "schemas"
    camera_directory = tmp_test_directory / "camera_efficiency"
    schema_directory.mkdir()
    camera_directory.mkdir()
    Path(schema_directory / "schema.yml").write_text("schema: test\n", encoding="utf-8")
    Path(camera_directory / "efficiency.ecsv").write_text("value\n1\n", encoding="utf-8")

    yml_list = gen.resolve_file_patterns(f"{schema_directory}/*.yml")
    assert len(yml_list) > 0
    yml_and_ecvs_list = gen.resolve_file_patterns(
        [
            f"{schema_directory}/*.yml",
            f"{camera_directory}/*.ecsv",
        ]
    )
    assert len(yml_and_ecvs_list) > len(yml_list)

    with pytest.raises(FileNotFoundError, match=r"^No files found"):
        gen.resolve_file_patterns(f"{tmp_test_directory}/*.non_existent")


def test_remove_key_from_dict():
    # Test with a simple dictionary
    input_data = {"key1": "value1", "key2": "value2", "key_to_remove": "value3"}
    expected_output = {"key1": "value1", "key2": "value2"}
    assert value_conversion.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with a nested dictionary
    input_data = {
        "key1": {"nested_key1": "value1", "key_to_remove": "value2"},
        "key2": "value3",
        "key_to_remove": "value4",
    }
    expected_output = {"key1": {"nested_key1": "value1"}, "key2": "value3"}
    assert value_conversion.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with a list of dictionaries
    input_data = [
        {"key1": "value1", "key_to_remove": "value2"},
        {"key2": "value3", "key_to_remove": "value4"},
    ]
    expected_output = [{"key1": "value1"}, {"key2": "value3"}]
    assert value_conversion.remove_key_from_dict(input_data, "key_to_remove") == expected_output

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
    assert value_conversion.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with no matching keys
    input_data = {"key1": "value1", "key2": "value2"}
    expected_output = {"key1": "value1", "key2": "value2"}
    assert value_conversion.remove_key_from_dict(input_data, "non_existent_key") == expected_output

    # Test with an empty dictionary
    input_data = {}
    expected_output = {}
    assert value_conversion.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with an empty list
    input_data = []
    expected_output = []
    assert value_conversion.remove_key_from_dict(input_data, "key_to_remove") == expected_output

    # Test with a list containing non-dictionary elements
    input_data = ["value1", {"key_to_remove": "value2"}, "value3"]
    expected_output = ["value1", {}, "value3"]
    assert value_conversion.remove_key_from_dict(input_data, "key_to_remove") == expected_output


def test_find_differences_dict():
    # Test with two identical dictionaries
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1", "key2": "value2"}
    diffs = []
    compare_productions._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == []

    # Test with a key added in obj2
    obj1 = {"key1": "value1"}
    obj2 = {"key1": "value1", "key2": "value2"}
    diffs = []
    compare_productions._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == [KEY2_ADDED]

    # Test with a key removed in obj2
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1"}
    diffs = []
    compare_productions._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == [KEY2_REMOVED]

    # Test with nested dictionaries
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value2"}}
    diffs = []
    compare_productions._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == ["['key1']['nested_key1']: value changed from value1 to value2"]

    # Test with a key added in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    diffs = []
    compare_productions._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == ["['key1']['nested_key2']: added in second object"]

    # Test with a key removed in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    obj2 = {"key1": {"nested_key1": "value1"}}
    diffs = []
    compare_productions._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == ["['key1']['nested_key2']: removed in second object"]

    # Test with completely different dictionaries
    obj1 = {"key1": "value1"}
    obj2 = {"key2": "value2"}
    diffs = []
    compare_productions._find_differences_dict(obj1, obj2, "", diffs)
    assert diffs == [KEY1_REMOVED, KEY2_ADDED]


def test_find_differences_in_json_objects():
    # Test with identical dictionaries
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1", "key2": "value2"}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == []

    # Test with different types
    obj1 = {"key1": "value1"}
    obj2 = ["value1"]
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        ": type changed from dict to list"
    ]

    # Test with a key added in obj2
    obj1 = {"key1": "value1"}
    obj2 = {"key1": "value1", "key2": "value2"}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [KEY2_ADDED]

    # Test with a key removed in obj2
    obj1 = {"key1": "value1", "key2": "value2"}
    obj2 = {"key1": "value1"}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [KEY2_REMOVED]

    # Test with nested dictionaries
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value2"}}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key1']: value changed from value1 to value2"
    ]

    # Test with a key added in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1"}}
    obj2 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key2']: added in second object"
    ]

    # Test with a key removed in a nested dictionary
    obj1 = {"key1": {"nested_key1": "value1", "nested_key2": "value2"}}
    obj2 = {"key1": {"nested_key1": "value1"}}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key2']: removed in second object"
    ]

    # Test with lists of different lengths
    obj1 = [1, 2, 3]
    obj2 = [1, 2]
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        ": list length changed from 3 to 2"
    ]

    # Test with lists containing different values
    obj1 = [1, 2, 3]
    obj2 = [1, 4, 3]
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        "[1]: value changed from 2 to 4"
    ]

    # Test with completely different structures
    obj1 = {"key1": "value1"}
    obj2 = {"key2": "value2"}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        KEY1_REMOVED,
        KEY2_ADDED,
    ]

    # Test with deeply nested structures
    obj1 = {"key1": {"nested_key1": {"deep_key": "value1"}}}
    obj2 = {"key1": {"nested_key1": {"deep_key": "value2"}}}
    assert compare_productions.find_differences_in_json_objects(obj1, obj2) == [
        "['key1']['nested_key1']['deep_key']: value changed from value1 to value2"
    ]


def test_ensure_list():
    assert value_conversion.ensure_list(None) == []
    assert value_conversion.ensure_list([1, 2, 3]) == [1, 2, 3]
    assert value_conversion.ensure_list((1, 2, 3)) == [1, 2, 3]
    assert value_conversion.ensure_list(5) == [5]
    # Test falsy values are correctly wrapped (not treated as None)
    assert value_conversion.ensure_list(0) == [0]
    assert value_conversion.ensure_list(0.0) == [0.0]
    assert value_conversion.ensure_list("") == [""]
    assert value_conversion.ensure_list(False) == [False]
    assert value_conversion.ensure_list("abc") == ["abc"]
    # json list
    raw_value = '["alice", "bob", "charlie"]'
    assert value_conversion.ensure_list(raw_value) == ["alice", "bob", "charlie"]


def test_parse_typed_sequence():
    assert value_conversion.parse_typed_sequence(None) == []
    assert value_conversion.parse_typed_sequence(["1", "2"], int) == [1, 2]
    assert value_conversion.parse_typed_sequence(("1", "2"), int) == [1, 2]
    assert value_conversion.parse_typed_sequence("1, 2,3", int) == [1, 2, 3]
    assert value_conversion.parse_typed_sequence(["1e6", "2e6"], int) == [1000000, 2000000]
    assert value_conversion.parse_typed_sequence(5, int) == [5]
    assert value_conversion.parse_typed_sequence("2.5") == [2.5]

    with pytest.raises(ValueError, match="Cannot safely cast non-integer value"):
        value_conversion.parse_typed_sequence("1.5", int)


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
    assert not log_inspector.is_safe_tar_member("/etc/passwd")
    assert not log_inspector.is_safe_tar_member("/path/to/file.log")


def test_is_safe_tar_member_unsafe_traversal() -> None:
    assert not log_inspector.is_safe_tar_member("logs/../../../etc/passwd")
    assert not log_inspector.is_safe_tar_member("../logs/test.log")
    assert not log_inspector.is_safe_tar_member("path/../../dangerous/file")


def test_is_safe_tar_member_unsafe_null_byte() -> None:
    assert not log_inspector.is_safe_tar_member("path\x00withNull")
    assert not log_inspector.is_safe_tar_member("file.log\x00.exe")


def test_ensure_string_lists():
    assert value_conversion.ensure_string_lists(None) is None
    assert value_conversion.ensure_string_lists("hello") == ["hello"]
    assert value_conversion.ensure_string_lists(["a", "b", "c"]) == ["a", "b", "c"]
    assert value_conversion.ensure_string_lists(("x", "y")) == ["x", "y"]
    # Tests that integers (non-iterable) raise a TypeError, as per list() behavior
    with pytest.raises(TypeError):
        value_conversion.ensure_string_lists(123)
