#!/usr/bin/python3

import json
import logging
from pathlib import Path
from unittest.mock import mock_open, patch

import astropy.units as u
import numpy as np
import pytest
import yaml

import simtools.io.ascii_handler as ascii_handler
from simtools.constants import MODEL_PARAMETER_METASCHEMA, MODEL_PARAMETER_SCHEMA_PATH
from simtools.io.ascii_handler import read_file_encoded_in_utf_or_latin

FAILED_TO_READ_FILE_ERROR = r"^Failed to read file"
url_simtools = "https://raw.githubusercontent.com/gammasim/simtools/main/"


def test_collect_dict_data(io_handler) -> None:
    dict_for_yaml = {"k3": {"kk3": 4, "kk4": 3.0}, "k4": ["bla", 2]}
    test_yaml_file = io_handler.get_output_file(file_name="test_collect_dict_data.yml")
    if not Path(test_yaml_file).exists():
        with open(test_yaml_file, "w") as output:
            yaml.dump(dict_for_yaml, output, sort_keys=False)

    d2 = ascii_handler.collect_data_from_file(test_yaml_file)
    assert "k3" in d2.keys()
    assert d2["k4"] == ["bla", 2]

    _lines = ascii_handler.collect_data_from_file("tests/resources/test_file.list")
    assert len(_lines) == 2

    # astropy-type yaml file
    _file = "tests/resources/corsikaConfigTest_astropy_headers.yml"
    _dict = ascii_handler.collect_data_from_file(_file)
    assert isinstance(_dict, dict)
    assert len(_dict) > 0

    # file with several documents
    _list = ascii_handler.collect_data_from_file(MODEL_PARAMETER_METASCHEMA)
    assert isinstance(_list, list)
    assert len(_list) > 0

    # file with several documents - get first document
    _dict = ascii_handler.collect_data_from_file(MODEL_PARAMETER_METASCHEMA, 0)
    assert _dict["schema_version"] != "0.1.0"

    with pytest.raises(IndexError, match=r"^Failed to read file"):
        ascii_handler.collect_data_from_file(MODEL_PARAMETER_METASCHEMA, 999)

    # document type not supported
    with pytest.raises(TypeError, match=FAILED_TO_READ_FILE_ERROR):
        ascii_handler.collect_data_from_file(
            "tests/resources/run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst"
        )


def test_collect_data_from_file_exceptions(io_handler) -> None:
    """Test error handling in collect_data_from_file."""
    # Create an invalid YAML file
    test_file = io_handler.get_output_file(file_name="invalid.yml")
    with open(test_file, "w") as f:
        f.write("invalid: {\n")  # Invalid YAML syntax

    # Test with invalid YAML file
    with pytest.raises(Exception, match=FAILED_TO_READ_FILE_ERROR):
        ascii_handler.collect_data_from_file(test_file)

    # Test with invalid JSON file
    test_json = io_handler.get_output_file(file_name="invalid.json")
    with open(test_json, "w") as f:
        f.write("{invalid json")

    with pytest.raises(Exception, match=r"^JSONDecodeError"):
        ascii_handler.collect_data_from_file(test_json)

    # Test with unsupported file extension
    test_unsupported = io_handler.get_output_file(file_name="test.xyz")
    with open(test_unsupported, "w") as f:
        f.write("some content")

    with pytest.raises(TypeError, match=r"^Failed to read"):
        ascii_handler.collect_data_from_file(test_unsupported)


def test_collect_dict_from_url() -> None:
    _file = MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"
    _reference_dict = ascii_handler.collect_data_from_file(_file)

    _file = "src/simtools/schemas/model_parameters/num_gains.schema.yml"

    _url = url_simtools
    _url_dict = ascii_handler.collect_data_from_http(_url + _file)

    assert _reference_dict == _url_dict

    _dict = ascii_handler.collect_data_from_file(_url + _file)
    assert isinstance(_dict, dict)
    assert len(_dict) > 0

    _url = "https://raw.githubusercontent.com/gammasim/simtools/not_main/"
    with pytest.raises(FileNotFoundError):
        ascii_handler.collect_data_from_http(_url + _file)

    # yaml file with astropy header
    _url = url_simtools
    _url_dict = ascii_handler.collect_data_from_http(
        _url + "tests/resources/corsikaConfigTest_astropy_headers.yml"
    )
    assert isinstance(_url_dict, dict)
    assert len(_dict) > 0

    # simple list
    _url = url_simtools
    _url_list = ascii_handler.collect_data_from_http(_url + "tests/resources/test_file.list")
    assert isinstance(_url_list, list)
    assert len(_url_list) == 2


def test_collect_data_dict_from_json():
    _file = "tests/resources/reference_point_altitude.json"
    data = ascii_handler.collect_data_from_file(_file)
    assert len(data) == 6
    assert data["unit"] == "m"


def test_collect_data_from_http():
    _file = "src/simtools/schemas/model_parameters/num_gains.schema.yml"
    url = url_simtools

    data = ascii_handler.collect_data_from_http(url + _file)
    assert isinstance(data, dict)

    _file = "tests/resources/reference_point_altitude.json"
    data = ascii_handler.collect_data_from_http(url + _file)
    assert isinstance(data, dict)

    _file = (
        "tests/resources/proton_run000201_za20deg_azm000deg_North_alpha_6.0.0_test_file.simtel.zst"
    )
    with pytest.raises(TypeError):
        ascii_handler.collect_data_from_http(url + _file)

    url = "https://raw.githubusercontent.com/gammasim/simtools/not_right/"
    with pytest.raises(FileNotFoundError):
        ascii_handler.collect_data_from_http(url + _file)


def test_read_file_encoded_in_utf_or_latin(tmp_test_directory, caplog) -> None:
    """
    Test the read_file_encoded_in_utf_or_latin function.
    """

    # Test with a UTF-8 encoded file.
    utf8_file = tmp_test_directory / "utf8_file.txt"
    utf8_content = "This is a UTF-8 encoded file.\n"
    with open(utf8_file, "w", encoding="utf-8") as file:
        file.write(utf8_content)
    lines = ascii_handler.read_file_encoded_in_utf_or_latin(utf8_file)
    assert lines == [utf8_content]
    assert ascii_handler.is_utf8_file(utf8_file) is True

    # Test with a Latin-1 encoded file.
    latin1_file = tmp_test_directory / "latin1_file.txt"
    latin1_content = "This is a Latin-1 encoded file with latin character ñ.\n"
    with open(latin1_file, "w", encoding="latin-1") as file:
        file.write(latin1_content)
    with caplog.at_level(logging.DEBUG):
        lines = ascii_handler.read_file_encoded_in_utf_or_latin(latin1_file)
        assert lines == [latin1_content]
    assert "Unable to decode file using UTF-8. Trying Latin-1." in caplog.text
    assert ascii_handler.is_utf8_file(latin1_file) is False

    # I could not find a way to create a file that cannot be decoded with Latin-1
    # and raises a UnicodeDecodeError. I left the raise statement in the function
    # in case we ever encounter such a file, but I cannot test it here.

    # Test with a non-existent file.
    non_existent_file = tmp_test_directory / "non_existent_file.txt"
    with pytest.raises(FileNotFoundError):
        ascii_handler.read_file_encoded_in_utf_or_latin(non_existent_file)

    with pytest.raises(FileNotFoundError):
        ascii_handler.is_utf8_file(non_existent_file)


def test_read_file_encoded_in_utf_or_latin_unicode_decode_error():
    """Test read_file_encoded_in_utf_or_latin raises UnicodeDecodeError."""
    mock_file_name = "mock_file.txt"

    # Mock open to raise UnicodeDecodeError for both UTF-8 and Latin-1
    with patch("builtins.open", mock_open()) as mocked_open:
        mocked_open.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "mock reason")
        with pytest.raises(
            UnicodeDecodeError, match=r"Unable to decode file.*using UTF-8 or Latin-1"
        ):
            read_file_encoded_in_utf_or_latin(mock_file_name)


def test_json_numpy_encoder():
    encoder = ascii_handler.JsonNumpyEncoder()
    assert isinstance(encoder.default(np.float64(3.14)), float)
    assert isinstance(encoder.default(np.int64(3.14)), int)
    assert isinstance(encoder.default(np.array([])), list)
    assert isinstance(encoder.default(u.Unit("m")), str)
    assert encoder.default(u.Unit("")) is None
    assert isinstance(encoder.default(u.Unit("m/s")), str)
    assert isinstance(encoder.default(np.bool_(True)), bool)

    with pytest.raises(TypeError):
        encoder.default("abc")


def test_write_to_yaml(tmp_test_directory):
    """Test the _write_to_yaml function."""
    test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested_key": "nested_value"}}
    output_file = tmp_test_directory / "test_output.yaml"

    # Test writing YAML file
    ascii_handler._write_to_yaml(test_data, output_file, sort_keys=False)
    assert output_file.exists()

    # Verify the content of the written file
    with open(output_file, encoding="utf-8") as file:
        loaded_data = yaml.safe_load(file)
    assert loaded_data == test_data

    # Test writing YAML file with sorted keys
    sorted_output_file = tmp_test_directory / "test_output_sorted.yaml"
    ascii_handler._write_to_yaml(test_data, sorted_output_file, sort_keys=True)
    assert sorted_output_file.exists()

    # Verify the content of the sorted file
    with open(sorted_output_file, encoding="utf-8") as file:
        loaded_data_sorted = yaml.safe_load(file)
    assert list(loaded_data_sorted.keys()) == sorted(test_data.keys())


def test_write_to_json(tmp_test_directory):
    """Test the _write_to_json function."""
    test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested_key": "nested_value"}}
    output_file = tmp_test_directory / "test_output.json"

    # Test writing JSON file
    ascii_handler._write_to_json(test_data, output_file, sort_keys=False, numpy_types=False)
    assert output_file.exists()

    # Verify the content of the written file
    with open(output_file, encoding="utf-8") as file:
        loaded_data = json.load(file)
    assert loaded_data == test_data

    # Test writing JSON file with sorted keys
    sorted_output_file = tmp_test_directory / "test_output_sorted.json"
    ascii_handler._write_to_json(test_data, sorted_output_file, sort_keys=True, numpy_types=False)
    assert sorted_output_file.exists()

    # Verify the content of the sorted file
    with open(sorted_output_file, encoding="utf-8") as file:
        loaded_data_sorted = json.load(file)
    assert list(loaded_data_sorted.keys()) == sorted(test_data.keys())

    # Test writing JSON file with numpy types
    numpy_data = {"array": np.array([1, 2, 3]), "float": np.float64(3.14), "int": np.int64(42)}
    numpy_output_file = tmp_test_directory / "test_numpy_output.json"
    ascii_handler._write_to_json(numpy_data, numpy_output_file, sort_keys=False, numpy_types=True)
    assert numpy_output_file.exists()

    # Verify the content of the file with numpy types
    with open(numpy_output_file, encoding="utf-8") as file:
        loaded_numpy_data = json.load(file)
    assert loaded_numpy_data == {
        "array": [1, 2, 3],
        "float": 3.14,
        "int": 42,
    }


def test_write_data_to_file_json(tmp_test_directory):
    """Test write_data_to_file for JSON files."""
    test_data = {"key1": "value1", "key2": [1, 2, 3]}
    output_file = tmp_test_directory / "test_output.json"

    ascii_handler.write_data_to_file(test_data, output_file, sort_keys=False, numpy_types=False)
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as file:
        loaded_data = json.load(file)
    assert loaded_data == test_data


def test_write_data_to_file_yaml(tmp_test_directory):
    """Test write_data_to_file for YAML files."""
    test_data = {"key1": "value1", "key2": [1, 2, 3]}
    output_file = tmp_test_directory / "test_output.yaml"

    ascii_handler.write_data_to_file(test_data, output_file, sort_keys=False, numpy_types=False)
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as file:
        loaded_data = yaml.safe_load(file)
    assert loaded_data == test_data


def test_write_data_to_file_invalid_extension(tmp_test_directory):
    """Test write_data_to_file with unsupported file extension."""
    test_data = {"key1": "value1"}
    output_file = tmp_test_directory / "test_output.txt"

    with pytest.raises(ValueError, match="Unsupported file type"):
        ascii_handler.write_data_to_file(test_data, output_file)
