#!/usr/bin/python3

import logging
from pathlib import Path

import pytest
import yaml

import simtools.io_operations.ascii_handler as ascii_handler
from simtools.constants import MODEL_PARAMETER_METASCHEMA, MODEL_PARAMETER_SCHEMA_PATH

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

    with pytest.raises(IndexError, match=FAILED_TO_READ_FILE_ERROR):
        ascii_handler.collect_data_from_file(MODEL_PARAMETER_METASCHEMA, 999)

    # document type not supported
    with pytest.raises(TypeError, match=FAILED_TO_READ_FILE_ERROR):
        ascii_handler.collect_data_from_file(
            "tests/resources/run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst"
        )


def test_collect_data_from_file_exceptions(io_handler, caplog) -> None:
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


def test_collect_dict_from_url(io_handler) -> None:
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
