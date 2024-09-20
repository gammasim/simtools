#!/usr/bin/python3

import json
import logging

import pytest
import yaml
from astropy.table import Table

from simtools.testing import compare_output

logging.getLogger().setLevel(logging.DEBUG)


@pytest.fixture
def create_json_file(tmp_test_directory):
    def _create_json_file(file_name, content):
        file = tmp_test_directory / file_name
        file.write_text(json.dumps(content), encoding="utf-8")
        return file

    return _create_json_file


@pytest.fixture
def create_yaml_file(tmp_path):
    def _create_yaml_file(file_name, content):
        file = tmp_path / file_name
        with open(file, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
        return file

    return _create_yaml_file


@pytest.fixture
def create_ecsv_file(tmp_path):
    def _create_ecsv_file(file_name, content):
        table = Table(content)
        file_path = tmp_path / file_name
        table.write(file_path, format="ascii.ecsv")
        return file_path

    return _create_ecsv_file


@pytest.fixture
def file_name():
    def _file_name(counter, suffix):
        return f"file{counter}.{suffix}"

    return _file_name


def test_compare_json_files_float_strings(create_json_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert compare_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": "1.23 4.56 7.80"}
    file3 = create_json_file(file_name(3, "json"), content3)
    assert not compare_output.compare_json_or_yaml_files(file1, file3)


def test_compare_json_files_equal_integers(create_json_file, file_name):
    content = {"key": 1, "value": 5}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert compare_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": 7}
    file3 = create_json_file(file_name(3, "json"), content3)
    assert not compare_output.compare_json_or_yaml_files(file1, file3)


def test_compare_yaml_files_float_strings(create_yaml_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_yaml_file(file_name(1, "yaml"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    assert compare_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": "1.23 4.56 7.80"}
    file3 = create_yaml_file(file_name(3, "yaml"), content3)
    assert not compare_output.compare_json_or_yaml_files(file1, file3)


def test_compare_yaml_files_equal_integers(create_yaml_file, file_name):
    content = {"key": 1, "value": 5}
    file1 = create_yaml_file(file_name(1, "yaml"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    assert compare_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": 7}
    file3 = create_yaml_file(file_name(3, "yaml"), content3)
    assert not compare_output.compare_json_or_yaml_files(file1, file3)


def test_compare_ecsv_files_equal(create_ecsv_file, file_name):
    content = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content)

    assert compare_output.compare_ecsv_files(file1, file2)


def test_compare_ecsv_files_different_lengths(create_ecsv_file, file_name):
    content1 = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    content2 = {"col1": [1.1, 2.2], "col2": [4.4, 5.5]}
    file1 = create_ecsv_file(file_name(1, "yaml"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2)

    assert not compare_output.compare_ecsv_files(file1, file2)


def test_compare_ecsv_files_close_values(create_ecsv_file, file_name):
    content1 = {"col1": [1.1001, 2.2001, 3.3001], "col2": [4.4001, 5.5001, 6.6001]}
    content2 = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2)

    assert compare_output.compare_ecsv_files(file1, file2, tolerance=1.0e-3)


def test_compare_ecsv_files_large_difference(create_ecsv_file, file_name):
    content1 = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    content2 = {"col1": [10.1, 20.2, 30.3], "col2": [40.4, 50.5, 60.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2)

    assert not compare_output.compare_ecsv_files(file1, file2)


def test_compare_files_ecsv(create_ecsv_file, file_name):
    content = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content)

    assert compare_output.compare_files(file1, file2)


def test_compare_files_json(create_json_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert compare_output.compare_files(file1, file2)


def test_compare_files_yaml(create_yaml_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_yaml_file(file_name(1, "yaml"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    assert compare_output.compare_files(file1, file2)


def test_compare_files_different_suffixes(create_json_file, create_yaml_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    with pytest.raises(ValueError, match="File suffixes do not match"):
        compare_output.compare_files(file1, file2)


def test_compare_files_unknown_type(tmp_test_directory, file_name):
    file1 = tmp_test_directory / file_name(1, "txt")
    file2 = tmp_test_directory / file_name(2, "txt")
    file1.write_text("dummy content", encoding="utf-8")
    file2.write_text("dummy content", encoding="utf-8")

    assert not compare_output.compare_files(file1, file2)
