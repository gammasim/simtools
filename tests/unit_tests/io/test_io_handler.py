#!/usr/bin/python3

import copy
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

import simtools.io.io_handler as io_handler_module

logger = logging.getLogger()


test_file = "test-file.txt"


def test_get_output_directory(args_dict, io_handler):
    # default adding label
    assert io_handler.get_output_directory() == Path(f"{args_dict['output_path']}/output/")

    # label and subdirectory
    assert io_handler.get_output_directory(sub_dir="model") == Path(
        f"{args_dict['output_path']}/output/model/"
    )

    # path ends with '-output' - no additional 'output' is added
    io_handler_copy = copy.deepcopy(io_handler)
    io_handler_copy.output_path = Path(f"{args_dict['output_path']}/unittest-output")
    assert io_handler_copy.get_output_directory(sub_dir="model") == Path(
        f"{args_dict['output_path']}/unittest-output/model"
    )

    # FileNotFoundError
    with patch.object(Path, "mkdir", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match=r"^Error creating directory"):
            io_handler.get_output_directory(sub_dir="model")


def test_get_output_file(args_dict, io_handler):
    assert io_handler.get_output_file(file_name=test_file) == Path(
        f"{args_dict['output_path']}/output/{test_file}"
    )

    assert io_handler.get_output_file(
        file_name=test_file,
        sub_dir="test-io-handler",
    ) == Path(f"{args_dict['output_path']}/output/test-io-handler/{test_file}")


def test_get_data_file(args_dict, io_handler):
    assert (
        io_handler.get_input_data_file(
            parent_dir="test-io-handler",
            file_name=test_file,
        )
        == Path(f"{args_dict['data_path']}/test-io-handler/{test_file}").absolute()
    )

    assert (
        io_handler.get_input_data_file(file_name=test_file, test=True)
        == Path(f"tests/resources/{test_file}").absolute()
    )

    io_handler.data_path = None
    with pytest.raises(io_handler_module.IncompleteIOHandlerInitError):
        io_handler.get_input_data_file(file_name=test_file)


def test_get_model_configuration_directory(args_dict, io_handler):
    model_version = "1.0.0"
    label = "test-io-handler"

    # Test directory creation
    expected_path = Path(f"{args_dict['output_path']}/output/{label}/model/{model_version}")
    assert (
        io_handler.get_model_configuration_directory(sub_dir=label, model_version=model_version)
        == expected_path
    )

    # Test FileNotFoundError
    with patch.object(Path, "mkdir", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match=r"^Error creating directory"):
            io_handler.get_model_configuration_directory(sub_dir=label, model_version=model_version)


def test_mkdir(io_handler):
    # Test successful directory creation
    test_path = Path("tests/tmp/test-dir")
    created_path = io_handler._mkdir(test_path)
    assert created_path == test_path.absolute()
    assert test_path.exists()

    # Cleanup
    test_path.rmdir()

    # Test FileNotFoundError
    with patch.object(Path, "mkdir", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match=r"^Error creating directory"):
            io_handler._mkdir(test_path)
