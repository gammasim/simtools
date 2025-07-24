#!/usr/bin/python3

import copy
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

import simtools.io.io_handler as io_handler_module

logger = logging.getLogger()


test_file = "test-file.txt"


def test_get_output_directory(args_dict, io_handler, caplog):
    # default adding label
    assert io_handler.get_output_directory(label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/"
    )

    # label and subdirectory
    assert io_handler.get_output_directory(label="test-io-handler", sub_dir="model") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model"
    )

    # path ends with '-output' - no additional 'output' is added
    io_handler_copy = copy.deepcopy(io_handler)
    io_handler_copy.output_path = Path(f"{args_dict['output_path']}/unittest-output")
    assert io_handler_copy.get_output_directory(label="test-io-handler", sub_dir="model") == Path(
        f"{args_dict['output_path']}/unittest-output/test-io-handler/model"
    )

    # FileNotFoundError
    with patch.object(Path, "mkdir", side_effect=FileNotFoundError):
        with caplog.at_level("ERROR"):
            with pytest.raises(FileNotFoundError):
                io_handler.get_output_directory(label="test-io-handler", sub_dir="model")
        assert "Error creating directory" in caplog.text


def test_get_output_directory_plain_output_path(args_dict, io_handler):
    # all following tests: plain_path tests
    io_handler.use_plain_output_path = True

    # plain path (label has no effect), no subdirectories
    assert io_handler.get_output_directory(label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output"
    )

    # plain path, label has no effect, with sub directory as dir_type != 'simtools-result'
    assert io_handler.get_output_directory(label="test-io-handler", sub_dir="model") == Path(
        f"{args_dict['output_path']}/output"
    )


def test_get_output_file(args_dict, io_handler):
    assert io_handler.get_output_file(file_name=test_file, label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/{test_file}"
    )

    assert io_handler.get_output_file(
        file_name=test_file,
        label="test-io-handler",
    ) == Path(f"{args_dict['output_path']}/output/simtools-output/test-io-handler/{test_file}")

    assert io_handler.get_output_file(
        file_name=test_file,
        label="test-io-handler",
        sub_dir="model",
    ) == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model/{test_file}"
    )

    assert io_handler.get_output_file(
        file_name=test_file,
        label="test-io-handler",
        sub_dir="model",
    ) == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model/{test_file}"
    )


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
