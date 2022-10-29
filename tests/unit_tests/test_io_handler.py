#!/usr/bin/python3

import logging
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_output_directory(args_dict, io_handler):

    assert io_handler.getOutputDirectory(label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/"
    )

    assert io_handler.getOutputDirectory(label="test-io-handler", test=True) == Path(
        f"{args_dict['output_path']}/output/test-output/test-io-handler/"
    )

    assert io_handler.getOutputDirectory(label="test-io-handler", dirType="model") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model"
    )

    assert io_handler.getOutputDirectory(
        label="test-io-handler", dirType="model", test=True
    ) == Path(f"{args_dict['output_path']}/output/test-output/test-io-handler/model")


def test_get_output_file(args_dict, io_handler):

    assert io_handler.getOutputFile(fileName="test-file.txt", label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/test-file.txt"
    )

    assert io_handler.getOutputFile(
        fileName="test-file.txt",
        label="test-io-handler",
        test=True,
    ) == Path(f"{args_dict['output_path']}/output/test-output/test-io-handler/test-file.txt")

    assert io_handler.getOutputFile(
        fileName="test-file.txt",
        label="test-io-handler",
        dirType="model",
    ) == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model/test-file.txt"
    )

    assert io_handler.getOutputFile(
        fileName="test-file.txt",
        label="test-io-handler",
        dirType="model",
        test=True,
    ) == Path(f"{args_dict['output_path']}/output/test-output/test-io-handler/model/test-file.txt")


def test_get_data_file(args_dict, io_handler):

    assert (
        io_handler.getInputDataFile(
            parentDir="test-io-handler",
            fileName="test-file.txt",
        )
        == Path(f"{args_dict['data_path']}/test-io-handler/test-file.txt").absolute()
    )

    assert (
        io_handler.getInputDataFile(fileName="test-file.txt", test=True)
        == Path("tests/resources/test-file.txt").absolute()
    )
