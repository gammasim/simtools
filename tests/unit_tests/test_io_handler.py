#!/usr/bin/python3

import logging
from pathlib import Path

import simtools.io_handler as io

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_output_directory(args_dict):

    assert io.getOutputDirectory(
        filesLocation=args_dict["output_path"], label="test-io-handler"
    ) == Path(f"{args_dict['output_path']}/simtools-output/test-io-handler/")

    assert io.getOutputDirectory(
        filesLocation=args_dict["output_path"], label="test-io-handler", test=True
    ) == Path(f"{args_dict['output_path']}/test-output/test-io-handler/")

    assert io.getOutputDirectory(
        filesLocation=args_dict["output_path"], label="test-io-handler", dirType="model"
    ) == Path(f"{args_dict['output_path']}/simtools-output/test-io-handler/model")

    assert io.getOutputDirectory(
        filesLocation=args_dict["output_path"], label="test-io-handler", dirType="model", test=True
    ) == Path(f"{args_dict['output_path']}/test-output/test-io-handler/model")


def test_get_output_file(args_dict):

    assert io.getOutputFile(
        fileName="test-file.txt", filesLocation=args_dict["output_path"], label="test-io-handler"
    ) == Path(f"{args_dict['output_path']}/simtools-output/test-io-handler/test-file.txt")

    assert io.getOutputFile(
        fileName="test-file.txt",
        filesLocation=args_dict["output_path"],
        label="test-io-handler",
        test=True,
    ) == Path(f"{args_dict['output_path']}/test-output/test-io-handler/test-file.txt")

    assert io.getOutputFile(
        fileName="test-file.txt",
        filesLocation=args_dict["output_path"],
        label="test-io-handler",
        dirType="model",
    ) == Path(f"{args_dict['output_path']}/simtools-output/test-io-handler/model/test-file.txt")

    assert io.getOutputFile(
        fileName="test-file.txt",
        filesLocation=args_dict["output_path"],
        label="test-io-handler",
        dirType="model",
        test=True,
    ) == Path(f"{args_dict['output_path']}/test-output/test-io-handler/model/test-file.txt")


def test_get_data_file(args_dict):

    assert (
        io.getInputDataFile(
            dataLocation=args_dict["data_path"],
            parentDir="test-io-handler",
            fileName="test-file.txt",
        )
        == Path(f"{args_dict['data_path']}/test-io-handler/test-file.txt").absolute()
    )

    assert (
        io.getInputDataFile(
            dataLocation=args_dict["data_path"], fileName="test-file.txt", test=True
        )
        == Path("tests/resources/test-file.txt").absolute()
    )
