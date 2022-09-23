#!/usr/bin/python3

import logging
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_output_directory(cfg_setup):

    assert io.getOutputDirectory(label="test-io-handler") == Path(
        f"{cfg.get('outputLocation')}/simtools-output/test-io-handler/"
    )

    assert io.getOutputDirectory(label="test-io-handler", test=True) == Path(
        f"{cfg.get('outputLocation')}/test-output/test-io-handler/"
    )

    assert io.getOutputDirectory(label="test-io-handler", dirType="model") == Path(
        f"{cfg.get('outputLocation')}/simtools-output/test-io-handler/model"
    )

    assert io.getOutputDirectory(label="test-io-handler", dirType="model", test=True) == Path(
        f"{cfg.get('outputLocation')}/test-output/test-io-handler/model"
    )


def test_get_output_file(cfg_setup):

    assert io.getOutputFile(fileName="test-file.txt", label="test-io-handler") == Path(
        f"{cfg.get('outputLocation')}/simtools-output/test-io-handler/test-file.txt"
    )

    assert io.getOutputFile(fileName="test-file.txt", label="test-io-handler", test=True) == Path(
        f"{cfg.get('outputLocation')}/test-output/test-io-handler/test-file.txt"
    )

    assert io.getOutputFile(
        fileName="test-file.txt", label="test-io-handler", dirType="model"
    ) == Path(f"{cfg.get('outputLocation')}/simtools-output/test-io-handler/model/test-file.txt")

    assert io.getOutputFile(
        fileName="test-file.txt", label="test-io-handler", dirType="model", test=True
    ) == Path(f"{cfg.get('outputLocation')}/test-output/test-io-handler/model/test-file.txt")


def test_get_data_file(cfg_setup):

    assert (
        io.getInputDataFile(
            parentDir="test-io-handler",
            fileName="test-file.txt",
        )
        == Path(f"{cfg.get('dataLocation')}/test-io-handler/test-file.txt").absolute()
    )

    assert (
        io.getInputDataFile(fileName="test-file.txt", test=True)
        == Path("tests/resources/test-file.txt").absolute()
    )
