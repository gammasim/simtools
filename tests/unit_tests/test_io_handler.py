#!/usr/bin/python3

import logging
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_output_directory(cfg_setup):

    assert io.getOutputDirectory(
        filesLocation=cfg.get("outputLocation"), label="test-derived"
    ) == Path(f"{cfg.get('outputLocation')}/simtools-output/test-derived/")
