#!/usr/bin/python3

import pytest
import logging
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_derived_output_directory(cfg_setup):

    path = Path(cfg.get("outputLocation")).joinpath("derived")
    assert io.getDerivedOutputDirectory(
            filesLocation=cfg.get("outputLocation"),
            label="test-derived"
        ) == Path(f"{cfg.get('outputLocation')}/simtools-output/test-derived/derived/")
