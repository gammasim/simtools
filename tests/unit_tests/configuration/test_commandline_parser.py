#!/usr/bin/python3

import argparse
import logging

import pytest

import simtools.configuration.commandline_parser as parser

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_site():

    assert parser.CommandLineParser.site("North") == "North"
    assert parser.CommandLineParser.site("South") == "South"

    with pytest.raises(ValueError, match=r"Invalid name East"):
        parser.CommandLineParser.site("East")


def test_telescope():

    assert parser.CommandLineParser.telescope("LST-1") == "LST-1"
    assert parser.CommandLineParser.telescope("MST-FlashCam") == "MST-FlashCam"

    with pytest.raises(ValueError, match=r"Invalid name Whipple"):
        assert parser.CommandLineParser.telescope("Whipple")


def test_efficiency_interval():

    assert parser.CommandLineParser.efficiency_interval(0.5) == pytest.approx(0.5)
    assert parser.CommandLineParser.efficiency_interval(0.0) == pytest.approx(0.0)
    assert parser.CommandLineParser.efficiency_interval(1.0) == pytest.approx(1.0)

    with pytest.raises(
        argparse.ArgumentTypeError, match=r"1.5 outside of allowed \[0,1\] interval"
    ):
        parser.CommandLineParser.efficiency_interval(1.5)
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"-8.5 outside of allowed \[0,1\] interval"
    ):
        parser.CommandLineParser.efficiency_interval(-8.5)


def test_zenith_angle():

    assert parser.CommandLineParser.zenith_angle(0) == pytest.approx(0.0)
    assert parser.CommandLineParser.zenith_angle(45) == pytest.approx(45.0)
    assert parser.CommandLineParser.zenith_angle(90) == pytest.approx(90.0)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided zenith angle, -1.0, is outside of the allowed \[0, 180\] interval",
    ):
        parser.CommandLineParser.zenith_angle(-1)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided zenith angle, 190.0, is outside of the allowed \[0, 180\] interval",
    ):
        parser.CommandLineParser.zenith_angle(190)
