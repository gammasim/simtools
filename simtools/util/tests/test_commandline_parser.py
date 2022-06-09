#!/usr/bin/python3

import argparse
import logging
import pytest

import simtools.util.commandline_parser as parser

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_efficiency_interval():

    assert parser.CommandLineParser.efficiency_interval(0.5) == pytest.approx(0.5)
    assert parser.CommandLineParser.efficiency_interval(0.) == pytest.approx(0.)
    assert parser.CommandLineParser.efficiency_interval(1.) == pytest.approx(1.)

    with pytest.raises(argparse.ArgumentTypeError,
                       match=r"1.5 outside of allowed \[0,1\] interval"):
        parser.CommandLineParser.efficiency_interval(1.5)
    with pytest.raises(argparse.ArgumentTypeError,
                       match=r"-8.5 outside of allowed \[0,1\] interval"):
        parser.CommandLineParser.efficiency_interval(-8.5)
