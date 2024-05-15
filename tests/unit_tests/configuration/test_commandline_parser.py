#!/usr/bin/python3

import argparse
import logging

import astropy.units as u
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
    assert parser.CommandLineParser.telescope("LSTN-01") == "LSTN-01"
    assert parser.CommandLineParser.telescope("MSTN-design") == "MSTN-design"

    with pytest.raises(ValueError, match=r"Invalid name Whipple"):
        assert parser.CommandLineParser.telescope("Whipple")

    with pytest.raises(ValueError, match=r"Invalid name LST"):
        assert parser.CommandLineParser.telescope("LST")


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


def test_zenith_angle(caplog):
    assert parser.CommandLineParser.zenith_angle(0).value == pytest.approx(0.0)
    assert parser.CommandLineParser.zenith_angle(45).value == pytest.approx(45.0)
    assert parser.CommandLineParser.zenith_angle(90).value == pytest.approx(90.0)
    assert isinstance(parser.CommandLineParser.zenith_angle(0), u.Quantity)
    assert parser.CommandLineParser.zenith_angle("0 deg").value == pytest.approx(0.0)
    assert parser.CommandLineParser.zenith_angle("45 deg").value == pytest.approx(45.0)
    assert parser.CommandLineParser.zenith_angle("90 deg").value == pytest.approx(90.0)

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

    with pytest.raises(TypeError):
        parser.CommandLineParser.zenith_angle("North")
        assert "The zenith angle provided is not a valid numeric value" in caplog.text


def test_azimuth_angle(caplog):
    assert parser.CommandLineParser.azimuth_angle(0).value == pytest.approx(0.0)
    assert parser.CommandLineParser.azimuth_angle(45).value == pytest.approx(45.0)
    assert parser.CommandLineParser.azimuth_angle(90).value == pytest.approx(90.0)
    assert isinstance(parser.CommandLineParser.azimuth_angle(0), u.Quantity)
    assert parser.CommandLineParser.azimuth_angle("0 deg").value == pytest.approx(0.0)
    assert parser.CommandLineParser.azimuth_angle("45 deg").value == pytest.approx(45.0)
    assert parser.CommandLineParser.azimuth_angle("90 deg").value == pytest.approx(90.0)

    assert parser.CommandLineParser.azimuth_angle("North").value == pytest.approx(0.0)
    assert parser.CommandLineParser.azimuth_angle("South").value == pytest.approx(180.0)
    assert parser.CommandLineParser.azimuth_angle("East").value == pytest.approx(90.0)
    assert parser.CommandLineParser.azimuth_angle("West").value == pytest.approx(270.0)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided azimuth angle, -1.0, is outside of the allowed \[0, 360\] interval",
    ):
        parser.CommandLineParser.azimuth_angle(-1)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided azimuth angle, 370.0, is outside of the allowed \[0, 360\] interval",
    ):
        parser.CommandLineParser.azimuth_angle(370)
    with pytest.raises(argparse.ArgumentTypeError):
        parser.CommandLineParser.azimuth_angle("TEST")
        assert "The azimuth angle can only be a number or one of" in caplog.text
    with pytest.raises(TypeError):
        parser.CommandLineParser.azimuth_angle([0, 10])
        assert "is not a valid number nor one of (north, south, east, west)" in caplog.text


def test_initialize_default_arguments():
    # default arguments
    _parser_1 = parser.CommandLineParser()
    _parser_1.initialize_default_arguments()
    job_groups = _parser_1._action_groups
    for group in job_groups:
        assert str(group.title) in [
            "positional arguments",
            "optional arguments",
            "options",
            "paths",
            "configuration",
            "execution",
        ]

    _parser_2 = parser.CommandLineParser()
    _parser_2.initialize_default_arguments(output=True)
    job_groups = _parser_2._action_groups
    assert "output" in [str(group.title) for group in job_groups]

    # simulation model is none
    _parser_n = parser.CommandLineParser()
    _parser_n.initialize_default_arguments(simulation_model=None)

    # model version only, no site or telescope
    _parser_v = parser.CommandLineParser()
    _parser_v.initialize_default_arguments(simulation_model=["version"])
    job_groups = _parser_v._action_groups
    assert "simulation model" in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == "simulation model":
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert all(action.dest != "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # site model can exist without a telescope model
    _parser_s = parser.CommandLineParser()
    _parser_s.initialize_default_arguments(simulation_model=["site"])
    job_groups = _parser_s._action_groups
    assert "simulation model" in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == "simulation model":
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # no telescope model without site model
    _parser_t = parser.CommandLineParser()
    _parser_t.initialize_default_arguments(simulation_model=["telescope", "site"])
    job_groups = _parser_t._action_groups
    assert "simulation model" in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == "simulation model":
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert any(action.dest == "telescope" for action in group._group_actions)

    _parser_5 = parser.CommandLineParser()
    _parser_5.initialize_default_arguments(job_submission=True)
    job_groups = _parser_5._action_groups
    assert "job submission" in [str(group.title) for group in job_groups]

    _parser_6 = parser.CommandLineParser()
    _parser_6.initialize_default_arguments(db_config=True)
    job_groups = _parser_6._action_groups
    assert "MongoDB configuration" in [str(group.title) for group in job_groups]
