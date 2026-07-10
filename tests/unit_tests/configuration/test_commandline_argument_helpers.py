#!/usr/bin/python3

import argparse

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose

import simtools.configuration.commandline_argument_helpers as helpers
import simtools.configuration.commandline_parser as parser


def test_scientific_int():
    assert helpers.scientific_int("100") == 100
    assert helpers.scientific_int("0") == 0
    assert helpers.scientific_int("42") == 42
    assert helpers.scientific_int(100) == 100
    assert helpers.scientific_int(0) == 0
    assert helpers.scientific_int(100.0) == 100
    assert helpers.scientific_int("1e3") == 1000
    assert helpers.scientific_int("1E3") == 1000
    assert helpers.scientific_int("2.5e3") == 2500
    assert helpers.scientific_int("1e7") == 10000000
    assert helpers.scientific_int("5.5e6") == 5500000
    assert helpers.scientific_int("1.23e4") == 12300
    assert helpers.scientific_int("-100") == -100
    assert helpers.scientific_int("-1e3") == -1000
    assert helpers.scientific_int("123.0") == 123
    assert helpers.scientific_int("42.0") == 42
    assert helpers.scientific_int(100.0) == 100
    assert helpers.scientific_int(0.0) == 0
    assert helpers.scientific_int("-10.0") == -10
    assert helpers.scientific_int("1.5e1") == 15
    assert helpers.scientific_int("2.0e3") == 2000

    with pytest.raises(argparse.ArgumentTypeError, match=r"Invalid integer value: '1.9'"):
        helpers.scientific_int("1.9")
    with pytest.raises(argparse.ArgumentTypeError, match=r"Invalid integer value: '42.5'"):
        helpers.scientific_int("42.5")
    with pytest.raises(argparse.ArgumentTypeError):
        helpers.scientific_int(42.9)
    with pytest.raises(argparse.ArgumentTypeError):
        helpers.scientific_int(3.14)
    with pytest.raises(argparse.ArgumentTypeError, match=r"Invalid integer value: '1.23e1'"):
        helpers.scientific_int("1.23e1")
    with pytest.raises(argparse.ArgumentTypeError, match=r"Invalid integer value: 'abc'"):
        helpers.scientific_int("abc")
    with pytest.raises(argparse.ArgumentTypeError, match=r"Invalid integer value: 'not_a_number'"):
        helpers.scientific_int("not_a_number")
    with pytest.raises(argparse.ArgumentTypeError, match=r"Invalid integer value: 'None'"):
        helpers.scientific_int(None)


def test_site():
    assert helpers.site("North") == "North"
    assert helpers.site("South") == "South"

    with pytest.raises(ValueError, match=r"Invalid name East"):
        helpers.site("East")


def test_telescope():
    assert helpers.telescope("LSTN-01") == "LSTN-01"
    assert helpers.telescope("MSTx-NectarCam") == "MSTx-NectarCam"

    with pytest.raises(ValueError, match=r"Invalid name Whipple"):
        helpers.telescope("Whipple")
    with pytest.raises(ValueError, match=r"Invalid name LST"):
        helpers.telescope("LST")


def test_instrument():
    assert helpers.instrument("OBS-North") == "North"
    assert helpers.instrument("OBS-South") == "South"
    assert helpers.instrument("LSTN-01") == "LSTN-01"
    assert helpers.instrument("LSTN-design") == "LSTN-design"
    assert helpers.instrument("MSTS-FlashCam") == "MSTS-FlashCam"

    with pytest.raises(ValueError, match=r"Invalid name North"):
        helpers.instrument("North")
    with pytest.raises(ValueError, match=r"Invalid name South"):
        helpers.instrument("South")
    with pytest.raises(ValueError, match=r"Invalid name InvalidName"):
        helpers.instrument("InvalidName")


def test_efficiency_interval():
    assert helpers.efficiency_interval(0.5) == pytest.approx(0.5)
    assert helpers.efficiency_interval(0.0) == pytest.approx(0.0)
    assert helpers.efficiency_interval(1.0) == pytest.approx(1.0)

    with pytest.raises(
        argparse.ArgumentTypeError, match=r"1.5 outside of allowed \[0,1\] interval"
    ):
        helpers.efficiency_interval(1.5)
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"-8.5 outside of allowed \[0,1\] interval"
    ):
        helpers.efficiency_interval(-8.5)


def test_zenith_angle(caplog):
    assert helpers.zenith_angle(0).value == pytest.approx(0.0)
    assert helpers.zenith_angle(45).value == pytest.approx(45.0)
    assert helpers.zenith_angle(90).value == pytest.approx(90.0)
    assert isinstance(helpers.zenith_angle(0), u.Quantity)
    assert helpers.zenith_angle("0 deg").value == pytest.approx(0.0)
    assert helpers.zenith_angle("45 deg").value == pytest.approx(45.0)
    assert helpers.zenith_angle("90 deg").value == pytest.approx(90.0)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided zenith angle, -1.0, is outside of the allowed \[0, 180\] interval",
    ):
        helpers.zenith_angle(-1)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided zenith angle, 190.0, is outside of the allowed \[0, 180\] interval",
    ):
        helpers.zenith_angle(190)

    with caplog.at_level("WARNING"):
        with pytest.raises(TypeError):
            helpers.zenith_angle("North")
    assert "The zenith angle provided is not a valid numeric" in caplog.text


def test_parse_quantity_pair():
    for test_string in ["100 GeV 5 TeV", "100GeV 5TeV", "100 GeV 5 TeV", "100GeV 5 TeV"]:
        e_pair = helpers.parse_quantity_pair(test_string)
        assert e_pair[0].value == pytest.approx(100.0)
        assert e_pair[0].unit == u.GeV
        assert e_pair[1].value == pytest.approx(5.0)
        assert e_pair[1].unit == u.TeV

    q1, q2 = helpers.parse_quantity_pair("(<Quantity 200. GeV>, <Quantity 500. GeV>)")
    assert q1 == 200 * u.GeV
    assert q2 == 500 * u.GeV

    with pytest.raises(ValueError, match=r"Input string does not contain exactly two quantities."):
        helpers.parse_quantity_pair("100 GeV 5 TeV 20 PeV")
    with pytest.raises(
        ValueError, match=r"^Could not parse quantities: 'abc' did not parse as unit"
    ):
        helpers.parse_quantity_pair("100 GeV 5 abc")
    with pytest.raises(
        ValueError, match=r'^Could not parse quantities: Cannot parse "eV" as a Quantity.'
    ):
        helpers.parse_quantity_pair("a GeV 5 TeV")


def test_parse_integer_and_quantity():
    for test_string in ["5 1500 m", "5 1500m", "5 1500.0 m", "(5, <Quantity 1500 m>)"]:
        c_pair = helpers.parse_integer_and_quantity(test_string)
        assert c_pair[0] == 5
        assert c_pair[1].value == pytest.approx(1500.0)
        assert c_pair[1].unit == u.m

    with pytest.raises(ValueError, match=r"^'abc' did not parse as unit:"):
        helpers.parse_integer_and_quantity("5 5 abc")
    with pytest.raises(
        ValueError, match=r"Input string does not contain an integer and a astropy quantity."
    ):
        helpers.parse_integer_and_quantity("0 m 5 m")


def test_azimuth_angle(caplog):
    assert helpers.azimuth_angle(0).value == pytest.approx(0.0)
    assert helpers.azimuth_angle(45).value == pytest.approx(45.0)
    assert helpers.azimuth_angle(90).value == pytest.approx(90.0)
    assert isinstance(helpers.azimuth_angle(0), u.Quantity)
    assert helpers.azimuth_angle("0 deg").value == pytest.approx(0.0)
    assert helpers.azimuth_angle("45 deg").value == pytest.approx(45.0)
    assert helpers.azimuth_angle("90 deg").value == pytest.approx(90.0)
    assert helpers.azimuth_angle("North").value == pytest.approx(0.0)
    assert helpers.azimuth_angle("South").value == pytest.approx(180.0)
    assert helpers.azimuth_angle("East").value == pytest.approx(90.0)
    assert helpers.azimuth_angle("West").value == pytest.approx(270.0)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided azimuth angle, -1.0, is outside of the allowed \[0, 360\] interval",
    ):
        helpers.azimuth_angle(-1)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided azimuth angle, 370.0, is outside of the allowed \[0, 360\] interval",
    ):
        helpers.azimuth_angle(370)
    caplog.clear()
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"^The azimuth angle given as string can only be one of"
    ):
        helpers.azimuth_angle("TEST")
    with caplog.at_level("ERROR"):
        with pytest.raises(TypeError):
            helpers.azimuth_angle([0, 10])
    assert "The azimuth angle provided is not a valid numerical or string value." in caplog.text


def test_quantity():
    quantity_parser = helpers.quantity("km")

    assert quantity_parser("10") == 10 * u.km
    assert_quantity_allclose(quantity_parser("1500 m"), 1.5 * u.km)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"Invalid quantity value: 'invalid'. Expected a value convertible to km.",
    ):
        quantity_parser("invalid")


def test_build_info_action(mocker):
    mock_get_build_options = mocker.patch(
        "simtools.dependencies.get_build_options",
        return_value={"version": "1.0.0", "python": "3.9"},
    )
    mock_print = mocker.patch("builtins.print")
    mock_exit = mocker.patch.object(argparse.ArgumentParser, "exit")

    action = helpers.BuildInfoAction(option_strings=["--build_info"], build_info="Test build info")
    test_parser = parser.CommandLineParser()
    action(test_parser, None, None, "--build_info")

    assert mock_get_build_options.called
    assert mock_print.called
    assert mock_exit.called
    assert mock_print.call_args_list[0][0][0] == "Test build info"


def test_bounded_int():
    bounded_int_checker = helpers.bounded_int(1, 10)

    assert bounded_int_checker(1) == 1
    assert bounded_int_checker(5) == 5
    assert bounded_int_checker(10) == 10
    assert bounded_int_checker("1") == 1
    assert bounded_int_checker("5") == 5
    assert bounded_int_checker("10") == 10

    with pytest.raises(ValueError, match=r"0 not in \[1,10\]"):
        bounded_int_checker(0)
    with pytest.raises(ValueError, match=r"11 not in \[1,10\]"):
        bounded_int_checker(11)
    with pytest.raises(ValueError, match=r"-5 not in \[1,10\]"):
        bounded_int_checker(-5)

    bounded_int_checker_large = helpers.bounded_int(100, 1000)
    assert bounded_int_checker_large(100) == 100
    assert bounded_int_checker_large(500) == 500
    assert bounded_int_checker_large(1000) == 1000

    with pytest.raises(ValueError, match=r"99 not in \[100,1000\]"):
        bounded_int_checker_large(99)
    with pytest.raises(ValueError, match=r"1001 not in \[100,1000\]"):
        bounded_int_checker_large(1001)


def test_string_or_dict():
    assert helpers.string_or_dict("plain_string") == "plain_string"
    assert helpers.string_or_dict("{invalid}") == "{invalid}"

    dict_str = "{'key1': 'value1', 'key2': 'value2'}"
    result = helpers.string_or_dict(dict_str)
    assert isinstance(result, dict)
    assert result == {"key1": "value1", "key2": "value2"}

    dict_str_nums = "{'1.0': '/path/v1.sif', '2.0': '/path/v2.sif'}"
    result = helpers.string_or_dict(dict_str_nums)
    assert isinstance(result, dict)
    assert result == {"1.0": "/path/v1.sif", "2.0": "/path/v2.sif"}

    test_dict = {"already": "dict"}
    assert helpers.string_or_dict(test_dict) == test_dict

    dict_str_spaces = "  {'a': 'b'}  "
    result = helpers.string_or_dict(dict_str_spaces)
    assert isinstance(result, dict)
    assert result == {"a": "b"}


def test_one_or_many_action():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={"software": None, "corsika_configuration": ["primary"]}
    )

    args = test_parser.parse_args(["--primary", "gamma"])
    assert args.primary == "gamma"
    assert isinstance(args.primary, str)

    args = test_parser.parse_args(["--primary", "gamma", "proton"])
    assert args.primary == ["gamma", "proton"]
    assert isinstance(args.primary, list)


def test_quantity_pair_action():
    test_parser = parser.CommandLineParser()
    test_parser.initialize_default_arguments(
        simulation_configuration={"software": None, "corsika_configuration": ["energy_range"]}
    )

    args = test_parser.parse_args(["--energy_range", "30 GeV 300 GeV"])
    assert len(args.energy_range) == 2
    assert_quantity_allclose(args.energy_range[0], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[1], 300 * u.GeV)

    args = test_parser.parse_args(["--energy_range", "50 GeV", "5 TeV"])
    assert len(args.energy_range) == 2
    assert_quantity_allclose(args.energy_range[0], 50 * u.GeV)
    assert_quantity_allclose(args.energy_range[1], 5 * u.TeV)

    args = test_parser.parse_args(["--energy_range", "30", "GeV", "300", "GeV"])
    assert len(args.energy_range) == 2
    assert_quantity_allclose(args.energy_range[0], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[1], 300 * u.GeV)

    args = test_parser.parse_args(["--energy_range", "30 GeV 30 GeV", "300 GeV 300 GeV"])
    assert len(args.energy_range) == 2
    assert_quantity_allclose(args.energy_range[0][0], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[0][1], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[1][0], 300 * u.GeV)
    assert_quantity_allclose(args.energy_range[1][1], 300 * u.GeV)
