#!/usr/bin/python3

import argparse
import logging

import astropy.units as u
import pytest

import simtools.configuration.commandline_parser as parser

logger = logging.getLogger()
SIMULATION_MODEL_STRING = "simulation model"


def test_site():
    assert parser.CommandLineParser.site("North") == "North"
    assert parser.CommandLineParser.site("South") == "South"

    with pytest.raises(ValueError, match=r"Invalid name East"):
        parser.CommandLineParser.site("East")


def test_telescope():
    assert parser.CommandLineParser.telescope("LSTN-01") == "LSTN-01"
    assert parser.CommandLineParser.telescope("MSTx-NectarCam") == "MSTx-NectarCam"

    with pytest.raises(ValueError, match=r"Invalid name Whipple"):
        parser.CommandLineParser.telescope("Whipple")

    with pytest.raises(ValueError, match=r"Invalid name LST"):
        parser.CommandLineParser.telescope("LST")


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

    with caplog.at_level("WARNING"):
        with pytest.raises(TypeError):
            parser.CommandLineParser.zenith_angle("North")
    assert "The zenith angle provided is not a valid numeric" in caplog.text


def test_parse_quantity_pair():
    for test_string in ["100 GeV 5 TeV", "100GeV 5TeV", "100GeV 5 TeV"]:
        e_pair = parser.CommandLineParser.parse_quantity_pair(test_string)
        assert e_pair[0].value == pytest.approx(100.0)
        assert e_pair[0].unit == u.GeV
        assert e_pair[1].value == pytest.approx(5.0)
        assert e_pair[1].unit == u.TeV

    with pytest.raises(ValueError, match=r"Input string does not contain exactly two quantities."):
        parser.CommandLineParser.parse_quantity_pair("100 GeV 5 TeV 20 PeV")

    with pytest.raises(ValueError, match=r"^'abc' did not parse as unit:"):
        parser.CommandLineParser.parse_quantity_pair("100 GeV 5 abc")

    with pytest.raises(ValueError, match=r"Input string does not contain exactly two quantities."):
        parser.CommandLineParser.parse_quantity_pair("a GeV 5 TeV")


def test_parse_integer_and_quantity():
    for test_string in ["5 1500 m", "5 1500m", "5 1500.0 m", "(5, <Quantity 1500 m>)"]:
        c_pair = parser.CommandLineParser.parse_integer_and_quantity(test_string)
        assert c_pair[0] == 5
        assert c_pair[1].value == pytest.approx(1500.0)
        assert c_pair[1].unit == u.m

    with pytest.raises(ValueError, match=r"^'abc' did not parse as unit:"):
        parser.CommandLineParser.parse_integer_and_quantity("5 5 abc")
    with pytest.raises(
        ValueError, match=r"Input string does not contain an integer and a astropy quantity."
    ):
        parser.CommandLineParser.parse_integer_and_quantity("0 m 5 m")


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
    caplog.clear()
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"^The azimuth angle given as string can only be one of"
    ):
        parser.CommandLineParser.azimuth_angle("TEST")
    with caplog.at_level("ERROR"):
        with pytest.raises(TypeError):
            parser.CommandLineParser.azimuth_angle([0, 10])
    assert "The azimuth angle provided is not a valid numerical or string value." in caplog.text


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
            "user",
        ]

    _parser_2 = parser.CommandLineParser()
    _parser_2.initialize_default_arguments(output=True)
    job_groups = _parser_2._action_groups
    assert "output" in [str(group.title) for group in job_groups]


def test_simulation_model():
    # simulation model is none
    _parser_n = parser.CommandLineParser()
    _parser_n.initialize_default_arguments(simulation_model=None)

    # Model version only, no site or telescope
    _parser_v = parser.CommandLineParser()
    _parser_v.initialize_default_arguments(simulation_model=["model_version"])
    job_groups = _parser_v._action_groups

    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert all(action.dest != "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # Site model can exist without a telescope model, model and parameter version
    _parser_s = parser.CommandLineParser()
    _parser_s.initialize_default_arguments(
        simulation_model=["site", "model_version", "parameter_version"]
    )
    job_groups = _parser_s._action_groups
    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert any(action.dest == "parameter_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # No telescope model without site model; parameter_version only
    _parser_t = parser.CommandLineParser()
    _parser_t.initialize_default_arguments(
        simulation_model=["telescope", "site", "parameter_version"]
    )
    job_groups = _parser_t._action_groups
    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "parameter_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert any(action.dest == "telescope" for action in group._group_actions)


def test_db_configuration():
    _parser_6 = parser.CommandLineParser()
    _parser_6.initialize_default_arguments(db_config=True)
    job_groups = _parser_6._action_groups
    assert "database configuration" in [str(group.title) for group in job_groups]


def test_layout_parsers():
    _parser_7 = parser.CommandLineParser()
    _parser_7.initialize_default_arguments(simulation_model=["layout"])
    job_groups = _parser_7._action_groups
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)

    _parser_8 = parser.CommandLineParser()
    _parser_8.initialize_default_arguments(simulation_model=["layout", "layout_file"])
    job_groups = _parser_8._action_groups
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)
            assert any(action.dest == "array_layout_file" for action in group._group_actions)

    _parser_9 = parser.CommandLineParser()
    _parser_9.initialize_default_arguments(
        simulation_model=["layout", "layout_file", "plot_all_layouts", "layout_parameter_file"]
    )
    job_groups = _parser_9._action_groups
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)
            assert any(action.dest == "array_layout_file" for action in group._group_actions)
            assert any(action.dest == "plot_all_layouts" for action in group._group_actions)
            assert any(
                action.dest == "array_layout_parameter_file" for action in group._group_actions
            )


def test_simulation_configuration():
    _parser_9 = parser.CommandLineParser()
    _parser_9.initialize_default_arguments(
        simulation_configuration={
            "software": None,
            "corsika_configuration": ["all"],
            "sim_telarray_configuration": ["all"],
        }
    )
    job_groups = _parser_9._action_groups
    for group in job_groups:
        if str(group.title) == "simulation software":
            assert any(action.dest == "simulation_software" for action in group._group_actions)
        if str(group.title) == "simulation configuration":
            assert any(action.dest == "primary" for action in group._group_actions)
        if str(group.title) == "shower parameters":
            assert any(action.dest == "view_cone" for action in group._group_actions)
        if str(group.title) == "sim_telarray configuration":
            assert any(
                action.dest == "sim_telarray_instrument_seeds" for action in group._group_actions
            )

    _parser_10 = parser.CommandLineParser()
    _parser_10.initialize_default_arguments(
        simulation_configuration={"software": None, "corsika_configuration": ["wrong_parameter"]}
    )


def test_initialize_db_config_arguments_strip_string():
    parser_10 = parser.CommandLineParser()
    parser_10.initialize_db_config_arguments()
    for test_string in ["test", " test", "test ", " test "]:
        args = parser_10.parse_args(["--db_simulation_model", test_string])
        assert args.db_simulation_model == "test"


def test_get_dictionary_with_corsika_configuration(mocker):
    # Mock PrimaryParticle.particle_names to return a predefined dictionary
    mock_particle_names = {"proton": 1, "helium": 2, "iron": 3}
    mocker.patch(
        "simtools.corsika.primary_particle.PrimaryParticle.particle_names",
        return_value=mock_particle_names,
    )

    # Call the method to get the dictionary
    corsika_config = parser.CommandLineParser._get_dictionary_with_corsika_configuration()

    # Test the "primary" key
    assert "primary" in corsika_config
    assert corsika_config["primary"]["help"].startswith("Primary particle to simulate.")
    assert "proton" in corsika_config["primary"]["help"]
    assert "helium" in corsika_config["primary"]["help"]
    assert "iron" in corsika_config["primary"]["help"]
    assert corsika_config["primary"]["type"] is str.lower
    assert corsika_config["primary"]["required"] is True

    # Test the "primary_id_type" key
    assert "primary_id_type" in corsika_config
    assert corsika_config["primary_id_type"]["help"] == "Primary particle ID type"
    assert corsika_config["primary_id_type"]["type"] is str
    assert corsika_config["primary_id_type"]["required"] is False
    assert corsika_config["primary_id_type"]["choices"] == ["common_name", "corsika7_id", "pdg_id"]
    assert corsika_config["primary_id_type"]["default"] == "common_name"

    # Test the "azimuth_angle" key
    assert "azimuth_angle" in corsika_config
    assert corsika_config["azimuth_angle"]["help"].startswith(
        "Telescope pointing direction in azimuth."
    )
    assert corsika_config["azimuth_angle"]["type"] == parser.CommandLineParser.azimuth_angle
    assert corsika_config["azimuth_angle"]["required"] is True

    # Test the "zenith_angle" key
    assert "zenith_angle" in corsika_config
    assert corsika_config["zenith_angle"]["help"] == "Zenith angle in degrees (between 0 and 180)."
    assert corsika_config["zenith_angle"]["type"] == parser.CommandLineParser.zenith_angle
    assert corsika_config["zenith_angle"]["required"] is True

    # Test the "nshow" key
    assert "nshow" in corsika_config
    assert corsika_config["nshow"]["help"] == "Number of showers per run to simulate."
    assert corsika_config["nshow"]["type"] is int
    assert corsika_config["nshow"]["required"] is False

    # Test the "run_number_offset" key
    assert "run_number_offset" in corsika_config
    assert (
        corsika_config["run_number_offset"]["help"]
        == "An offset for the run number to be simulated."
    )
    assert corsika_config["run_number_offset"]["type"] is int
    assert corsika_config["run_number_offset"]["required"] is False
    assert corsika_config["run_number_offset"]["default"] == 0

    # Test the "run_number" key
    assert "run_number" in corsika_config
    assert corsika_config["run_number"]["help"] == "Run number to be simulated."
    assert corsika_config["run_number"]["type"] is int
    assert corsika_config["run_number"]["required"] is True
    assert corsika_config["run_number"]["default"] == 1

    # Test the "number_of_runs" key
    assert "number_of_runs" in corsika_config
    assert corsika_config["number_of_runs"]["help"] == "Number of runs to be simulated."
    assert corsika_config["number_of_runs"]["type"] is int
    assert corsika_config["number_of_runs"]["required"] is True
    assert corsika_config["number_of_runs"]["default"] == 1

    # Test the "event_number_first_shower" key
    assert "event_number_first_shower" in corsika_config
    assert corsika_config["event_number_first_shower"]["help"] == "Event number of first shower"
    assert corsika_config["event_number_first_shower"]["type"] is int
    assert corsika_config["event_number_first_shower"]["required"] is False
    assert corsika_config["event_number_first_shower"]["default"] == 1

    # Test the "correct_for_b_field_alignment" key
    assert "correct_for_b_field_alignment" in corsika_config
    assert (
        corsika_config["correct_for_b_field_alignment"]["help"] == "Correct for B-field alignment"
    )
    assert corsika_config["correct_for_b_field_alignment"]["action"] == "store_true"
    assert corsika_config["correct_for_b_field_alignment"]["required"] is False
    assert corsika_config["correct_for_b_field_alignment"]["default"] is True
