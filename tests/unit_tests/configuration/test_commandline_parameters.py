#!/usr/bin/python3

import astropy.units as u

import simtools.configuration.commandline_argument_helpers as helpers
from simtools.configuration.commandline_parameters import (
    PARAMETER_DEFINITIONS,
    get_corsika_configuration_args,
)


def test_parameter_definitions_contains_shared_groups():
    assert "APPLICATION_ARGS" in PARAMETER_DEFINITIONS
    assert "CONFIGURATION_ARGS" in PARAMETER_DEFINITIONS
    assert "DB_CONFIG_ARGS" in PARAMETER_DEFINITIONS
    assert "EXECUTION_ARGS" in PARAMETER_DEFINITIONS
    assert "OUTPUT_ARGS" in PARAMETER_DEFINITIONS
    assert "PATH_ARGS" in PARAMETER_DEFINITIONS
    assert "RUN_TIME_ARGS" in PARAMETER_DEFINITIONS
    assert "SHOWER_ARGS" in PARAMETER_DEFINITIONS
    assert "SIMTEL_ARGS" in PARAMETER_DEFINITIONS
    assert "SIMULATION_MODEL_ARGS" in PARAMETER_DEFINITIONS
    assert "SIMULATION_SOFTWARE_ARGS" in PARAMETER_DEFINITIONS
    assert "USER_ARGS" in PARAMETER_DEFINITIONS


def test_get_dictionary_with_corsika_configuration(mocker):
    mock_particle_names = {"proton": 1, "helium": 2, "iron": 3}
    mocker.patch(
        "simtools.corsika.primary_particle.PrimaryParticle.particle_names",
        return_value=mock_particle_names,
    )

    corsika_config = get_corsika_configuration_args()

    assert "primary" in corsika_config
    assert corsika_config["primary"]["help"].startswith("Primary particle to simulate.")
    assert "proton" in corsika_config["primary"]["help"]
    assert "helium" in corsika_config["primary"]["help"]
    assert "iron" in corsika_config["primary"]["help"]
    assert corsika_config["primary"]["type"] is str.lower
    assert corsika_config["primary"]["action"] is helpers.OneOrManyAction
    assert corsika_config["primary"]["nargs"] == "+"
    assert corsika_config["primary"]["required"] is True

    assert "primary_id_type" in corsika_config
    assert corsika_config["primary_id_type"]["help"] == "Primary particle ID type"
    assert corsika_config["primary_id_type"]["type"] is str
    assert corsika_config["primary_id_type"]["choices"] == ["common_name", "corsika7_id", "pdg_id"]
    assert corsika_config["primary_id_type"]["default"] == "common_name"

    assert "azimuth_angle" in corsika_config
    assert corsika_config["azimuth_angle"]["help"].startswith(
        "Telescope pointing direction in azimuth."
    )
    assert corsika_config["azimuth_angle"]["type"] == helpers.azimuth_angle
    assert corsika_config["azimuth_angle"]["action"] is helpers.OneOrManyAction
    assert corsika_config["azimuth_angle"]["nargs"] == "+"
    assert corsika_config["azimuth_angle"]["default"] == 0 * u.deg

    assert "zenith_angle" in corsika_config
    assert corsika_config["zenith_angle"]["help"] == "Zenith angle in degrees (between 0 and 180)."
    assert corsika_config["zenith_angle"]["type"] == helpers.zenith_angle
    assert corsika_config["zenith_angle"]["action"] is helpers.OneOrManyAction
    assert corsika_config["zenith_angle"]["nargs"] == "+"
    assert corsika_config["zenith_angle"]["default"] == 20 * u.deg

    assert "showers_per_run" in corsika_config
    assert (
        corsika_config["showers_per_run"]["help"] == "Baseline number of CORSIKA showers per run."
    )
    assert corsika_config["showers_per_run"]["type"] is int

    assert "run_number_offset" in corsika_config
    assert "Offset added to run number" in corsika_config["run_number_offset"]["help"]
    assert corsika_config["run_number_offset"]["type"] is int
    assert corsika_config["run_number_offset"]["default"] == 0

    assert "run_number" in corsika_config
    assert corsika_config["run_number"]["help"] == "Run number to be simulated."
    assert corsika_config["run_number"]["type"] is int
    assert corsika_config["run_number"]["default"] == 1

    assert "event_number_first_shower" in corsika_config
    assert corsika_config["event_number_first_shower"]["help"] == "Event number of first shower"
    assert corsika_config["event_number_first_shower"]["type"] is int
    assert corsika_config["event_number_first_shower"]["default"] == 1

    assert "correct_for_b_field_alignment" in corsika_config
    assert (
        corsika_config["correct_for_b_field_alignment"]["help"] == "Correct for B-field alignment"
    )
    assert corsika_config["correct_for_b_field_alignment"]["action"] == "store_true"
    assert corsika_config["correct_for_b_field_alignment"]["default"] is True
