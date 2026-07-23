"""Tests for shared command-line argument definitions."""

import astropy.units as u

import simtools.configuration.argument_helpers as helpers
from simtools.configuration.arguments import (
    AZIMUTH_ANGLE,
    CORSIKA_CONFIGURATION_ARGUMENTS,
    DATABASE_ARGUMENTS,
    EXECUTION_ARGUMENTS,
    PATH_ARGUMENTS,
    PRIMARY,
    PRIMARY_ID_TYPE,
    RUN_TIME_ARGUMENTS,
    SHOWER_ARGUMENTS,
    STANDARD_ARGUMENTS,
    USER_ARGUMENTS,
    ZENITH_ANGLE,
    ArgumentDefinition,
)


def test_shared_arguments_have_one_direct_definition():
    bundles = (
        DATABASE_ARGUMENTS,
        EXECUTION_ARGUMENTS,
        PATH_ARGUMENTS,
        RUN_TIME_ARGUMENTS,
        SHOWER_ARGUMENTS,
        USER_ARGUMENTS,
    )
    assert all(
        isinstance(argument, ArgumentDefinition) for bundle in bundles for argument in bundle
    )
    assert len({argument.name for argument in STANDARD_ARGUMENTS}) == len(STANDARD_ARGUMENTS)


def test_corsika_arguments_retain_argparse_configuration():
    assert CORSIKA_CONFIGURATION_ARGUMENTS[0] is PRIMARY
    assert PRIMARY.kwargs["type"] is str.lower
    assert PRIMARY.kwargs["action"] is helpers.OneOrManyAction
    assert PRIMARY.kwargs["nargs"] == "+"
    assert PRIMARY.kwargs["required"] is True
    assert "proton" in PRIMARY.kwargs["help"]

    assert PRIMARY_ID_TYPE.kwargs["choices"] == ["common_name", "corsika7_id", "pdg_id"]
    assert AZIMUTH_ANGLE.kwargs["default"] == 0 * u.deg
    assert ZENITH_ANGLE.kwargs["default"] == 20 * u.deg


def test_argument_override_returns_an_independent_definition():
    optional_primary = PRIMARY(required=False)

    assert optional_primary is not PRIMARY
    assert optional_primary.kwargs["required"] is False
    assert PRIMARY.kwargs["required"] is True
