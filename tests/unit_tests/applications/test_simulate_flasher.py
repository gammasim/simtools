"""Tests for flasher application argument validation."""

import argparse
import importlib

import pytest

from simtools.applications.simulate_flasher import (
    _validate_max_photoelectrons,
    _validate_simulation_chain,
)


@pytest.mark.parametrize("value", [None, 1000, 499999999])
def test_max_photoelectrons_accepts_limits(value):
    _validate_max_photoelectrons({"max_photoelectrons": value}, {}, argparse.ArgumentParser())


@pytest.mark.parametrize("value", [999, 500000000])
def test_max_photoelectrons_rejects_outside_limits(value, capsys):
    with pytest.raises(SystemExit, match="2"):
        _validate_max_photoelectrons({"max_photoelectrons": value}, {}, argparse.ArgumentParser())
    assert "must be at least 1000 and less than 500000000" in capsys.readouterr().err


def _argument(application, name):
    return next(argument for argument in application.arguments if argument.name == name)


@pytest.mark.parametrize("module", ["simulate_flasher", "simulate_illuminator"])
def test_simulation_chain_default_is_light_emission_chain(module):
    application = importlib.import_module(f"simtools.applications.{module}").APPLICATION
    argument = _argument(application, "simulation_chain")
    assert argument.kwargs["default"] == ["light_emission", "sim_telarray"]


@pytest.mark.parametrize("chain", [None, [], ["light_emission", "sim_telarray"]])
def test_default_chain_is_allowed_in_direct_injection(chain):
    _validate_simulation_chain(
        {"run_mode": "direct_injection", "simulation_chain": chain}, {}, argparse.ArgumentParser()
    )


def test_other_chain_is_allowed_in_full_simulation():
    _validate_simulation_chain(
        {"run_mode": "full_simulation", "simulation_chain": ["obdeect_flasher", "sim_telarray"]},
        {},
        argparse.ArgumentParser(),
    )


def test_other_chain_is_rejected_in_direct_injection(capsys):
    args = {"run_mode": "direct_injection", "simulation_chain": ["obdeect_flasher"]}
    with pytest.raises(SystemExit, match="2"):
        _validate_simulation_chain(args, {}, argparse.ArgumentParser())
    assert "available in run mode full_simulation only" in capsys.readouterr().err
