"""Tests for flasher application argument validation."""

import argparse
import importlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simtools.applications import simulate_flasher
from simtools.applications.simulate_flasher import (
    _validate,
    _validate_max_photoelectrons,
    _validate_simulation_software,
    main,
)
from simtools.runtime import software_catalog

_CUSTOM_CHAIN = [
    "obdeect_flasher",
    "stream_to_simtel",
    "sim_telarray:bypass_optics=camera_entrance",
]


@pytest.mark.parametrize("value", [None, 1000, 499999999])
def test_max_photoelectrons_accepts_limits(value):
    _validate_max_photoelectrons({"max_photoelectrons": value}, {}, argparse.ArgumentParser())


@pytest.mark.parametrize("value", [999, 500000000])
def test_max_photoelectrons_rejects_outside_limits(value, capsys):
    with pytest.raises(SystemExit, match="2"):
        _validate_max_photoelectrons({"max_photoelectrons": value}, {}, argparse.ArgumentParser())
    assert "must be at least 1000 and less than 500000000" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option", ["light_source", "light_source_type", "flasher_photons", "max_photoelectrons"]
)
def test_dc_rejects_pulsed_options(option, capsys):
    args = {"run_mode": "dc", "number_of_events": [10], option: 1000}
    with pytest.raises(SystemExit, match="2"):
        _validate_max_photoelectrons(args, {}, argparse.ArgumentParser())
    assert f"--{option}" in capsys.readouterr().err


def test_dc_dispatches_to_pedestal_simulator(mocker, run_context):
    context = run_context.replace(args={"run_mode": "dc", "number_of_events": [10]})
    mocker.patch.object(type(simulate_flasher.APPLICATION), "start", return_value=context)
    simulator = mocker.patch.object(simulate_flasher, "Simulator")
    simulate_flasher.main()
    args = simulator.call_args.kwargs["context"].args
    assert args["run_mode"] == "pedestals_flasher_dc"
    assert args["number_of_events"] == 10
    simulator.return_value.simulate.assert_called_once()
    simulator.return_value.validate_simulations.assert_called_once()


def _argument(application, name):
    return next(argument for argument in application.arguments if argument.name == name)


@pytest.mark.parametrize("module", ["simulate_flasher", "simulate_illuminator"])
def test_simulation_software_default_is_light_emission_chain(module):
    application = importlib.import_module(f"simtools.applications.{module}").APPLICATION
    argument = _argument(application, "simulation_software")
    assert argument.kwargs["default"] == ["light_emission", "sim_telarray"]


@pytest.mark.parametrize("chain", [None, [], ["light_emission", "sim_telarray"]])
def test_default_chain_is_allowed_in_direct_injection(chain):
    _validate_simulation_software(
        {"run_mode": "direct_injection", "simulation_software": chain},
        {},
        argparse.ArgumentParser(),
    )


def test_other_chain_is_allowed_in_full_simulation():
    _validate_simulation_software(
        {"run_mode": "full_simulation", "simulation_software": ["obdeect_flasher", "sim_telarray"]},
        {},
        argparse.ArgumentParser(),
    )


def test_other_chain_is_rejected_in_direct_injection(capsys):
    args = {"run_mode": "direct_injection", "simulation_software": ["obdeect_flasher"]}
    with pytest.raises(SystemExit, match="2"):
        _validate_simulation_software(args, {}, argparse.ArgumentParser())
    assert "available in run mode full_simulation only" in capsys.readouterr().err


def _full_simulation_args(simulation_software):
    return {
        "run_mode": "full_simulation",
        "light_source": "LSFN-design",
        "telescopes": ["LSTN-01"],
        "array_layout_name": None,
        "number_of_events": 1,
        "label": None,
        "simulation_software": simulation_software,
    }


@patch("simtools.applications.simulate_flasher.SimulatorLightEmission")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_full_simulation_default_chain_uses_default_script(mock_start, mock_simulator_class):
    default = list(software_catalog.default_chain(software_catalog.LIGHT_SOURCE))
    mock_start.return_value = Mock(args=_full_simulation_args(default))

    main()

    simulator = mock_simulator_class.return_value
    simulator.simulate.assert_called_once_with(script_writer=None)
    simulator.validate_simulations.assert_called_once()


@patch("simtools.runners.light_emission_runner.write_chain_script")
@patch("simtools.applications.simulate_flasher.SimulatorLightEmission")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_full_simulation_custom_chain_writes_chain_script(
    mock_start, mock_simulator_class, mock_write_chain_script
):
    mock_start.return_value = Mock(args=_full_simulation_args(_CUSTOM_CHAIN))

    main()

    simulator = mock_simulator_class.return_value
    script_writer = simulator.simulate.call_args.kwargs["script_writer"]
    assert script_writer is not None
    script_file = Path("chain.sh")
    script_writer(simulator, script_file)
    mock_write_chain_script.assert_called_once_with(simulator, _CUSTOM_CHAIN, script_file)
    simulator.validate_simulations.assert_called_once()


def test_validate_checks_photoelectrons_and_chain(capsys):
    args = {
        "run_mode": "direct_injection",
        "light_source": "LSFN-design",
        "max_photoelectrons": None,
        "simulation_software": ["obdeect_flasher"],
    }
    with pytest.raises(SystemExit, match="2"):
        _validate(args, {}, argparse.ArgumentParser())
    assert "available in run mode full_simulation only" in capsys.readouterr().err


@patch("simtools.applications.simulate_flasher.Simulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_direct_injection_runs_injection_sequence(mock_start, mock_simulator):
    args = {
        "run_mode": "direct_injection",
        "light_source": "LSFN-design",
        "telescopes": ["LSTN-01"],
        "number_of_events": 1,
        "label": "injection",
    }
    context = Mock(args=args)
    mock_start.return_value = context

    main()

    mock_simulator.simulate_direct_injection_sequence.assert_called_once_with(
        label="injection", context=context
    )


@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_rejects_unsupported_run_mode(mock_start):
    args = {
        "run_mode": "unknown",
        "light_source": "LSFN-design",
        "telescopes": ["LSTN-01"],
        "number_of_events": 1,
    }
    mock_start.return_value = Mock(args=args)
    with pytest.raises(ValueError, match="Unsupported run_mode: unknown"):
        main()
