"""Tests for flasher application argument validation."""

import argparse
import importlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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


def test_parallel_full_simulation_prepares_isolated_levels(tmp_test_directory):
    from simtools.applications.simulate_flasher import _full_simulation_runs
    from simtools.io.output_paths import OutputPaths
    from simtools.runtime.context import RunContext

    context = RunContext.from_arguments(
        {
            "output_path": str(tmp_test_directory),
            "model_path": str(tmp_test_directory / "input-models"),
            "run_number": 100,
            "max_workers": 2,
            "flasher_photons": [30000, 100000],
            "number_of_events": [2, 3],
            "simulation_seed": [101, 102],
        },
        paths=OutputPaths(tmp_test_directory, model_path=str(tmp_test_directory / "input-models")),
    )
    runs = _full_simulation_runs(context)
    assert [run.args["run_number"] for run in runs] == [100, 101]
    assert [run.args["flasher_photons"] for run in runs] == [30000, 100000]
    assert [run.args["number_of_events"] for run in runs] == [2, 3]
    assert all(run.args["simulation_seed"][0] == 101 for run in runs)
    assert runs[0].args["simulation_seed"][1] != runs[1].args["simulation_seed"][1]
    assert [run.args["simulation_seed"] for run in runs] == [
        run.args["simulation_seed"] for run in _full_simulation_runs(context)
    ]
    assert [run.paths.get_output_directory() for run in runs] == [
        tmp_test_directory / "run000100",
        tmp_test_directory / "run000101",
    ]
    assert [run.paths.get_model_configuration_directory("8.0.0") for run in runs] == [
        tmp_test_directory / "run000100/model/8.0.0",
        tmp_test_directory / "run000101/model/8.0.0",
    ]
    assert all(run.paths.model_path == str(tmp_test_directory / "input-models") for run in runs)


def test_full_simulation_runs_levels_concurrently(monkeypatch, tmp_test_directory):
    from threading import Barrier

    from simtools.applications import simulate_flasher
    from simtools.runtime.context import RunContext

    barrier = Barrier(2, timeout=5)
    completed = []

    class FakeSimulation:
        def __init__(self, light_emission_config, **_kwargs):
            self.run = light_emission_config["run_number"]

        def simulate(self, script_writer=None):
            barrier.wait()

        def validate_simulations(self):
            completed.append(self.run)

    args = _full_simulation_args(None) | {
        "max_workers": 2,
        "run_number": 100,
        "output_path": str(tmp_test_directory),
        "flasher_photons": [30000, 100000],
        "number_of_events": [1],
    }
    context = RunContext.from_arguments(args).replace(logger=Mock())
    monkeypatch.setattr(simulate_flasher.APPLICATION.__class__, "start", lambda _self: context)
    monkeypatch.setattr(simulate_flasher, "SimulatorLightEmission", FakeSimulation)
    main()
    assert sorted(completed) == [100, 101]
