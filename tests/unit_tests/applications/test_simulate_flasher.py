"""Tests for flasher application argument validation."""

import argparse

import pytest

from simtools.applications import simulate_flasher
from simtools.applications.simulate_flasher import _validate_max_photoelectrons


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
