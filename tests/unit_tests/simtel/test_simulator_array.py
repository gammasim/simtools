#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.runners.simtel_runner import InvalidOutputFileError
from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(corsika_config_mock_array_model, simtel_path):
    return SimulatorArray(
        corsika_config=corsika_config_mock_array_model,
        simtel_path=simtel_path,
        label="test-simtel-runner",
        use_multipipe=False,
    )


def test_simtel_runner(simtel_runner):
    sr = simtel_runner
    assert "simtel" in str(sr._directory["output"])
    assert "simtel" in str(sr._directory["data"])
    assert isinstance(sr._directory["data"], Path)


def test_make_run_command(simtel_runner):
    run_command = simtel_runner._make_run_command(
        run_number=3, input_file="test_make_run_command.inp"
    )
    assert "sim_telarray" in run_command
    assert "-run" in run_command
    assert "3" in run_command
    assert "test-simtel-runner.zst" in run_command
    assert "test_make_run_command.inp" in run_command


def test_check_run_result(simtel_runner):
    expected_pattern = r"sim_telarray output file .+ does not exist\."
    with pytest.raises(InvalidOutputFileError, match=expected_pattern):
        assert simtel_runner._check_run_result(run_number=3)


def test_get_power_law_for_sim_telarray_histograms():
    from simtools.corsika.primary_particle import PrimaryParticle

    gamma = PrimaryParticle(particle_id="gamma", particle_id_type="common_name")
    electron = PrimaryParticle(particle_id="electron", particle_id_type="common_name")
    proton = PrimaryParticle(particle_id="proton", particle_id_type="common_name")
    helium = PrimaryParticle(particle_id="helium", particle_id_type="common_name")
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(gamma) == pytest.approx(2.5)
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(electron) == pytest.approx(3.3)
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(proton) == pytest.approx(2.68)
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(helium) == pytest.approx(2.68)
