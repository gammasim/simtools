#!/usr/bin/python3

import copy
import logging
from pathlib import Path

import pytest

from simtools.runners.corsika_simtel_runner import (
    CorsikaRunner,
    CorsikaSimtelRunner,
    SimulatorArray,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def corsika_simtel_runner(io_handler, corsika_config, simtel_path):
    return CorsikaSimtelRunner(
        corsika_config=corsika_config,
        simtel_path=simtel_path,
        label="test-corsika-simtel-runner",
        use_multipipe=True,
    )


def test_corsika_simtel_runner(corsika_simtel_runner):

    assert isinstance(corsika_simtel_runner.corsika_runner, CorsikaRunner)
    assert isinstance(corsika_simtel_runner.simulator_array, SimulatorArray)


def test_prepare_run_script(corsika_simtel_runner):
    # No run number is given

    script = corsika_simtel_runner.prepare_run_script()

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content

    # Run number is given
    run_number = 3
    script = corsika_simtel_runner.prepare_run_script(run_number=run_number)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content
        assert "-R 3" in script_content


def test_prepare_run_script_with_invalid_run(corsika_simtel_runner):
    for run_number in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_simtel_runner.prepare_run_script(run_number=run_number)


def test_export_multipipe_script(corsika_simtel_runner):
    corsika_simtel_runner._export_multipipe_script(run_number=1)
    script = Path(corsika_simtel_runner.corsika_config.config_file_path.parent).joinpath(
        corsika_simtel_runner.corsika_config.get_corsika_config_file_name("multipipe")
    )

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "bin/sim_telarray" in script_content
        assert "-C telescope_theta=20" in script_content
        assert "-C telescope_phi=0" in script_content
        assert "-C show=all" in script_content


def test_write_multipipe_script(corsika_simtel_runner):
    corsika_simtel_runner._export_multipipe_script(run_number=1)
    multipipe_file = Path(corsika_simtel_runner.corsika_config.config_file_path.parent).joinpath(
        corsika_simtel_runner.corsika_config.get_corsika_config_file_name("multipipe")
    )
    corsika_simtel_runner._write_multipipe_script(multipipe_file)
    script = Path(corsika_simtel_runner.corsika_config.config_file_path.parent).joinpath(
        "run_cta_multipipe"
    )

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "bin/multipipe_corsika" in script_content
        assert f"-c {multipipe_file}" in script_content
        assert "'Fan-out failed'" in script_content


def test_make_run_command(corsika_simtel_runner):
    command = corsika_simtel_runner._make_run_command(input_file="-", run_number=1)
    assert "bin/sim_telarray" in command
    assert "-C telescope_theta=20" in command
    assert "-C telescope_phi=0" in command
    assert "-C show=all" in command
    assert "run000001_proton_za20deg_azm000deg_South_test_layout_test" in command

    _test_corsika_simtel_runner = copy.deepcopy(corsika_simtel_runner)
    _test_corsika_simtel_runner.label = None
    command = _test_corsika_simtel_runner._make_run_command(input_file="-", run_number=1)
    assert "-W" not in command


def test_make_run_command_divergent(corsika_simtel_runner):
    corsika_simtel_runner.label = "test-corsika-simtel-runner-divergent-pointing"
    command = corsika_simtel_runner._make_run_command(input_file="-", run_number=1)
    assert "bin/sim_telarray" in command
    assert "-W telescope_theta=20" in command  # -W is for pointing
    assert "-W telescope_phi=0" in command
    assert "-C show=all" in command
    assert "run000001_proton_za20deg_azm000deg_South_test_layout_test" in command


def test_get_file_name(corsika_simtel_runner):

    assert (
        corsika_simtel_runner.get_file_name(
            simulation_software="corsika", file_type="log", run_number=1
        ).name
        == "run000001_proton_za20deg_azm000deg_South_test_layout_test-corsika-simtel-runner.log.gz"
    )

    assert (
        corsika_simtel_runner.get_file_name(
            simulation_software="simtel", file_type="simtel_output", run_number=1
        ).name
        == "run000001_proton_za20deg_azm000deg_South_test_layout_test-corsika-simtel-runner.simtel.zst"
    )

    # preference given to simtel runner
    assert (
        corsika_simtel_runner.get_file_name(
            simulation_software=None, file_type="simtel_output", run_number=1
        ).name
        == "run000001_proton_za20deg_azm000deg_South_test_layout_test-corsika-simtel-runner.simtel.zst"
    )

    # no simulator_array
    _test_corsika_simtel_runner = copy.deepcopy(corsika_simtel_runner)
    _test_corsika_simtel_runner.simulator_array = None

    assert (
        _test_corsika_simtel_runner.get_file_name(
            simulation_software=None, file_type="simtel_output", run_number=1
        ).name
        == "run000001_proton_za20deg_azm000deg_South_test_layout_test-corsika-simtel-runner.simtel.zst"
    )
