"""Tests for the sim_telarray simulation stage."""

import pytest

from simtools.runners.sim_telarray_stage import SimTelArrayStage
from simtools.runners.stages import STDIN, StageInputs, StreamTo


@pytest.fixture
def sim_telarray_stage(corsika_config_mock_array_model):
    return SimTelArrayStage(
        simulation_parameters=corsika_config_mock_array_model.simulation_parameters,
        array_model=corsika_config_mock_array_model.array_model,
        label="test-sim-telarray-stage",
    )


def test_instances_one_per_array_model(corsika_config_mock_array_model):
    inputs = StageInputs(
        array_models=(corsika_config_mock_array_model.array_model,) * 2,
        simulation_parameters=(corsika_config_mock_array_model.simulation_parameters,) * 2,
        shower_configurations=(),
        args={},
    )

    assert len(SimTelArrayStage.instances(inputs, label="test")) == 2


def test_prepare_with_input_file(sim_telarray_stage, mocker):
    mocker.patch.object(
        sim_telarray_stage.simulator_array,
        "make_run_command",
        return_value=["sim_telarray", "command"],
    )
    sim_telarray_stage.simulator_array.file_list = {}

    lines = sim_telarray_stage.prepare(run_number=1, input_file="input.corsika")

    assert len(lines) == 1
    assert "sim_telarray command" in lines[0]
    assert "input.corsika" in lines[0]
    assert "SIM_TELARRAY_CONFIG_PATH" in lines[0]


def test_prepare_keeps_resource_record_for_each_run(sim_telarray_stage, mocker):
    mocker.patch.object(
        sim_telarray_stage.simulator_array,
        "make_run_command",
        return_value=["sim_telarray", "command"],
    )
    sim_telarray_stage.simulator_array.file_list = {}
    sim_telarray_stage.simulator_array.runs_per_set = 2

    lines = sim_telarray_stage.prepare(run_number=1, input_file="input.corsika")

    resource_files = sim_telarray_stage.file_list["sim_telarray_resources"]
    assert len(lines) == 2
    assert resource_files[0] != resource_files[1]
    assert str(resource_files[0]) in lines[0]
    assert str(resource_files[1]) in lines[1]


def test_prepare_reading_stream(sim_telarray_stage):
    lines = sim_telarray_stage.prepare(run_number=1, input_file=STDIN)

    assert len(lines) == 1
    assert "bin/sim_telarray" in lines[0]
    assert "-C show=all" in lines[0]
    assert "--input-file" not in lines[0]


def test_prepare_rejects_streamed_output(sim_telarray_stage):
    with pytest.raises(ValueError, match="sim_telarray output can only be written"):
        sim_telarray_stage.prepare(run_number=1, input_file=STDIN, output=StreamTo(("x",)))


def test_output_file_environment_resources(sim_telarray_stage):
    assert "sim_telarray" in str(sim_telarray_stage.output_file(1))
    assert sim_telarray_stage.environment() == {"SIM_TELARRAY_CONFIG_PATH": ""}
    assert sim_telarray_stage.get_resources(runtime=1.5)["runtime"] == pytest.approx(1.5)
