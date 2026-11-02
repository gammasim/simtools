"""Tests for shower photon routing commands."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import astropy.units as u
import pytest

from simtools.job_execution.runner_services import RunnerServices
from simtools.runners.shower_routing import ShowerRouting, _shower_route_seeds


class _Telescope:
    def __init__(self, name, mirror_class):
        self.name = name
        self.mirror_class = mirror_class

    def get_parameter_value(self, name):
        assert name == "mirror_class"
        return self.mirror_class

    def get_parameter_value_with_unit(self, name):
        assert name == "telescope_sphere_radius"
        return 12.5 * u.m


class _Routing(ShowerRouting):
    def _minimum_input_plane_height_m(self, telescope, theta_deg):
        return 29.9


class _SimulatorArray:
    def __init__(self, config, context):
        self.array_model = SimpleNamespace(
            telescope_models={
                "LSTN-01": _Telescope("LST", 1),
                "LSTN-02": _Telescope("LST", 1),
                "MSTN-01": _Telescope("MST", 2),
            },
            site="South",
            layout_name="test",
            model_version="7.0.0",
            sim_telarray_seed=SimpleNamespace(seed_string="11,22"),
        )
        self.simulation_parameters = config.simulation_parameters
        self.runner_service = RunnerServices(
            config, "sim_telarray", array_model=self.array_model, context=context
        )

    def make_run_command(self, run_number, input_file):
        self.file_list = self.runner_service.load_files(run_number=run_number)
        return ["sim_telarray", "-C", "output_file=output.simtel", "-C", "show=all", input_file]


@pytest.fixture
def route_inputs(corsika_config_mock_array_model):
    config = corsika_config_mock_array_model
    context = config.context.replace(
        software=replace(config.context.software, ray_tracing_backend="obdeect")
    )
    return _Routing(config, context), _SimulatorArray(config, context)


@pytest.mark.parametrize("input_file", ["-", "events.corsika.zst"])
def test_routing_commands_by_telescope_type(route_inputs, input_file):
    routing, simulator = route_inputs
    lines = routing.commands(simulator, 1, input_file)
    route_lines = [line for line in lines if "simtools.simtel.shower_route" in line]
    assert len(route_lines) == 2
    assert all("--seed 11 --tracing-seed 22" in line for line in route_lines)
    assert all("--ray-tracing-backend obdeect" in line for line in route_lines)
    assert all("shower_route.log" in line for line in route_lines)
    assert any("--telescope-ids 0,1" in line for line in route_lines)
    assert any("--telescope-ids 2" in line for line in route_lines)
    if input_file == "-":
        assert len(lines) == 2
        assert all(" | env -u MULTIPIPE_SEQUENTIAL " in line for line in lines)
        assert all("--input - --output -" in line for line in lines)
    else:
        assert len(lines) == 4
        assert all("events.corsika.zst" in line for line in route_lines)
        assert all(".camera.iact.gz" in line for line in lines)
    assert any("Bypass_Optics=1" in line for line in lines)
    assert any("Bypass_Optics=2" in line for line in lines)
    assert len(simulator.file_list["sim_telarray_output"]) == 2
    assert all(
        path.name.startswith(("LST_", "MST_"))
        for path in simulator.file_list["sim_telarray_output"]
    )
    mapping = json.loads(
        simulator.runner_service.get_file_name("telescope_route_map", 1).read_text()
    )
    assert mapping["2"]["dual_mirror"] is True
    assert math.isclose(mapping["0"]["input_plane_height_m"], 29.9)


def test_seed_file_requires_instrument_seed():
    seed = SimpleNamespace(seed_string="file-by-run:seeds,55", instrument_seed=None)
    with pytest.raises(ValueError, match="Instrument seed is not set"):
        _shower_route_seeds(seed)


def test_seed_file_uses_both_configured_seeds():
    seed = SimpleNamespace(
        seed_string="file-by-run:seeds,55", instrument_seed=66, simulation_seed=55
    )
    assert _shower_route_seeds(seed) == (66, 55)


@pytest.mark.parametrize("mode", ["stream", "disk"])
def test_chain_writes_routing_at_the_correct_stage(
    route_inputs,
    corsika_config_mock_array_model,
    corsika_stage_mock_array_model,
    tmp_test_directory,
    mode,
):
    from pathlib import Path

    from simtools.runners.chain_runner import ChainRunner
    from simtools.runners.corsika_stage import CorsikaStage
    from simtools.runners.stages import StageInputs

    routing, simulator = route_inputs

    class RouteStage:
        name = "sim_telarray"
        streams = mode == "stream"
        shower_routing = routing
        file_list = {}

        @classmethod
        def instances(cls, inputs, label=None):
            return [cls()]

        def prepare(self, run_number, input_file=None, output=None):
            return routing.commands(simulator, run_number, input_file)

    config = corsika_config_mock_array_model
    inputs = StageInputs(
        array_models=(config.array_model,),
        simulation_parameters=(config.simulation_parameters,),
        shower_configurations=(config,),
        args={"shower_route_mode": mode},
        context=corsika_stage_mock_array_model.context.replace(
            software=SimpleNamespace(
                **{
                    **vars(corsika_stage_mock_array_model.context.software),
                    "ray_tracing_backend": "obdeect",
                }
            )
        ),
    )
    runner = ChainRunner(SimpleNamespace(stage_classes=(CorsikaStage, RouteStage)), inputs)
    script = Path(tmp_test_directory) / "run.sh"
    runner.prepare_run(1, script)
    corsika = runner.stages[0][0]
    multipipe = corsika.stream_service.get_file_name("multi_pipe_config", 1).read_text()
    text = script.read_text()
    if mode == "stream":
        assert multipipe.count("simtools.simtel.shower_route") == 2
        assert "simtools.simtel.shower_route" not in text
    else:
        assert "simtools.simtel.shower_route" not in multipipe
        assert "zstd > " in multipipe
        assert text.count("simtools.simtel.shower_route") == 2
        assert text.index("# Running corsika") < text.index("simtools.simtel.shower_route")


def test_route_command_preserves_git_model_source(route_inputs):
    routing, simulator = route_inputs
    routing.context = routing.context.replace(
        model_reader=SimpleNamespace(
            source_config={"type": "git", "repository": "/models.git", "commit": "abc123"}
        )
    )
    lines = routing.commands(simulator, 1, "-")
    assert all("SIMTOOLS_SIMULATION_MODELS_GIT_PATH=/models.git" in line for line in lines)
    assert all("SIMTOOLS_SIMULATION_MODELS_GIT_REVISION=abc123" in line for line in lines)
    assert all("-u SIMTOOLS_SIMULATION_MODELS_PATH" in line for line in lines)
