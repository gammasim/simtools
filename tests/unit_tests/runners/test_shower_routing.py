"""Tests for shower photon routing commands."""

import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import pytest

from simtools.job_execution.runner_services import RunnerServices
from simtools.runners.shower_routing import ShowerRouting, _shower_route_seeds

SHOWER_ROUTE_MODULE = "simtools.simtel.shower_route"
PIPE_SEPARATOR = " | "


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


def _option(args, name):
    """Return the value following a command-line option."""
    return args[args.index(name) + 1]


def _split_stream_line(line):
    """Return the route and sim_telarray shell texts of a piped (stream) line."""
    route_text, simtel_text = line.split(PIPE_SEPARATOR, 1)
    return route_text, simtel_text


@pytest.mark.parametrize("input_file", ["-", "events.corsika.zst"])
def test_routing_commands_by_telescope_type(route_inputs, input_file):
    routing, simulator = route_inputs
    lines = routing.commands(simulator, 1, input_file)
    streamed = input_file == "-"

    # One route and one sim_telarray command per telescope type.
    if streamed:
        pairs = [_split_stream_line(line) for line in lines]
        route_texts = [route for route, _ in pairs]
        simtel_texts = [simtel for _, simtel in pairs]
        assert all(PIPE_SEPARATOR in line for line in lines)
    else:
        assert not any(PIPE_SEPARATOR in line for line in lines)
        route_texts = [line for line in lines if SHOWER_ROUTE_MODULE in line]
        simtel_texts = [line for line in lines if SHOWER_ROUTE_MODULE not in line]
    assert len(route_texts) == 2
    assert len(simtel_texts) == 2

    # Options of the route itself, after the process accounting wrapper's own options.
    route_args = [shlex.split(text) for text in route_texts]
    route_args = [args[args.index(SHOWER_ROUTE_MODULE) :] for args in route_args]
    assert {_option(args, "--telescope-ids") for args in route_args} == {"0,1", "2"}
    for args in route_args:
        assert _option(args, "--seed") == "11"
        assert _option(args, "--tracing-seed") == "22"
        assert _option(args, "--ray-tracing-backend") == "obdeect"
    if streamed:
        for args in route_args:
            assert _option(args, "--input") == "-"
            assert _option(args, "--output") == "-"
    else:
        for args in route_args:
            assert Path(_option(args, "--input")).name == "events.corsika.zst"

    # sim_telarray reads from stdin for streams, and from the camera bunch file otherwise.
    for text in simtel_texts:
        last_argument = shlex.split(text)[-1]
        if streamed:
            assert "env -u MULTIPIPE_SEQUENTIAL" in text
            assert last_argument == "-"
        else:
            assert last_argument.endswith(".camera.iact.gz")

    bypass_levels = {re.search(r"Bypass_Optics=(\d+)", text).group(1) for text in simtel_texts}
    assert bypass_levels == {"1", "2"}
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


def test_route_shell_text_fails_the_job_and_shows_the_log(tmp_test_directory):
    log_file = tmp_test_directory / "route.log"
    failing = [
        sys.executable,
        "-c",
        "import sys; print('route broke', file=sys.stderr); sys.exit(3)",
    ]
    script = f"set -e\n{ShowerRouting._route_shell_text(failing, log_file)}\n"

    result = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 3
    assert "route broke" in result.stderr
    assert "route broke" in log_file.read_text(encoding="utf-8")


def test_route_shell_text_passes_on_success(tmp_test_directory):
    log_file = tmp_test_directory / "route.log"
    succeeding = [sys.executable, "-c", "print('ok')"]
    script = f"set -e\n{ShowerRouting._route_shell_text(succeeding, log_file)}\n"

    result = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert log_file.exists()
