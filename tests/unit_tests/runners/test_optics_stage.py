"""Tests for the optics stage of the obdeect chain."""

import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.runners import stages
from simtools.runners.chain_runner import ChainRunner
from simtools.runners.chain_spec import (
    MIRROR_REFLECTIVITY,
    TELESCOPE_TRANSMISSION,
    ChainSpec,
)
from simtools.runners.optics_stage import ObdeectOpticsStage
from simtools.runners.stages import STDIN, SimulationChain, StageInputs, StreamTo, get_stage_class
from simtools.simtel.shower_photons import derive_seed
from simtools.simulation_software.products import (
    CHERENKOV_PHOTONS,
    PHOTONS_CAMERA,
    PHOTONS_GROUND,
    PHOTONS_TELESCOPE,
)

_MIRROR_LIST = """0 0 100.0 0.0 0
125 0 100.0 0.0 0
-125 0 100.0 0.0 0
62.5 108.3 100.0 0.0 0
-62.5 108.3 100.0 0.0 0
62.5 -108.3 100.0 0.0 0
-62.5 -108.3 100.0 0.0 0
"""


def _record(value, unit=None, file=False):
    return {"value": value, "unit": unit, "file": file, "parameter_version": "1.0.0"}


def _telescope(type_name, directory):
    """Telescope model of a single-mirror dish with seven 1 m mirrors (f = 8 m)."""
    directory.mkdir(exist_ok=True)
    (directory / "mirrors.dat").write_text(_MIRROR_LIST, encoding="utf-8")
    parameters = {
        "mirror_class": _record(0),
        "focal_length": _record(800.0, "cm"),
        "mirror_offset": _record(-50.0, "cm"),
        "dish_shape_length": _record(800.0, "cm"),
        "parabolic_dish": _record(False),
        "mirror_focal_length": _record(0.0, "cm"),
        "mirror_list": _record("mirrors.dat", file=True),
        "focus_offset": _record([0.0, 20.0, 0.0, 0.0], ["cm", "deg", None, None]),
        "camera_body_diameter": _record(60.0, "cm"),
        "camera_body_shape": _record(2),
        "camera_depth": _record(0.0, "cm"),
        "random_focal_length": _record([20.0, 0.0], ["cm", "cm"]),
        "mirror_reflection_random_angle": _record([0.0, 0.0, 0.0], ["deg", "deg", None]),
        "mirror_align_random_distance": _record(0.0, "cm"),
        "mirror_align_random_horizontal": _record(
            [0.02, 20.0, 0.0, 0.0], ["deg", "deg", None, None]
        ),
        "mirror_align_random_vertical": _record([0.0, 20.0, 0.0, 0.0], ["deg", "deg", None, None]),
        "telescope_transmission": _record([0.9, 0, 0, 0, 0, 0]),
        "pixel_shape": _record(1),
    }
    return SimpleNamespace(
        name=type_name,
        model_version="9.9.9",
        model_files_exported=True,
        parameters=parameters,
        config_file_directory=directory,
        get_parameter_value=lambda name: parameters[name]["value"],
    )


@pytest.fixture
def context(tmp_test_directory):
    output = Path(tmp_test_directory) / "output"
    paths = SimpleNamespace(get_output_directory=lambda: output)
    return SimpleNamespace(paths=paths)


@pytest.fixture
def array_model(tmp_test_directory):
    pytest.importorskip("obdeect.ray_tracing")
    names = {"LSTN-01": "LST", "MSTN-01": "MST", "LSTN-02": "LST"}
    directory = Path(tmp_test_directory) / "model"
    return SimpleNamespace(
        telescope_models={
            name: _telescope(type_name, directory) for name, type_name in names.items()
        },
        model_version="7.0.0",
        site="North",
        sim_telarray_seed=SimpleNamespace(seed_string="11,22", instrument_seed=None),
    )


@pytest.fixture
def inputs(array_model, context):
    parameters = SimpleNamespace(zenith_angle=20.0, azimuth_angle=180.0)
    return StageInputs(
        array_models=(array_model,),
        simulation_parameters=(parameters,),
        shower_configurations=(),
        args={},
        context=context,
    )


def _command_tokens(line):
    """Return the tokens of the obdeect command inside the accounting wrapper of a line."""
    text = line.split(" | ")[0].removeprefix("{ ").split(" 2> ")[0]
    tokens = shlex.split(text)
    return tokens[tokens.index("--") + 1 :]


def test_stage_attributes():
    assert ObdeectOpticsStage.name == "obdeect_optics"
    assert ObdeectOpticsStage.consumes == (PHOTONS_GROUND, PHOTONS_TELESCOPE)
    assert ObdeectOpticsStage.produces == PHOTONS_CAMERA
    assert ObdeectOpticsStage.streams
    assert get_stage_class("obdeect_optics") is ObdeectOpticsStage


def test_instances_one_per_telescope(inputs):
    stages = ObdeectOpticsStage.instances(inputs, label="test")

    assert [stage.request.telescope_name for stage in stages] == ["LSTN-01", "MSTN-01", "LSTN-02"]
    assert [stage.request.telescope_type for stage in stages] == ["LST", "MST", "LST"]
    # Stream ids start at 1; the instrument seed follows the IACT id (from 0).
    assert [stage.request.telescope_id for stage in stages] == [1, 2, 3]
    assert [stage.request.instrument_seed for stage in stages] == [
        derive_seed(11, telescope_id) for telescope_id in (0, 1, 2)
    ]
    assert [(stage.position, stage.count) for stage in stages] == [(0, 3), (1, 3), (2, 3)]
    assert stages[0].request.zenith_deg == pytest.approx(20.0)
    assert stages[2].telescope_model is inputs.array_models[0].telescope_models["LSTN-02"]


def test_instances_of_several_array_models(inputs):
    two = StageInputs(
        array_models=inputs.array_models * 2,
        simulation_parameters=inputs.simulation_parameters * 2,
        shower_configurations=(),
        args={"sequential": True},
        context=inputs.context,
    )

    stages = ObdeectOpticsStage.instances(two)

    assert len(stages) == 6
    assert {stage.request.threads for stage in stages} == {1}


def test_instances_take_options(inputs):
    options = {"obdeect_optics": {"telescope_transmission": "full", "threads": 3}}
    chained = StageInputs(
        array_models=inputs.array_models,
        simulation_parameters=inputs.simulation_parameters,
        shower_configurations=(),
        args={},
        context=inputs.context,
        chain_options=options,
    )

    stages = ObdeectOpticsStage.instances(chained)

    assert {stage.request.telescope_transmission for stage in stages} == {"full"}
    assert {stage.request.threads for stage in stages} == {3}


def test_instances_need_context(inputs):
    without_context = StageInputs(
        array_models=inputs.array_models,
        simulation_parameters=inputs.simulation_parameters,
        shower_configurations=(),
        args={},
    )
    with pytest.raises(ValueError, match="needs a run context"):
        ObdeectOpticsStage.instances(without_context)


def test_applied_factors():
    assert ObdeectOpticsStage.applied_factors({}) == {MIRROR_REFLECTIVITY, TELESCOPE_TRANSMISSION}
    with pytest.raises(ValueError, match="stay unapplied"):
        ObdeectOpticsStage.applied_factors({"telescope_transmission": "angular"})
    with pytest.raises(ValueError, match="Invalid telescope transmission"):
        ObdeectOpticsStage.applied_factors({"telescope_transmission": "none"})


def test_prepare_streaming_input_to_file(inputs, context):
    stage = ObdeectOpticsStage.instances(inputs)[0]

    (line,) = stage.prepare(5, input_file=STDIN)

    tokens = _command_tokens(line)
    assert tokens[tokens.index("ray-tracing") :][:2] == ["ray-tracing", "trace-stream"]
    assert tokens[tokens.index("--input") + 1] == "-"
    assert tokens[tokens.index("--output") + 1] == str(stage.output_file(5))
    assert tokens[tokens.index("--telescopes") + 1] == "1"
    model_file = stage.telescope_model.config_file_directory / stage.request.model_file_name
    assert tokens[tokens.index("--model") + 1] == str(model_file)
    assert model_file.is_file()
    assert stage.output_file(5).name == "LSTN-01_7.0.0_run000005_obdeect_optics.camera.pst"
    assert stage.output_file(5).parent == context.paths.get_output_directory()
    assert len(stage.file_list["obdeect_optics_resources"]) == 1


def test_prepare_reading_file(inputs):
    stage = ObdeectOpticsStage.instances(inputs)[0]

    (line,) = stage.prepare(5, input_file="photons.pst")

    tokens = _command_tokens(line)
    assert tokens[tokens.index("--input") + 1] == "photons.pst"


def test_prepare_requires_input(inputs):
    stage = ObdeectOpticsStage.instances(inputs)[0]
    with pytest.raises(ValueError, match="needs an input photon stream"):
        stage.prepare(5)


def test_prepare_streamed_output_pipes_to_matching_downstream(inputs):
    first, second, _third = ObdeectOpticsStage.instances(inputs)
    downstream = StreamTo(("consume-a", "consume-b", "consume-c"))

    (line_first,) = first.prepare(5, input_file=STDIN, output=downstream)
    (line_second,) = second.prepare(5, input_file=STDIN, output=downstream)

    assert line_first.endswith(" | consume-a")
    assert line_second.endswith(" | consume-b")
    assert _command_tokens(line_first)[-1] == "-"


def test_prepare_single_downstream_command(inputs):
    stage = ObdeectOpticsStage.instances(inputs)[0]

    (line,) = stage.prepare(5, input_file=STDIN, output=StreamTo(("consume",)))

    assert line.endswith(" | consume")


def test_prepare_mismatching_downstream_commands(inputs):
    stage = ObdeectOpticsStage.instances(inputs)[0]
    with pytest.raises(ValueError, match="downstream commands receive its stream"):
        stage.prepare(5, input_file=STDIN, output=StreamTo(("a", "b")))


def test_failing_command_stops_multipipe(inputs):
    stage = ObdeectOpticsStage.instances(inputs)[0]

    (line,) = stage.prepare(5, input_file=STDIN)

    assert "MULTIPIPE_PID" in line
    assert "obdeect_optics.log" in line


def test_environment_and_resources(inputs):
    stage = ObdeectOpticsStage.instances(inputs)[0]

    assert stage.environment() == {}
    assert stage.get_resources() is None


def _fake_class(name, consumes, produces, factors=(), lines=()):
    """Return a stage class standing in for a stage of another module."""

    class Fake:
        streams = True

        def __init__(self, tag=None):
            self.tag = tag
            self.file_list = {}

        @classmethod
        def instances(cls, _inputs, label=None):
            del label
            return [cls(tag) for tag in lines] or [cls()]

        @classmethod
        def applied_factors(cls, _options):
            return set(factors)

        def prepare(self, run_number, input_file=None, output=None):
            del run_number
            if output is not None:
                return [f"{name} | {command}" for command in output.commands]
            return [f"{name}-{self.tag} {input_file}"]

    Fake.name, Fake.consumes, Fake.produces = name, consumes, produces
    return Fake


@pytest.fixture
def fake_chain_stages(monkeypatch):
    fakes = {
        "fake_to_stream": _fake_class("fake_to_stream", CHERENKOV_PHOTONS, PHOTONS_GROUND),
        "fake_camera": _fake_class(
            "consume", PHOTONS_CAMERA, "camera_events", lines=("a", "b", "c")
        ),
        "fake_double": _fake_class(
            "fake_double", PHOTONS_CAMERA, "camera_events", {MIRROR_REFLECTIVITY}
        ),
    }
    for name, fake in fakes.items():
        monkeypatch.setitem(stages.STAGES, name, lambda fake=fake: fake)


def test_chain_is_valid(fake_chain_stages):
    chain = SimulationChain(("fake_to_stream", "obdeect_optics", "fake_camera"))

    ChainSpec(chain.stages).check_applied_factors()
    assert chain.has("obdeect_optics")


def test_chain_without_converter_is_invalid(fake_chain_stages):
    with pytest.raises(ValueError, match="consumes"):
        SimulationChain(("fake_camera", "obdeect_optics"))


def test_factor_applied_twice_is_rejected(fake_chain_stages):
    spec = ChainSpec(("fake_to_stream", "obdeect_optics", "fake_double"))

    with pytest.raises(ValueError, match="applies 'mirror_reflectivity' twice"):
        spec.check_applied_factors()


def test_runner_pipes_each_telescope_into_its_own_command(
    inputs, fake_chain_stages, tmp_test_directory
):
    chain = SimulationChain(("fake_to_stream", "obdeect_optics", "fake_camera"))
    script = Path(tmp_test_directory) / "run.sh"

    ChainRunner(chain, inputs).prepare_run(1, script)

    lines = script.read_text(encoding="utf-8").splitlines()
    optics_lines = [line for line in lines if "trace-stream" in line]
    assert len(optics_lines) == 3
    for line, tag in zip(optics_lines, "abc"):
        assert line.endswith(f"| consume-{tag} -")


def test_runner_pipes_all_telescopes_into_a_shared_command(
    inputs, fake_chain_stages, monkeypatch, tmp_test_directory
):
    shared = _fake_class("consume", PHOTONS_CAMERA, "camera_events", lines=("shared",))
    monkeypatch.setitem(stages.STAGES, "fake_camera", lambda: shared)
    chain = SimulationChain(("fake_to_stream", "obdeect_optics", "fake_camera"))
    script = Path(tmp_test_directory) / "run.sh"

    ChainRunner(chain, inputs).prepare_run(1, script)

    optics_lines = [
        line for line in script.read_text(encoding="utf-8").splitlines() if "trace-stream" in line
    ]
    assert len(optics_lines) == 3
    assert all(line.endswith("| consume-shared -") for line in optics_lines)
