"""Tests for the optics stage of the obdeect chain."""

import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.runners import stages
from simtools.runners.chain_runner import ChainRunner
from simtools.runners.chain_spec import parse_chain
from simtools.runners.stages import STDIN, StageInputs, StreamTo, get_stage_class
from simtools.simulation_software.optics import derive_seed

CHERENKOV_PHOTONS = "cherenkov_photons"
CAMERA_PHOTONS_IACT = "camera_photons_iact"
PHOTONS_CAMERA = "photons_camera"
PHOTONS_GROUND = "photons_ground"
ObdeectOpticsStage = get_stage_class("obdeect_optics")

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


MIRROR_REFLECTIVITY = "mirror_reflectivity"
TELESCOPE_TRANSMISSION = "telescope_transmission"
OPTICS_CHAIN = ["obdeect_optics"]


class _RunFiles:
    """File naming of a run: ``<directory>/run<N>.<stage><suffix of the file type>``."""

    SUFFIXES = {"photon_stream": ".pst", "stage_log": ".log", "stage_resources": ".resources.json"}

    def __init__(self, directory):
        self.directory = Path(directory)

    def get_stage_file_name(self, stage_name, file_type, run_number=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"run{run_number:06d}.{stage_name}{self.SUFFIXES[file_type]}"


@pytest.fixture
def context(tmp_test_directory):
    output = Path(tmp_test_directory) / "output"
    paths = SimpleNamespace(get_output_directory=lambda: output)
    return SimpleNamespace(paths=paths)


@pytest.fixture
def run_files(context):
    return _RunFiles(context.paths.get_output_directory())


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


def _inputs(array_model, context, run_files, chain_items=OPTICS_CHAIN, args=None):
    parameters = SimpleNamespace(zenith_angle=20.0, azimuth_angle=180.0)
    return StageInputs(
        array_models=(array_model,),
        simulation_parameters=(parameters,),
        shower_configurations=(),
        args={} if args is None else args,
        context=context,
        chain=parse_chain(list(chain_items)),
        calibration=SimpleNamespace(runner_service=run_files),
    )


@pytest.fixture
def inputs(array_model, context, run_files):
    return _inputs(array_model, context, run_files)


@pytest.fixture
def stage(inputs):
    (stage,) = ObdeectOpticsStage.instances(inputs)
    return stage


def _all_cores():
    return os.process_cpu_count() or 1


def _command_tokens(line):
    """Return the tokens of the obdeect command inside the accounting wrapper of a line."""
    text = line.split(" | ")[0].removeprefix("{ ").split(" 2> ")[0]
    tokens = shlex.split(text)
    return tokens[tokens.index("--") + 1 :]


def test_stage_attributes():
    assert ObdeectOpticsStage.name == "obdeect_optics"
    assert ObdeectOpticsStage.consumes == (PHOTONS_GROUND, "photons_telescope")
    assert ObdeectOpticsStage.produces == PHOTONS_CAMERA
    assert ObdeectOpticsStage.streams


def test_instances_create_one_stage_for_the_array_model(inputs):
    (stage,) = ObdeectOpticsStage.instances(inputs, label="test")

    assert [r.telescope_name for r in stage.requests] == ["LSTN-01", "MSTN-01", "LSTN-02"]
    assert [r.telescope_type for r in stage.requests] == ["LST", "MST", "LST"]
    # Stream ids start at 1; the instrument seed follows the IACT id (from 0).
    assert [r.telescope_id for r in stage.requests] == [1, 2, 3]
    assert [r.instrument_seed for r in stage.requests] == [
        derive_seed(11, telescope_id) for telescope_id in (0, 1, 2)
    ]
    assert stage.requests[0].zenith_deg == pytest.approx(20.0)
    assert stage.telescope_models[2] is inputs.array_models[0].telescope_models["LSTN-02"]


def test_instances_take_options(inputs, array_model, context, run_files):
    chained = _inputs(
        array_model,
        context,
        run_files,
        chain_items=["obdeect_optics:telescope_transmission=full,threads=3"],
    )

    (stage,) = ObdeectOpticsStage.instances(chained)

    assert {r.telescope_transmission for r in stage.requests} == {"full"}
    assert {r.threads for r in stage.requests} == {3}


def test_threads_default_to_all_cores(inputs):
    (stage,) = ObdeectOpticsStage.instances(inputs)

    assert {r.threads for r in stage.requests} == {_all_cores()}


def test_threads_of_sequential_run_is_one(inputs):
    sequential = StageInputs(
        array_models=inputs.array_models,
        simulation_parameters=inputs.simulation_parameters,
        shower_configurations=(),
        args={"sequential": True},
        context=inputs.context,
        chain=inputs.chain,
        calibration=inputs.calibration,
    )
    (stage,) = ObdeectOpticsStage.instances(sequential)

    assert {r.threads for r in stage.requests} == {1}


def test_instances_need_context(array_model, run_files):
    without_context = StageInputs(
        array_models=(array_model,),
        simulation_parameters=(SimpleNamespace(zenith_angle=20.0, azimuth_angle=180.0),),
        shower_configurations=(),
        args={},
        chain=parse_chain(OPTICS_CHAIN),
    )
    with pytest.raises(ValueError, match="needs a run context"):
        ObdeectOpticsStage.instances(without_context)


def test_applied_factors():
    assert ObdeectOpticsStage.spec.applied_factors({}) == {
        MIRROR_REFLECTIVITY,
        TELESCOPE_TRANSMISSION,
    }


def test_chain_option_rejects_unsupported_transmission():
    with pytest.raises(ValueError, match="Invalid telescope_transmission option of stage"):
        parse_chain(["obdeect_optics:telescope_transmission=angular"])


def test_prepare_traces_all_telescopes_in_one_command(stage, context):
    (line,) = stage.prepare(5, input_file=STDIN)

    tokens = _command_tokens(line)
    assert tokens[tokens.index("ray-tracing") :][:2] == ["ray-tracing", "trace-stream"]
    assert tokens[tokens.index("--input") + 1] == "-"
    assert tokens[tokens.index("--output") + 1] == str(stage.output_file(5))
    assert "--telescopes" not in tokens
    models = [tokens[index + 1] for index, token in enumerate(tokens) if token == "--model"]
    expected = [
        f"{request.telescope_id}={telescope.config_file_directory / request.model_file_name}"
        for request, telescope in zip(stage.requests, stage.telescope_models)
    ]
    assert models == expected
    assert all(Path(model.split("=", 1)[1]).is_file() for model in models)
    assert stage.output_file(5).name == "run000005.obdeect_optics.pst"
    assert stage.output_file(5).parent == context.paths.get_output_directory()
    assert stage.file_list["obdeect_optics_log"] == [
        context.paths.get_output_directory() / "run000005.obdeect_optics.log"
    ]
    assert len(stage.file_list["obdeect_optics_resources"]) == 1


def test_prepare_reading_file(stage):
    (line,) = stage.prepare(5, input_file="photons.pst")

    tokens = _command_tokens(line)
    assert tokens[tokens.index("--input") + 1] == "photons.pst"


def test_prepare_requires_input(stage):
    with pytest.raises(ValueError, match="needs an input photon stream"):
        stage.prepare(5)


def test_prepare_streams_to_one_downstream_command(stage):
    (line,) = stage.prepare(5, input_file=STDIN, output=StreamTo(("consume",)))

    assert line.endswith(" | consume")
    assert _command_tokens(line)[-1] == "-"


def test_prepare_rejects_several_downstream_commands(stage):
    with pytest.raises(ValueError, match="pipes into one downstream command, got 2"):
        stage.prepare(5, input_file=STDIN, output=StreamTo(("a", "b")))


def test_failing_command_stops_multipipe(stage):
    (line,) = stage.prepare(5, input_file=STDIN)

    assert "MULTIPIPE_PID" in line
    assert "obdeect_optics.log" in line


def _fake_class(name, consumes, produces, lines=()):
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

        def prepare(self, run_number, input_file=None, output=None):
            del run_number
            if output is not None:
                return [f"{name} | {command}" for command in output.commands]
            return [f"{name}-{self.tag} {input_file}"]

        def output_file(self, run_number):
            return Path(f"{name}-{run_number}.pst")

    Fake.name, Fake.consumes, Fake.produces = name, consumes, produces
    return Fake


@pytest.fixture
def fake_chain_stages(monkeypatch):
    fakes = {
        "fake_to_stream": _fake_class("fake_to_stream", CHERENKOV_PHOTONS, PHOTONS_GROUND),
        "fake_camera": _fake_class(
            "consume", PHOTONS_CAMERA, CAMERA_PHOTONS_IACT, lines=("shared",)
        ),
    }
    for name, fake in fakes.items():
        monkeypatch.setitem(stages.registry(), name, fake)


def test_chain_is_valid(fake_chain_stages):
    chain = parse_chain(["fake_to_stream", "obdeect_optics", "fake_camera"])

    assert chain.stages == ("fake_to_stream", "obdeect_optics", "fake_camera")


def test_chain_without_converter_is_invalid(fake_chain_stages):
    with pytest.raises(ValueError, match="consumes"):
        parse_chain(["fake_camera", "obdeect_optics"])


def test_runner_streams_the_array_through_one_optics_process(
    array_model, context, run_files, fake_chain_stages, tmp_test_directory
):
    chain = parse_chain(["fake_to_stream", "obdeect_optics", "fake_camera"])
    inputs = _inputs(array_model, context, run_files, chain_items=chain.stages)
    script = Path(tmp_test_directory) / "run.sh"

    ChainRunner(chain, inputs).prepare_run(1, script)

    lines = script.read_text(encoding="utf-8").splitlines()
    optics_lines = [line for line in lines if "trace-stream" in line]
    assert len(optics_lines) == 1
    assert optics_lines[0].startswith("fake_to_stream | ")
    assert optics_lines[0].endswith("| consume-shared -")
