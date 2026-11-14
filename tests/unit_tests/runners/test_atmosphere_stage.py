"""Tests for the obdeect atmosphere simulation stage."""

import shlex
from pathlib import Path

import pytest

from simtools.photon_chain.atmosphere import AtmosphereRequest, propagation_program
from simtools.photon_chain.products import CAMERA_EVENTS, PHOTONS_GROUND, PHOTONS_SKY
from simtools.runners import stages
from simtools.runners.atmosphere_stage import ObdeectAtmosphereStage
from simtools.runners.chain_spec import ATMOSPHERIC_EXTINCTION, ChainSpec
from simtools.runners.stages import STDIN, SimulationChain, StageInputs, StreamTo


@pytest.fixture
def request_for_stage(tmp_test_directory):
    return AtmosphereRequest(
        site="North",
        observation_level_m=2158.0,
        profile_file=Path(tmp_test_directory) / "atmprof.dat",
        transmission_file=Path(tmp_test_directory) / "transmission.dat",
        seed=3,
        telescope_pathlen_m={1: 28.0},
    )


@pytest.fixture
def atmosphere_stage(request_for_stage, run_context):
    return ObdeectAtmosphereStage(request_for_stage, label="test-atmosphere", context=run_context)


def _tokens(line):
    return shlex.split(line)


def test_stage_protocol_attributes():
    assert ObdeectAtmosphereStage.name == "obdeect_atmosphere"
    assert ObdeectAtmosphereStage.consumes == (PHOTONS_SKY, PHOTONS_GROUND)
    assert ObdeectAtmosphereStage.produces == PHOTONS_GROUND
    assert ObdeectAtmosphereStage.streams is True


def test_applied_factors_is_atmospheric_extinction():
    assert ObdeectAtmosphereStage.applied_factors({}) == frozenset({ATMOSPHERIC_EXTINCTION})
    assert ObdeectAtmosphereStage.applied_factors({"seed": 2}) == frozenset(
        {ATMOSPHERIC_EXTINCTION}
    )


def test_prepare_reads_stdin_and_writes_file(atmosphere_stage, run_context):
    lines = atmosphere_stage.prepare(run_number=4, input_file=STDIN)

    output = atmosphere_stage.output_file(4)
    assert len(lines) == 1
    assert _tokens(lines[0]) == [
        propagation_program(),
        "--transmission",
        str(atmosphere_stage.request.transmission_file),
        "--site-altitude",
        "2158.0",
        "--profile",
        str(atmosphere_stage.request.profile_file),
        "--seed",
        "3",
        "--mode",
        "thinned",
        "--telescope-pathlen",
        "1=28.0",
        "-i",
        "-",
        "-o",
        str(output),
    ]
    assert output.parent.is_dir()
    assert output.parent == run_context.paths.get_output_directory("atmosphere")


def test_prepare_reads_input_file(atmosphere_stage):
    lines = atmosphere_stage.prepare(run_number=1, input_file="corsika_photons.pst")

    tokens = _tokens(lines[0])
    assert tokens[tokens.index("-i") + 1] == "corsika_photons.pst"


def test_prepare_pipes_streamed_output_into_downstream_command(atmosphere_stage):
    downstream = "sim_telarray -C input_file=-"

    lines = atmosphere_stage.prepare(run_number=1, input_file=STDIN, output=StreamTo((downstream,)))

    assert len(lines) == 1
    assert lines[0].endswith(f" | {downstream}")
    tokens = _tokens(lines[0].split(" | ")[0])
    assert tokens[tokens.index("-o") + 1] == "-"


def test_prepare_rejects_several_downstream_commands(atmosphere_stage):
    with pytest.raises(ValueError, match=r"pipes into one downstream command, got 2"):
        atmosphere_stage.prepare(
            run_number=1, input_file=STDIN, output=StreamTo(("cmd_a", "cmd_b"))
        )


def test_prepare_without_input_raises(atmosphere_stage):
    with pytest.raises(ValueError, match=r"needs an input photon stream"):
        atmosphere_stage.prepare(run_number=1, input_file=None)


def test_output_file_name_uses_label_and_run(atmosphere_stage):
    assert atmosphere_stage.output_file(7).name == "test-atmosphere-run000007.pst"


def test_environment_and_resources_are_empty(atmosphere_stage):
    assert atmosphere_stage.environment() == {}
    assert atmosphere_stage.get_resources() == {}


def test_instances_one_per_array_model(array_model_north, run_context):
    inputs = StageInputs(
        array_models=(array_model_north, array_model_north),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=run_context,
    )

    stages_of_run = ObdeectAtmosphereStage.instances(inputs, label="test")

    assert len(stages_of_run) == 2
    assert all(stage.request.site == "North" for stage in stages_of_run)
    assert all(stage.request.seed == 1 for stage in stages_of_run)
    assert all(stage.request.mode == "thinned" for stage in stages_of_run)


def test_instances_use_stage_options(array_model_north, run_context):
    inputs = StageInputs(
        array_models=(array_model_north,),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=run_context,
        chain_options={
            "obdeect_atmosphere": {
                "seed": 7,
                "mode": "weighted",
                "wavelength_range_nm": [250.0, 700.0],
            }
        },
    )

    (stage,) = ObdeectAtmosphereStage.instances(inputs)

    assert stage.request.seed == 7
    assert stage.request.mode == "weighted"
    assert stage.request.wavelength_range_nm == (250.0, 700.0)


def test_instances_reject_unknown_options(array_model_north, run_context):
    inputs = StageInputs(
        array_models=(array_model_north,),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=run_context,
        chain_options={"obdeect_atmosphere": {"threads": 4}},
    )

    with pytest.raises(ValueError, match=r"Unknown options of stage 'obdeect_atmosphere'"):
        ObdeectAtmosphereStage.instances(inputs)


@pytest.fixture
def fake_stage_factories(monkeypatch):
    """Register fake upstream stages (other stages are not part of this worktree)."""

    def _register(name, consumes, produces):
        fake = type(
            f"Fake_{name}",
            (),
            {"name": name, "consumes": consumes, "produces": produces, "streams": True},
        )
        monkeypatch.setitem(stages.STAGES, name, lambda: fake)

    return _register


def test_chain_accepts_photons_from_upstream_stage(fake_stage_factories):
    fake_stage_factories("fake_sky_source", None, PHOTONS_SKY)

    chain = SimulationChain(("fake_sky_source", "obdeect_atmosphere"))

    assert chain.stage_classes[-1] is ObdeectAtmosphereStage


def test_chain_rejects_upstream_product_it_does_not_consume(fake_stage_factories):
    fake_stage_factories("fake_camera", None, CAMERA_EVENTS)

    with pytest.raises(ValueError, match=r"'obdeect_atmosphere' consumes"):
        SimulationChain(("fake_camera", "obdeect_atmosphere"))


def test_chain_spec_rejects_two_atmospheres():
    spec = ChainSpec("two-atmospheres", ("obdeect_atmosphere", "obdeect_atmosphere"))

    with pytest.raises(ValueError, match=r"applies 'atmospheric_extinction' twice"):
        spec.chain()
