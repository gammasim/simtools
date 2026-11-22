"""Tests for the run script of a calibration light source built from a simulation chain."""

import pickle

import pytest

from simtools.runners import light_emission_runner, stages
from simtools.runners.chain_spec import SimulationChain, parse_chain
from simtools.runtime import software_catalog

OBDEECT_FLASHER_CHAIN = [
    "obdeect_flasher",
    "stream_to_simtel",
    "sim_telarray:bypass_optics=camera_entrance",
]
OBDEECT_ILLUMINATOR_CHAIN = [
    "obdeect_illuminator",
    "obdeect_atmosphere",
    "obdeect_optics",
    "stream_to_simtel",
    "sim_telarray:bypass_optics=camera_only",
]
STANDARD_CHAIN = ["light_emission", "sim_telarray"]


@pytest.mark.parametrize(
    "items", [OBDEECT_FLASHER_CHAIN, OBDEECT_ILLUMINATOR_CHAIN, STANDARD_CHAIN]
)
def test_light_chains_are_valid(items):
    chain = parse_chain(items)
    light_emission_runner.check_light_chain(chain)
    assert chain.source == stages.LIGHT_SOURCE


@pytest.mark.parametrize(
    "chain_stages",
    [
        ("corsika", "sim_telarray"),
        ("sim_telarray",),
    ],
)
def test_check_light_chain_rejects_chains_without_light_source(chain_stages):
    chain = SimulationChain(chain_stages)
    with pytest.raises(ValueError, match="starts with a light-source stage"):
        light_emission_runner.check_light_chain(chain)


def test_stage_inputs(light_simulation):
    light_simulation.light_emission_config["telescopes"] = ["LSTN-01", "LSTN-02"]
    chain = parse_chain(STANDARD_CHAIN)
    inputs = light_emission_runner.stage_inputs(light_simulation, chain)
    assert inputs.calibration is light_simulation
    assert inputs.chain is chain
    assert inputs.args["telescope"] == "MSTN-04"
    assert inputs.args["light_source"] == "MSFx-FlashCam"
    assert inputs.options_of("sim_telarray")["bypass_optics"] == "none"
    assert inputs.stages == ("light_emission", "sim_telarray")
    (array,) = inputs.array_models
    assert list(array.telescope_models) == ["MSTN-04"]
    assert (array.site, array.model_version) == ("North", "6.0.2")
    assert array.sim_telarray_seed.seed_string == "11,13"
    (pointing,) = inputs.simulation_parameters
    assert (pointing.zenith_angle, pointing.azimuth_angle) == (0.0, 0.0)
    assert inputs.shower_configurations == ()


def test_chain_script_runs_stages_in_order(light_simulation, tmp_test_directory):
    script = tmp_test_directory / "chain.sh"
    light_emission_runner.write_chain_script(light_simulation, STANDARD_CHAIN, script)
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["#!/usr/bin/env bash", "set -e", "set -o pipefail"]
    iact_file = tmp_test_directory / "iact_output.dat"
    assert lines[3] == f"ff-1m -o {iact_file}"
    assert lines[4] == (
        f"[ -s '{iact_file}' ] || {{ echo 'LightEmission did not produce IACT file' >&2; exit 1; }}"
    )
    assert "-C Bypass_Optics=1" in lines[5]
    assert f"-C input_file={iact_file}" in lines[5]
    assert lines[6] == f"rm -f '{iact_file}'"
    assert len(lines) == 7


def test_chain_script_rejects_shower_chain(light_simulation, tmp_test_directory):
    with pytest.raises(ValueError, match="starts with a light-source stage"):
        light_emission_runner.write_chain_script(
            light_simulation, ["corsika", "sim_telarray"], tmp_test_directory / "chain.sh"
        )


def test_build_chain_runner_has_one_instance_per_stage(light_simulation):
    runner = light_emission_runner.build_chain_runner(light_simulation, STANDARD_CHAIN)
    assert [len(instances) for instances in runner.stages] == [1, 1]


@pytest.mark.parametrize(
    ("short", "full", "bypass_optics"),
    [
        (
            ["obdeect_flasher", "stream_to_simtel", "sim_telarray:bypass_optics=camera_entrance"],
            ("obdeect_flasher", "stream_to_simtel", "sim_telarray"),
            "camera_entrance",
        ),
        (
            [
                "obdeect_illuminator",
                "obdeect_atmosphere",
                "obdeect_optics",
                "stream_to_simtel",
                "sim_telarray:bypass_optics=camera_only",
            ],
            (
                "obdeect_illuminator",
                "obdeect_atmosphere",
                "obdeect_optics",
                "stream_to_simtel",
                "sim_telarray",
            ),
            "camera_only",
        ),
    ],
)
def test_light_chains_with_explicit_converters(short, full, bypass_optics):
    chain = parse_chain(short)
    light_emission_runner.check_light_chain(chain)
    assert chain.stages == full
    assert chain.stage_options("sim_telarray")["bypass_optics"] == bypass_optics


def test_script_writer_for_chain_is_none_for_default_chain():
    assert light_emission_runner.script_writer_for_chain(None) is None
    assert light_emission_runner.script_writer_for_chain([]) is None
    default = list(software_catalog.default_chain(software_catalog.LIGHT_SOURCE))
    assert light_emission_runner.script_writer_for_chain(default) is None
    assert light_emission_runner.script_writer_for_chain(tuple(default)) is None


def test_script_writer_for_chain_writes_chain_script(light_simulation, tmp_test_directory):
    writer = light_emission_runner.script_writer_for_chain(
        ["light_emission", "sim_telarray:bypass_optics=none"]
    )
    assert writer is not None
    script = tmp_test_directory / "chain_by_writer.sh"
    writer(light_simulation, script)
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[3] == f"ff-1m -o {tmp_test_directory / 'iact_output.dat'}"
    assert "-C Bypass_Optics=1" in lines[5]


def test_script_writer_for_chain_is_picklable():
    writer = light_emission_runner.script_writer_for_chain(OBDEECT_FLASHER_CHAIN)
    restored = pickle.loads(pickle.dumps(writer))
    assert restored.keywords == {"items": tuple(OBDEECT_FLASHER_CHAIN)}


def test_script_writer_for_chain_accepts_single_item_string():
    writer = light_emission_runner.script_writer_for_chain("light_emission")
    assert writer is not None
    assert writer.keywords == {"items": ("light_emission",)}
