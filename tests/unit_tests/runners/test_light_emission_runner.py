"""Tests for the run script of a calibration light source built from a simulation chain."""

import pytest

from simtools.runners import chain_spec, light_emission_runner
from simtools.runners.stages import SimulationChain

OBDEECT_FLASHER_CHAIN = ["obdeect_flasher", "stream_to_simtel", "sim_telarray:bypass_optics=1"]
OBDEECT_ILLUMINATOR_CHAIN = [
    "obdeect_illuminator",
    "obdeect_atmosphere",
    "obdeect_optics",
    "stream_to_simtel",
    "sim_telarray:bypass_optics=3",
]
STANDARD_CHAIN = ["light_emission", "sim_telarray:bypass_optics=1"]


@pytest.mark.parametrize(
    "items", [OBDEECT_FLASHER_CHAIN, OBDEECT_ILLUMINATOR_CHAIN, STANDARD_CHAIN]
)
def test_light_chains_are_valid(items):
    chain = chain_spec.parse_chain(items).chain()
    light_emission_runner.check_light_chain(chain)
    assert chain.starts_with_showers


@pytest.mark.parametrize(
    "stages",
    [
        ("corsika", "sim_telarray"),
        ("sim_telarray",),
        ("stream_to_simtel", "sim_telarray"),
    ],
)
def test_check_light_chain_rejects_chains_without_light_source(stages):
    chain = SimulationChain(stages)
    with pytest.raises(ValueError, match="starts with a light source stage"):
        light_emission_runner.check_light_chain(chain)


def test_stage_inputs(light_simulation):
    light_simulation.light_emission_config["telescopes"] = ["LSTN-01", "LSTN-02"]
    spec = chain_spec.parse_chain(STANDARD_CHAIN)
    inputs = light_emission_runner.stage_inputs(light_simulation, spec, spec.chain())
    assert inputs.calibration is light_simulation
    assert inputs.args["telescope"] == "MSTN-04"
    assert inputs.args["light_source"] == "MSFx-FlashCam"
    assert inputs.chain_options == {"sim_telarray": {"bypass_optics": 1}}
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
    runner = light_emission_runner.write_chain_script(light_simulation, STANDARD_CHAIN, script)
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
    assert runner.environment() == {}


def test_chain_script_rejects_shower_chain(light_simulation, tmp_test_directory):
    with pytest.raises(ValueError, match="starts with a light source stage"):
        light_emission_runner.write_chain_script(
            light_simulation, ["corsika", "sim_telarray"], tmp_test_directory / "chain.sh"
        )


def test_build_chain_runner_has_one_instance_per_stage(light_simulation):
    runner = light_emission_runner.build_chain_runner(light_simulation, STANDARD_CHAIN)
    assert [len(instances) for instances in runner.stages] == [1, 1]


@pytest.mark.parametrize(
    ("short", "full"),
    [
        (
            ["obdeect_flasher", "sim_telarray:bypass_optics=1"],
            ("obdeect_flasher", "stream_to_simtel", "sim_telarray"),
        ),
        (
            [
                "obdeect_illuminator",
                "obdeect_atmosphere",
                "obdeect_optics",
                "sim_telarray:bypass_optics=3",
            ],
            (
                "obdeect_illuminator",
                "obdeect_atmosphere",
                "obdeect_optics",
                "stream_to_simtel",
                "sim_telarray",
            ),
        ),
    ],
)
def test_light_chains_without_converters_get_them_inserted(short, full):
    spec = chain_spec.parse_chain(short)
    chain = spec.chain()

    light_emission_runner.check_light_chain(chain)
    assert chain.stages == full
    assert spec.stage_options("sim_telarray")["bypass_optics"] in (1, 3)
