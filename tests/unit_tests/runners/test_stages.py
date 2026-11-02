"""Tests for simulation stages and chains."""

import pytest

from simtools.runners import stages
from simtools.runners.corsika_stage import CorsikaStage
from simtools.runners.sim_telarray_stage import SimTelArrayStage


@pytest.mark.parametrize(
    ("software", "expected"),
    [
        ("corsika", ("corsika",)),
        ("sim_telarray", ("sim_telarray",)),
        ("corsika_sim_telarray", ("corsika", "sim_telarray")),
    ],
)
def test_chain_from_software(software, expected):
    chain = stages.SimulationChain.from_software(software)

    assert chain.stages == expected
    assert chain.software == software
    assert chain.starts_with_showers == (expected[0] == "corsika")
    assert chain.reads_input_file == (expected[0] != "corsika")
    assert all(chain.has(name) for name in expected)


def test_chain_stage_classes():
    chain = stages.SimulationChain(("corsika", "sim_telarray"))

    assert chain.stage_classes == [CorsikaStage, SimTelArrayStage]
    assert CorsikaStage.produces == SimTelArrayStage.consumes == stages.CHERENKOV_PHOTONS


def test_register_stage_extends_chains():
    class FakeStage:
        name = "fake_stage"
        consumes = stages.CHERENKOV_PHOTONS
        produces = stages.CAMERA_EVENTS
        streams = True

    stages.register_stage("fake_stage", lambda: FakeStage)
    try:
        chain = stages.SimulationChain(("corsika", "fake_stage"))
        assert chain.stage_classes[1] is FakeStage
        assert chain.has("fake_stage")
    finally:
        stages.STAGES.pop("fake_stage")


def test_chain_rejects_unknown_software_and_stage():
    with pytest.raises(ValueError, match="Invalid simulation software"):
        stages.SimulationChain.from_software("unknown")
    with pytest.raises(ValueError, match="Unknown simulation stage 'unknown'"):
        stages.SimulationChain(("unknown",))
    with pytest.raises(ValueError, match="at least one stage"):
        stages.SimulationChain(())


def test_chain_rejects_mismatched_data_products():
    with pytest.raises(ValueError, match="'corsika' consumes None, but 'sim_telarray' produces"):
        stages.SimulationChain(("sim_telarray", "corsika"))
