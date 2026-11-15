"""Tests for simulation stages and chains."""

import pytest

from simtools.runners import stages
from simtools.runners.corsika_stage import CorsikaStage
from simtools.runners.sim_telarray_stage import SimTelArrayStage


@pytest.mark.parametrize(
    ("names", "software"),
    [
        (("corsika",), {"corsika"}),
        (("sim_telarray",), {"sim_telarray"}),
        (("corsika", "sim_telarray"), {"corsika", "sim_telarray"}),
        (
            ("corsika", "corsika_to_stream", "obdeect_optics", "stream_to_simtel", "sim_telarray"),
            {"corsika", "obdeect", "sim_telarray"},
        ),
    ],
)
def test_chain_software_and_first_stage(names, software):
    chain = stages.SimulationChain(names)

    assert chain.stages == names
    assert chain.software == software
    assert chain.starts_with_showers == (names[0] == "corsika")
    assert chain.reads_input_file == (names[0] != "corsika")
    assert all(chain.has(name) for name in names)


def test_chain_stage_classes():
    chain = stages.SimulationChain(("corsika", "sim_telarray"))

    assert chain.stage_classes == [CorsikaStage, SimTelArrayStage]
    assert CorsikaStage.produces == stages.CHERENKOV_PHOTONS
    assert CorsikaStage.produces in SimTelArrayStage.consumes


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
        stages.STAGE_SOFTWARE.pop("fake_stage")


def test_chain_rejects_unknown_stage_and_empty_chain():
    with pytest.raises(ValueError, match="Unknown simulation stage 'unknown'"):
        stages.SimulationChain(("unknown",))
    with pytest.raises(ValueError, match="at least one stage"):
        stages.SimulationChain(())


def test_chain_rejects_mismatched_data_products():
    with pytest.raises(ValueError, match="'corsika' consumes None, but 'sim_telarray' produces"):
        stages.SimulationChain(("sim_telarray", "corsika"))


def test_stage_may_consume_several_products():
    class FakeStage:
        name = "fake_stage"
        consumes = (stages.CHERENKOV_PHOTONS, "other_photons")
        produces = stages.CAMERA_EVENTS
        streams = True

    stages.register_stage("fake_stage", lambda: FakeStage)
    try:
        assert stages.SimulationChain(("corsika", "fake_stage")).stages == ("corsika", "fake_stage")
        with pytest.raises(ValueError, match="consumes"):
            stages.SimulationChain(("fake_stage", "fake_stage"))
    finally:
        stages.STAGES.pop("fake_stage")


def test_stages_are_registered_lazily():
    assert {
        "corsika_to_stream",
        "stream_to_simtel",
        "obdeect_atmosphere",
        "obdeect_optics",
        "obdeect_flasher",
        "obdeect_illuminator",
    } <= set(stages.STAGES)


def test_stage_inputs_options_of_stage():
    inputs = stages.StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        chain_options={"obdeect_optics": {"threads": 2}},
    )

    assert inputs.options_of("obdeect_optics") == {"threads": 2}
    assert inputs.options_of("sim_telarray") == {}


def test_register_stage_declares_software():
    stages.register_stage("fake_stage", lambda: None, software=["fake_program"])
    try:
        assert stages.stages_software(["corsika", "fake_stage"]) == {"corsika", "fake_program"}
    finally:
        stages.STAGES.pop("fake_stage")
        stages.STAGE_SOFTWARE.pop("fake_stage")


def test_stages_software_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown simulation stage 'unknown'"):
        stages.stages_software(["corsika", "unknown"])


def test_converters_must_be_listed_explicitly():
    with pytest.raises(ValueError, match="consumes"):
        stages.SimulationChain(("corsika", "obdeect_atmosphere"))
