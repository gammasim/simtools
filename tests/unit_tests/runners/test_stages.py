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


def test_obdeect_stages_are_registered_lazily():
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


def teststage_catalog_declares_consistent_stages():
    catalog = stages.stage_catalog()

    assert {"corsika", "sim_telarray"} <= set(catalog)
    for name, entry in catalog.items():
        stage_class = stages.get_stage_class(name)
        assert stage_class.name == name
        assert entry["software"]
        assert isinstance(stage_class.produces, str)


def _fake_converter(name, consumes, produces):
    class FakeConverter:
        streams = True

    FakeConverter.name = name
    FakeConverter.consumes = consumes
    FakeConverter.produces = produces
    stages.register_stage(name, lambda: FakeConverter, converter=True)


@pytest.fixture
def fake_converters():
    names = []

    def register(name, consumes, produces):
        names.append(name)
        _fake_converter(name, consumes, produces)

    yield register
    for name in names:
        stages.STAGES.pop(name)
        stages.STAGE_SOFTWARE.pop(name)
        stages.STAGE_CONVERTERS.discard(name)


def test_catalog_declares_converters():
    assert {"corsika_to_stream", "stream_to_simtel"} <= stages.STAGE_CONVERTERS
    assert not {"corsika", "sim_telarray", "obdeect_optics"} & stages.STAGE_CONVERTERS


def test_insert_converters_bridges_neighbouring_stages():
    resolved = stages.insert_converters(
        ["corsika", "obdeect_atmosphere", "obdeect_optics", "sim_telarray"]
    )

    assert resolved == (
        "corsika",
        "corsika_to_stream",
        "obdeect_atmosphere",
        "obdeect_optics",
        "stream_to_simtel",
        "sim_telarray",
    )
    assert stages.SimulationChain(resolved).stages == resolved


@pytest.mark.parametrize(
    "names",
    [
        ("corsika",),
        ("corsika", "sim_telarray"),
        ("corsika", "corsika_to_stream", "obdeect_atmosphere"),
        ("obdeect_flasher", "stream_to_simtel", "sim_telarray"),
    ],
)
def test_insert_converters_keeps_matching_and_explicit_chains(names):
    assert stages.insert_converters(names) == names


def test_insert_converters_for_a_source_stage():
    assert stages.insert_converters(["obdeect_flasher", "sim_telarray"]) == (
        "obdeect_flasher",
        "stream_to_simtel",
        "sim_telarray",
    )


def test_insert_converters_leaves_a_mismatch_without_converter_to_the_validation():
    names = stages.insert_converters(["sim_telarray", "corsika"])

    assert names == ("sim_telarray", "corsika")
    with pytest.raises(ValueError, match="'corsika' consumes None"):
        stages.SimulationChain(names)


def test_insert_converters_uses_several_converters(fake_converters):
    fake_converters("first_step", "product_a", "product_b")
    fake_converters("second_step", "product_b", "product_c")
    stages.register_stage("source_a", lambda: _source("product_a"))
    stages.register_stage("sink_c", lambda: _sink("product_c"))
    try:
        assert stages.insert_converters(["source_a", "sink_c"]) == (
            "source_a",
            "first_step",
            "second_step",
            "sink_c",
        )
    finally:
        for name in ("source_a", "sink_c"):
            stages.STAGES.pop(name)
            stages.STAGE_SOFTWARE.pop(name)


def test_insert_converters_rejects_ambiguous_paths(fake_converters):
    fake_converters("route_one", "product_a", "product_c")
    fake_converters("route_two", "product_a", "product_c")
    stages.register_stage("source_a", lambda: _source("product_a"))
    stages.register_stage("sink_c", lambda: _sink("product_c"))
    try:
        with pytest.raises(ValueError, match=r"Several converters connect 'source_a' and 'sink_c'"):
            stages.insert_converters(["source_a", "sink_c"])
    finally:
        for name in ("source_a", "sink_c"):
            stages.STAGES.pop(name)
            stages.STAGE_SOFTWARE.pop(name)


def _source(produces):
    class Source:
        name = "source_a"
        consumes = None
        streams = True

    Source.produces = produces
    return Source


def _sink(consumes):
    class Sink:
        name = "sink_c"
        produces = stages.CAMERA_EVENTS
        streams = True

    Sink.consumes = consumes
    return Sink
