"""Tests for simulation stages and chains."""

from types import SimpleNamespace

import pytest

from simtools.runners import stages
from simtools.runners.corsika_stage import CorsikaStage
from simtools.runners.sim_telarray_stage import SimTelArrayStage
from simtools.runners.stages import StageInputs, StreamTo

CAMERA_EVENTS = "camera_events"
CHERENKOV_PHOTONS = "cherenkov_photons"


@pytest.fixture
def fake_stage():
    """Register a fake stage class and remove it afterwards."""
    registered = []

    def register(name, consumes=CHERENKOV_PHOTONS, produces=CAMERA_EVENTS, **roles):
        stage_class = type(
            "FakeStage",
            (),
            {"name": name, "consumes": consumes, "produces": produces, "streams": True, **roles},
        )
        stages.register_stage(stage_class)
        registered.append(name)
        return stage_class

    yield register
    for name in registered:
        stages.STAGES.pop(name, None)


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
    assert chain.generates_air_showers == (names[0] == "corsika")
    assert chain.reads_input_file == (names[0] != "corsika")
    assert all(chain.has(name) for name in names)


def test_catalog_stage_classes_carry_the_catalog_entry():
    corsika = stages.get_stage_class("corsika")

    assert issubclass(corsika, CorsikaStage)
    assert corsika.name == "corsika"
    assert corsika.consumes is None
    assert corsika.produces == CHERENKOV_PHOTONS
    assert corsika.spec.software == {"corsika"}
    assert stages.get_stage_class("corsika") is corsika
    assert CHERENKOV_PHOTONS in stages.get_stage_class("sim_telarray").consumes


def test_one_implementation_serves_several_stages():
    flasher = stages.get_stage_class("obdeect_flasher")
    illuminator = stages.get_stage_class("obdeect_illuminator")

    assert flasher.__mro__[1] is illuminator.__mro__[1]
    assert (flasher.spec.light_source, illuminator.spec.light_source) == ("flasher", "illuminator")
    assert flasher.produces != illuminator.produces


def test_chain_runs_implementation():
    chain = stages.SimulationChain(("corsika", "sim_telarray"))

    assert chain.runs(CorsikaStage)
    assert chain.runs(SimTelArrayStage)
    assert not stages.SimulationChain(("sim_telarray",)).runs(CorsikaStage)


def test_register_stage_extends_chains(fake_stage):
    stage_class = fake_stage("fake_stage")

    chain = stages.SimulationChain(("corsika", "fake_stage"))

    assert chain.stage_classes[1] is stage_class
    assert chain.software == {"corsika"}


def test_chain_rejects_unknown_stage_and_empty_chain():
    with pytest.raises(ValueError, match="Unknown simulation stage 'unknown'"):
        stages.SimulationChain(("unknown",))
    with pytest.raises(ValueError, match="at least one stage"):
        stages.SimulationChain(())


def test_chain_rejects_mismatched_data_products():
    with pytest.raises(ValueError, match="'corsika' consumes None, but 'sim_telarray' produces"):
        stages.SimulationChain(("sim_telarray", "corsika"))


def test_stage_may_consume_several_products(fake_stage):
    fake_stage("fake_stage", consumes=(CHERENKOV_PHOTONS, "other_photons"))

    assert stages.SimulationChain(("corsika", "fake_stage")).stages == ("corsika", "fake_stage")
    with pytest.raises(ValueError, match="consumes"):
        stages.SimulationChain(("fake_stage", "fake_stage"))


def test_registry_holds_the_catalog_stages():
    assert {
        "corsika",
        "sim_telarray",
        "light_emission",
        "corsika_to_stream",
        "stream_to_simtel",
        "obdeect_atmosphere",
        "obdeect_optics",
        "obdeect_flasher",
        "obdeect_illuminator",
    } <= set(stages.registry())


def test_stage_inputs_options_of_stage():
    inputs = StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        chain_options={"obdeect_optics": {"threads": 2}},
    )

    assert inputs.options_of("obdeect_optics") == {"threads": 2}
    assert inputs.options_of("sim_telarray") == {}


def test_stage_roles_come_from_catalog_or_class(fake_stage):
    fake_stage("fake_thinning", thins_photons=True, software=frozenset({"fake_program"}))

    assert stages.stage_role(stages.get_stage_class("obdeect_atmosphere"), "thins_photons")
    assert stages.stage_role(stages.get_stage_class("corsika"), "thins_photons", False) is False
    assert stages.stages_with(
        ("corsika", "obdeect_atmosphere", "fake_thinning"), "thins_photons"
    ) == [
        "obdeect_atmosphere",
        "fake_thinning",
    ]
    assert stages.stages_software(["corsika", "fake_thinning"]) == {"corsika", "fake_program"}
    chain = stages.SimulationChain(("light_emission", "sim_telarray"))
    assert chain.source == stages.LIGHT_SOURCE
    assert chain.with_role("source", stages.LIGHT_SOURCE) == ["light_emission"]


def test_converter_stages_are_declared_in_the_catalog():
    assert set(stages.converter_stages()) == {"corsika_to_stream", "stream_to_simtel"}


def test_stages_software_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown simulation stage 'unknown'"):
        stages.stages_software(["corsika", "unknown"])


def test_applied_factors_from_catalog_and_class_hook(fake_stage):
    fake_stage("fake_hook", applied_factors=classmethod(lambda cls, options, upstream: {"x"}))

    assert stages.applied_factors_of("obdeect_atmosphere", {}) == {"atmospheric_extinction"}
    assert stages.applied_factors_of("sim_telarray", {"bypass_optics": 1}) == {
        "atmospheric_extinction",
        "quantum_efficiency",
    }
    assert stages.applied_factors_of(
        "sim_telarray", {"bypass_optics": 1}, upstream=frozenset({"atmospheric_extinction"})
    ) == {"quantum_efficiency"}
    assert stages.applied_factors_of("fake_hook", {}) == {"x"}
    fake_stage("fake_plain")
    assert stages.applied_factors_of("fake_plain", {}) == frozenset()


def test_stage_file_and_single_downstream():
    files = SimpleNamespace(
        get_stage_file_name=lambda name, file_type, run_number: (name, file_type, run_number)
    )
    stage = SimpleNamespace(
        name="obdeect_optics", spec=stages.get_stage_class("obdeect_optics").spec, run_files=files
    )

    assert stages.stage_file(stage, "log", 4) == ("obdeect_optics", "stage_log", 4)
    assert stages.single_downstream(StreamTo(("next",)), "obdeect_optics") == "next"
    with pytest.raises(ValueError, match="pipes into one downstream command, got 2"):
        stages.single_downstream(StreamTo(("a", "b")), "obdeect_optics")


def test_run_files_of_light_emission_and_shower_runs(run_context):
    calibration = SimpleNamespace(runner_service="light-emission files")
    light = StageInputs((), (), (), {}, context=run_context, calibration=calibration)
    assert stages.files_of_run(light) == "light-emission files"
    with pytest.raises(ValueError, match="need a light-emission or a shower configuration"):
        stages.files_of_run(StageInputs((), (), (), {}, stages=("corsika",)))


def test_ends_camera_only():
    def _inputs(names, options):
        return StageInputs((), (), (), {}, chain_options=options, stages=names)

    assert stages.ends_camera_only(
        _inputs(("sim_telarray",), {"sim_telarray": {"bypass_optics": 3}})
    )
    assert not stages.ends_camera_only(_inputs(("sim_telarray",), {}))
    assert not stages.ends_camera_only(_inputs(("corsika",), {}))
    assert not stages.ends_camera_only(_inputs((), {}))


def test_chain_envelope_enabled_follows_atmosphere_stage(run_context):
    def _inputs(names, options=None):
        return StageInputs(
            array_models=(),
            simulation_parameters=(),
            shower_configurations=(),
            args={},
            context=run_context,
            chain_options=options or {},
            stages=names,
        )

    assert stages.efficiency_envelope_enabled(_inputs(("corsika", "sim_telarray"))) is False
    assert stages.efficiency_envelope_enabled(_inputs(("obdeect_atmosphere",))) is False
    assert (
        stages.efficiency_envelope_enabled(
            _inputs(
                ("obdeect_atmosphere",),
                {"obdeect_atmosphere": {"efficiency_envelope": "qe"}},
            )
        )
        is True
    )
    assert (
        stages.efficiency_envelope_enabled(
            _inputs(
                ("obdeect_atmosphere",),
                {"obdeect_atmosphere": {"efficiency_envelope": "none"}},
            )
        )
        is False
    )


def _chain(names, chain_options=None, context=None):
    return StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        context=context,
        chain_options=chain_options or {},
        stages=names,
    )


@pytest.mark.parametrize(
    ("names", "chain_options", "expected"),
    [
        # default: on for a chain ending in sim_telarray with Bypass_Optics 3
        (("obdeect_atmosphere", "sim_telarray"), {"sim_telarray": {"bypass_optics": 3}}, True),
        # default: off for Bypass_Optics 1 or 2, and for a chain not ending in sim_telarray
        (("obdeect_atmosphere", "sim_telarray"), {"sim_telarray": {"bypass_optics": 1}}, False),
        (("obdeect_atmosphere", "sim_telarray"), {"sim_telarray": {"bypass_optics": 2}}, False),
        (("obdeect_atmosphere", "sim_telarray"), {}, False),
        (("obdeect_atmosphere", "stream_to_simtel"), {}, False),
        # explicit option overrides the default in both directions
        (
            ("obdeect_atmosphere", "sim_telarray"),
            {
                "obdeect_atmosphere": {"efficiency_envelope": "none"},
                "sim_telarray": {"bypass_optics": 3},
            },
            False,
        ),
        (
            ("obdeect_atmosphere", "sim_telarray"),
            {
                "obdeect_atmosphere": {"efficiency_envelope": "qe"},
                "sim_telarray": {"bypass_optics": 1},
            },
            True,
        ),
    ],
)
def test_envelope_default_depends_on_chain_end(names, chain_options, expected):
    inputs = _chain(names, chain_options)

    assert stages.get_stage_class("obdeect_atmosphere").efficiency_envelope(inputs) is expected
    assert stages.efficiency_envelope_enabled(inputs) is expected


def test_envelope_default_requires_sim_telarray_last():
    inputs = _chain(
        ("obdeect_atmosphere", "sim_telarray", "corsika_to_stream"),
        {"sim_telarray": {"bypass_optics": 3}},
    )

    assert stages.get_stage_class("obdeect_atmosphere").efficiency_envelope(inputs) is False
