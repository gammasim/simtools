"""Tests for simulation stage roles, stage inputs and the checks of stage arrays."""

from types import SimpleNamespace

import pytest

from simtools.runners import stages
from simtools.runners.chain_spec import parse_chain
from simtools.runners.corsika_stage import CorsikaStage
from simtools.runners.sim_telarray_stage import SimTelArrayStage
from simtools.runners.stages import CatalogStage, StageInputs, StreamTo

CAMERA_EVENTS = "camera_events"
CHERENKOV_PHOTONS = "cherenkov_photons"
OBDEECT_ITEMS = [
    "corsika",
    "obdeect_atmosphere",
    "obdeect_optics",
    "sim_telarray:bypass_optics=camera_only",
]


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


def test_catalog_stage_classes_carry_the_catalog_entry():
    corsika = stages.get_stage_class("corsika")

    assert issubclass(corsika, CorsikaStage)
    assert corsika.name == "corsika"
    assert corsika.consumes is None
    assert corsika.produces == CHERENKOV_PHOTONS
    assert corsika.spec.software == {"corsika"}
    assert stages.get_stage_class("corsika") is corsika
    assert CHERENKOV_PHOTONS in stages.get_stage_class("sim_telarray").consumes
    assert issubclass(stages.get_stage_class("sim_telarray"), SimTelArrayStage)


def test_one_implementation_serves_several_stages():
    flasher = stages.get_stage_class("obdeect_flasher")
    illuminator = stages.get_stage_class("obdeect_illuminator")

    assert flasher.__mro__[1] is illuminator.__mro__[1]
    assert (flasher.spec.light_source, illuminator.spec.light_source) == ("flasher", "illuminator")
    assert flasher.produces != illuminator.produces


def test_register_stage_makes_stage_available(fake_stage):
    stage_class = fake_stage("fake_stage")

    assert stages.get_stage_class("fake_stage") is stage_class


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


def test_catalog_stage_defaults():
    stage = CatalogStage()

    assert stage.environment() == {}
    assert stage.get_resources(runtime=3.0) == {}
    assert stage.get_resources() == {}


def test_stage_inputs_stages_and_options_of_chain():
    chain = parse_chain(OBDEECT_ITEMS)
    inputs = StageInputs((), (), (), {}, chain=chain)

    assert inputs.stages == chain.stages
    assert inputs.chain is chain
    assert inputs.options_of("sim_telarray") == {"bypass_optics": "camera_only"}
    assert inputs.options_of("obdeect_optics") == {"telescope_transmission": "full"}
    assert inputs.options_of("light_emission") == {}


def test_stage_inputs_without_chain():
    inputs = StageInputs((), (), (), {})

    assert inputs.stages == ()
    assert inputs.chain is None
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


def test_converter_stages_are_declared_in_the_catalog():
    assert set(stages.converter_stages()) == {"corsika_to_stream", "stream_to_simtel"}


def test_stages_software_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown simulation stage 'unknown'"):
        stages.stages_software(["corsika", "unknown"])


def test_applied_factors_from_catalog(fake_stage):
    assert stages.applied_factors_of("obdeect_atmosphere", {}) == {"atmospheric_extinction"}
    assert stages.applied_factors_of("sim_telarray", {"bypass_optics": "none"}) == {
        "atmospheric_extinction",
        "mirror_reflectivity",
        "telescope_transmission",
        "quantum_efficiency",
    }
    assert stages.applied_factors_of("sim_telarray", {"bypass_optics": "camera_entrance"}) == {
        "atmospheric_extinction",
        "quantum_efficiency",
    }
    assert stages.applied_factors_of(
        "sim_telarray",
        {"bypass_optics": "camera_entrance"},
        upstream=frozenset({"atmospheric_extinction"}),
    ) == {"quantum_efficiency"}


def test_applied_factors_of_stage_outside_the_catalog(fake_stage):
    fake_stage("fake_plain")

    assert stages.applied_factors_of("fake_plain", {}) == frozenset()


def test_stage_file_requires_run_configuration():
    stage = SimpleNamespace(
        name="obdeect_optics", spec=stages.get_stage_class("obdeect_optics").spec, run_files=None
    )

    with pytest.raises(ValueError, match="has no run configuration to name its log file"):
        stages.stage_file(stage, "log", 4)


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


def test_files_of_run_of_light_emission(run_context):
    calibration = SimpleNamespace(runner_service="light-emission files")
    light = StageInputs((), (), (), {}, context=run_context, calibration=calibration)

    assert stages.files_of_run(light) == "light-emission files"


def test_files_of_run_without_configuration():
    inputs = StageInputs((), (), (), {})

    with pytest.raises(ValueError, match="need a light-emission or a shower configuration"):
        stages.files_of_run(inputs)
    assert stages.files_of_run(inputs, required=False) is None


def test_ends_camera_only():
    def _inputs(items):
        return StageInputs((), (), (), {}, chain=parse_chain(items))

    assert stages.ends_camera_only(_inputs(OBDEECT_ITEMS))
    assert not stages.ends_camera_only(
        _inputs(
            [
                "corsika",
                "obdeect_atmosphere",
                "obdeect_optics",
                "sim_telarray:bypass_optics=camera_entrance",
            ]
        )
    )
    assert not stages.ends_camera_only(_inputs(["corsika", "sim_telarray"]))
    assert not stages.ends_camera_only(_inputs(["corsika"]))
    assert not stages.ends_camera_only(StageInputs((), (), (), {}))


def test_efficiency_envelope_enabled_follows_atmosphere_stage(run_context):
    def _inputs(items):
        return StageInputs(
            array_models=(),
            simulation_parameters=(),
            shower_configurations=(),
            args={},
            context=run_context,
            chain=parse_chain(items),
        )

    assert stages.efficiency_envelope_enabled(_inputs(["corsika", "sim_telarray"])) is False
    assert stages.efficiency_envelope_enabled(_inputs(["obdeect_atmosphere"])) is False
    assert (
        stages.efficiency_envelope_enabled(_inputs(["obdeect_atmosphere:efficiency_envelope=qe"]))
        is True
    )
    assert (
        stages.efficiency_envelope_enabled(_inputs(["obdeect_atmosphere:efficiency_envelope=none"]))
        is False
    )


def _chain(items):
    return StageInputs(
        array_models=(),
        simulation_parameters=(),
        shower_configurations=(),
        args={},
        chain=parse_chain(items),
    )


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        # default: on for a chain whose last stage simulates the camera only
        (OBDEECT_ITEMS, True),
        # default: off for a chain not ending in camera-only sim_telarray
        (
            [
                "corsika",
                "obdeect_atmosphere",
                "obdeect_optics",
                "sim_telarray:bypass_optics=camera_entrance",
            ],
            False,
        ),
        (["corsika", "obdeect_atmosphere", "obdeect_optics"], False),
        # explicit option overrides the default in both directions
        (
            [
                "corsika",
                "obdeect_atmosphere:efficiency_envelope=none",
                "obdeect_optics",
                "sim_telarray:bypass_optics=camera_only",
            ],
            False,
        ),
        (
            [
                "corsika",
                "obdeect_atmosphere:efficiency_envelope=qe",
                "obdeect_optics",
                "sim_telarray:bypass_optics=camera_entrance",
            ],
            True,
        ),
    ],
)
def test_envelope_default_depends_on_chain_end(items, expected):
    inputs = _chain(items)

    assert stages.get_stage_class("obdeect_atmosphere").efficiency_envelope(inputs) is expected
    assert stages.efficiency_envelope_enabled(inputs) is expected


def test_check_array_models_accepts_one_model_per_run():
    stages.check_array_models(parse_chain(OBDEECT_ITEMS).stages, ("model-6.0.2",))
    stages.check_array_models(("corsika", "sim_telarray"), ("model-6.0.2",))


def test_check_array_models_rejects_several_models_for_single_model_stages():
    with pytest.raises(
        ValueError,
        match=r"Stages \['obdeect_atmosphere', 'obdeect_optics', 'stream_to_simtel'\] "
        r"simulate one model version per run; got 2 model versions",
    ):
        stages.check_array_models(parse_chain(OBDEECT_ITEMS).stages, ("model-6.0.2", "model-6.0.3"))


def test_check_array_models_allows_several_models_without_single_model_stages():
    stages.check_array_models(("corsika", "sim_telarray"), ("model-6.0.2", "model-6.0.3"))
