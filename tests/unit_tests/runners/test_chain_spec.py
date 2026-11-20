"""Tests for simulation chains: parsing, validation, options and efficiency factors of the stages."""

import dataclasses
import subprocess
import sys

import pytest

from simtools.runners import stages
from simtools.runners.chain_spec import SimulationChain, parse_chain
from simtools.runners.corsika_stage import CorsikaStage
from simtools.runners.sim_telarray_stage import SimTelArrayStage
from simtools.runtime import software_catalog
from simtools.runtime.software_catalog import StageSpec

PHOTONS_CAMERA = "photons_camera"
CAMERA_EVENTS = "camera_events"
CHERENKOV_PHOTONS = "cherenkov_photons"
PHOTONS_GROUND = "photons_ground"
ALL_AIR_SHOWER_FACTORS = {
    "atmospheric_extinction",
    "mirror_reflectivity",
    "telescope_transmission",
    "quantum_efficiency",
}

OBDEECT_ITEMS = [
    "corsika",
    "obdeect_atmosphere",
    "obdeect_optics",
    "sim_telarray:bypass_optics=camera_only",
]
OBDEECT_STAGES = (
    "corsika",
    "corsika_to_stream",
    "obdeect_atmosphere",
    "obdeect_optics",
    "stream_to_simtel",
    "sim_telarray",
)


def _register(name, consumes, produces, applies=(), **entry):
    """Register a stage whose catalog entry is built here (not read from the catalog)."""
    if isinstance(consumes, str):
        consumes = [consumes]
    spec = StageSpec(
        name,
        {
            "software": "obdeect",
            "consumes": consumes,
            "produces": produces,
            "applies": list(applies),
            **entry,
        },
    )
    stage_class = type(
        name,
        (),
        {
            "name": name,
            "consumes": spec.consumes,
            "produces": produces,
            "streams": True,
            "spec": spec,
        },
    )
    stages.register_stage(stage_class)
    return stage_class


@pytest.fixture
def fake_stages():
    """Register fake stages for one test and remove them afterwards."""
    names = []

    def register(name, consumes, produces, applies=(), **entry):
        names.append(name)
        return _register(name, consumes, produces, applies, **entry)

    yield register
    for name in names:
        stages.STAGES.pop(name, None)


@pytest.mark.parametrize("single", ["corsika", ["corsika"], ("corsika",)])
def test_parse_chain_of_one_stage(single):
    assert parse_chain(single).stages == ("corsika",)


def test_parse_chain_of_stage_names_applies_defaults():
    chain = parse_chain(["corsika", "sim_telarray"])

    assert chain.stages == ("corsika", "sim_telarray")
    assert chain.stage_options("sim_telarray") == {"bypass_optics": "none"}
    assert chain.stage_options("corsika") == {}
    assert chain == SimulationChain(("corsika", "sim_telarray"))


def test_parse_chain_options_are_yaml_scalars():
    chain = parse_chain(["corsika", "obdeect_atmosphere: seed = 3 , mode = weighted"])

    assert chain.stage_options("obdeect_atmosphere") == {"seed": 3, "mode": "weighted"}


def test_parse_chain_options_keep_commas_inside_lists():
    chain = parse_chain(
        [
            "corsika",
            "obdeect_atmosphere:wavelength_range_nm=[300,700],seed=3",
            "obdeect_optics",
            "stream_to_simtel:allow_pending=[atmosphere, mirror],scaled_factors=[qe]",
            "sim_telarray:bypass_optics=camera_only",
        ]
    )

    assert chain.stage_options("obdeect_atmosphere") == {
        "mode": "thinned",
        "wavelength_range_nm": [300, 700],
        "seed": 3,
    }
    assert chain.stage_options("stream_to_simtel") == {
        "allow_pending": ["atmosphere", "mirror"],
        "scaled_factors": ["qe"],
    }


def test_parse_chain_wraps_single_list_option_value():
    chain = parse_chain(
        [*OBDEECT_ITEMS[:-1], "stream_to_simtel:allow_pending=atmosphere", OBDEECT_ITEMS[-1]]
    )

    assert chain.stage_options("stream_to_simtel") == {"allow_pending": ["atmosphere"]}


@pytest.mark.parametrize(
    ("item", "match"),
    [
        ("sim_telarray:bypass=1", r"Unknown option\(s\) \['bypass'\] of stage 'sim_telarray'"),
        (
            "sim_telarray:bypass_optics=5",
            r"Option 'bypass_optics' of stage 'sim_telarray' must be of type string; got 5",
        ),
        (
            "sim_telarray:bypass_optics=3",
            r"Option 'bypass_optics' of stage 'sim_telarray' must be of type string; got 3",
        ),
        (
            "sim_telarray:bypass_optics=yes",
            r"Option 'bypass_optics' of stage 'sim_telarray' must be of type string",
        ),
        (
            "sim_telarray:bypass_optics=sometimes",
            r"Invalid bypass_optics option of stage 'sim_telarray': 'sometimes'",
        ),
        (
            "obdeect_atmosphere:mode=fast",
            r"Invalid mode option of stage 'obdeect_atmosphere': 'fast'",
        ),
        (
            "obdeect_atmosphere:wavelength_range_nm=[300]",
            r"must be of type list of number with 2 values; got \[300\]",
        ),
    ],
)
def test_parse_chain_checks_options_against_the_catalog(item, match):
    with pytest.raises(ValueError, match=match):
        parse_chain(["corsika", item])


@pytest.mark.parametrize(
    ("items", "match"),
    [
        ([], "at least one stage"),
        (None, "at least one stage"),
        (["corsika", "corsika"], "'corsika' is listed twice"),
        ([":a=1"], "has no stage name"),
        (["  "], "has no stage name"),
        (["sim_telarray:novalue"], r"Option 'novalue' of chain item .* must be KEY=VALUE"),
        (["sim_telarray:=3"], r"must be KEY=VALUE"),
        (["sim_telarray:a=1,a=2"], r"Option 'a' is given twice"),
        ([3], "must be a string"),
    ],
)
def test_parse_chain_rejects_malformed_items(items, match):
    with pytest.raises(ValueError, match=match):
        parse_chain(items)


def test_parse_chain_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown simulation stage 'nothing'"):
        parse_chain(["corsika", "nothing"])


def test_simulation_chain_requires_stages_and_known_stages():
    with pytest.raises(ValueError, match="at least one stage"):
        SimulationChain(())
    with pytest.raises(ValueError, match="Unknown simulation stage 'unknown'"):
        SimulationChain(("unknown",))


def test_simulation_chain_is_frozen():
    chain = parse_chain(["corsika", "sim_telarray"])

    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field 'stages'"):
        chain.stages = ("corsika",)


def test_simulation_chain_is_checked_on_creation():
    with pytest.raises(
        ValueError,
        match=r"Stage 'corsika' consumes None, but 'sim_telarray' produces 'camera_events'",
    ):
        SimulationChain(("sim_telarray", "corsika"))


def test_chain_without_fitting_converter_is_rejected():
    with pytest.raises(ValueError, match="Stage 'corsika' consumes None"):
        parse_chain(["obdeect_atmosphere", "corsika"])


def test_software_of_chain():
    assert parse_chain(OBDEECT_ITEMS).software == frozenset({"corsika", "obdeect", "sim_telarray"})
    assert parse_chain(["corsika", "sim_telarray"]).software == frozenset(
        {"corsika", "sim_telarray"}
    )
    assert parse_chain(["light_emission", "sim_telarray"]).software == frozenset({"sim_telarray"})


def test_chain_stages_of_camera_only_obdeect_chain():
    chain = parse_chain(OBDEECT_ITEMS)

    assert chain.stages == OBDEECT_STAGES
    assert chain.stage_options("sim_telarray") == {"bypass_optics": "camera_only"}
    assert chain.stage_options("obdeect_atmosphere") == {"mode": "thinned"}
    assert chain.stage_options("obdeect_optics") == {"telescope_transmission": "full"}


def test_stage_options_of_stage_outside_the_chain_are_empty():
    assert parse_chain(["corsika", "sim_telarray"]).stage_options("obdeect_optics") == {}


def test_stage_classes_and_properties_of_air_shower_chain():
    chain = parse_chain(["corsika", "sim_telarray"])

    assert [stage.name for stage in chain.stage_classes] == ["corsika", "sim_telarray"]
    assert chain.source == software_catalog.AIR_SHOWER_SOURCE
    assert chain.generates_air_showers is True
    assert chain.reads_input_file is False
    assert chain.runs(CorsikaStage) is True
    assert chain.runs(SimTelArrayStage) is True


def test_chain_starting_with_input_file():
    chain = parse_chain(["sim_telarray"])

    assert chain.source is None
    assert chain.generates_air_showers is False
    assert chain.reads_input_file is True
    assert chain.runs(CorsikaStage) is False


def test_light_source_chain_properties():
    chain = parse_chain(["light_emission", "sim_telarray"])

    assert chain.source == software_catalog.LIGHT_SOURCE
    assert chain.generates_air_showers is False
    assert chain.reads_input_file is False
    assert chain.with_role("source", software_catalog.LIGHT_SOURCE) == ["light_emission"]


def test_with_role_returns_stages_of_the_chain():
    chain = parse_chain(OBDEECT_ITEMS)

    assert chain.with_role("single_array_model") == [
        "obdeect_atmosphere",
        "obdeect_optics",
        "stream_to_simtel",
    ]
    assert chain.with_role("thins_photons") == ["obdeect_atmosphere"]
    assert chain.with_role("converter") == ["corsika_to_stream", "stream_to_simtel"]


def test_upstream_stages():
    chain = parse_chain(OBDEECT_ITEMS)

    assert chain.upstream_stages("obdeect_optics") == OBDEECT_STAGES[:3]
    assert chain.upstream_stages("corsika") == ()


def test_chain_connections_depend_on_stage_options():
    with pytest.raises(
        ValueError,
        match=(
            r"Stage 'sim_telarray' with options \{'bypass_optics': 'none'\} consumes .*, "
            r"but 'stream_to_simtel' produces 'camera_photons_iact'; "
            r"set the options of 'sim_telarray'"
        ),
    ):
        parse_chain(["obdeect_flasher", "sim_telarray"])


def test_chain_connection_error_without_options():
    with pytest.raises(
        ValueError,
        match=r"Stage 'sim_telarray' consumes .*, but 'obdeect_atmosphere' produces 'photons_ground'",
    ):
        parse_chain(["corsika", "obdeect_atmosphere", "sim_telarray:bypass_optics=camera_only"])


def test_light_source_chain_connection_depends_on_stage_options():
    with pytest.raises(
        ValueError,
        match=r"but 'light_emission' produces 'light_emission_photons'; set the options of",
    ):
        parse_chain(["light_emission", "sim_telarray:bypass_optics=camera_only"])


def test_chain_ending_in_camera_events_applies_required_factors():
    with pytest.raises(
        ValueError,
        match=r"does not apply the efficiency factors \['atmospheric_extinction'\]",
    ):
        parse_chain(["corsika", "obdeect_optics", "sim_telarray:bypass_optics=camera_only"])
    with pytest.raises(
        ValueError,
        match=(
            r"does not apply the efficiency factors \['atmospheric_extinction', "
            r"'mirror_reflectivity', 'telescope_transmission'\]"
        ),
    ):
        parse_chain(["sim_telarray:bypass_optics=camera_only"])


def test_light_source_chain_needs_only_its_required_factors():
    assert parse_chain(["light_emission", "sim_telarray"]).stages == (
        "light_emission",
        "sim_telarray",
    )


def test_chain_not_ending_in_camera_events_needs_no_required_factors():
    assert parse_chain(["corsika", "obdeect_optics"]).stages == (
        "corsika",
        "corsika_to_stream",
        "obdeect_optics",
    )


def test_chain_rejects_factor_applied_twice(fake_stages):
    fake_stages("fake_a", CHERENKOV_PHOTONS, PHOTONS_GROUND, {"mirror_reflectivity"})
    fake_stages("fake_b", PHOTONS_GROUND, CAMERA_EVENTS, {"mirror_reflectivity"})

    with pytest.raises(
        ValueError,
        match=r"applies 'mirror_reflectivity' twice: in stage 'fake_a' and in stage 'fake_b'",
    ):
        parse_chain(["corsika", "fake_a", "fake_b"])


def test_registered_stage_extends_chains(fake_stages):
    fake_stages("fake_stage", CHERENKOV_PHOTONS, CAMERA_EVENTS, ALL_AIR_SHOWER_FACTORS)

    chain = parse_chain(["corsika", "fake_stage"])

    assert chain.stages == ("corsika", "fake_stage")
    assert chain.software == frozenset({"corsika", "obdeect"})
    assert dict(chain.applied_factors())["fake_stage"] == ALL_AIR_SHOWER_FACTORS


def test_applied_factors_per_stage():
    assert parse_chain(OBDEECT_ITEMS).applied_factors() == [
        ("corsika", frozenset()),
        ("corsika_to_stream", frozenset()),
        ("obdeect_atmosphere", {"atmospheric_extinction"}),
        ("obdeect_optics", {"mirror_reflectivity", "telescope_transmission"}),
        ("stream_to_simtel", frozenset()),
        ("sim_telarray", {"quantum_efficiency"}),
    ]


def test_applied_factors_of_sim_telarray_with_optics():
    chain = parse_chain(["corsika", "sim_telarray"])

    assert dict(chain.applied_factors())["sim_telarray"] == ALL_AIR_SHOWER_FACTORS
    assert chain.skipped_factors("sim_telarray") == frozenset()


def test_camera_entrance_skips_extinction_applied_upstream():
    chain = parse_chain(
        [
            "corsika",
            "obdeect_atmosphere",
            "obdeect_optics",
            "sim_telarray:bypass_optics=camera_entrance",
        ]
    )

    assert chain.skipped_factors("sim_telarray") == {"atmospheric_extinction"}
    assert dict(chain.applied_factors())["sim_telarray"] == {"quantum_efficiency"}
    assert chain.upstream_factors("sim_telarray") == {
        "atmospheric_extinction",
        "mirror_reflectivity",
        "telescope_transmission",
    }
    assert chain.upstream_factors("obdeect_atmosphere") == frozenset()
    assert chain.upstream_factors(None) == ALL_AIR_SHOWER_FACTORS


def test_stage_outside_the_catalog_applies_and_skips_no_factors():
    class PlainStage:
        name = "plain_source"
        consumes = None
        produces = PHOTONS_GROUND
        streams = True

    stages.register_stage(PlainStage)
    try:
        chain = parse_chain(["plain_source", "obdeect_optics"])

        assert chain.skipped_factors("plain_source") == frozenset()
        assert chain.applied_factors()[0] == ("plain_source", frozenset())
    finally:
        stages.STAGES.pop("plain_source", None)


def test_chain_ending_in_product_outside_the_catalog_needs_no_factors(fake_stages):
    fake_stages("fake_end", CHERENKOV_PHOTONS, "unknown_product")

    assert parse_chain(["corsika", "fake_end"]).stages == ("corsika", "fake_end")


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (OBDEECT_ITEMS, OBDEECT_STAGES),
        (
            [
                "corsika",
                "obdeect_atmosphere",
                "obdeect_optics",
                "stream_to_simtel",
                "sim_telarray:bypass_optics=camera_only",
            ],
            OBDEECT_STAGES,
        ),
        (
            ["obdeect_flasher", "sim_telarray:bypass_optics=camera_entrance"],
            ("obdeect_flasher", "stream_to_simtel", "sim_telarray"),
        ),
        (
            [
                "obdeect_illuminator",
                "obdeect_atmosphere",
                "obdeect_optics",
                "sim_telarray:bypass_optics=camera_only",
            ],
            ("obdeect_illuminator", *OBDEECT_STAGES[2:]),
        ),
        (["corsika", "sim_telarray"], ("corsika", "sim_telarray")),
    ],
)
def test_parse_chain_inserts_missing_converters(items, expected):
    assert parse_chain(items).stages == expected


def test_explicit_converter_keeps_its_options():
    chain = parse_chain(
        [
            "obdeect_flasher",
            "stream_to_simtel:allow_pending=[atmosphere]",
            "sim_telarray:bypass_optics=camera_entrance",
        ]
    )

    assert chain.stages == ("obdeect_flasher", "stream_to_simtel", "sim_telarray")
    assert chain.stage_options("stream_to_simtel") == {"allow_pending": ["atmosphere"]}


def test_inserted_converter_software_is_part_of_the_chain(fake_stages):
    fake_stages("camera_source", None, PHOTONS_CAMERA, source=software_catalog.LIGHT_SOURCE)

    chain = parse_chain(["camera_source", "sim_telarray:bypass_optics=camera_entrance"])

    assert chain.stages == ("camera_source", "stream_to_simtel", "sim_telarray")
    assert chain.software == frozenset({"obdeect", "sim_telarray"})


def test_parse_chain_rejects_several_fitting_converters(fake_stages):
    fake_stages("source", None, "raw")
    fake_stages("convert_a", "raw", CHERENKOV_PHOTONS, converter=True)
    fake_stages("convert_b", "raw", CHERENKOV_PHOTONS, converter=True)

    with pytest.raises(ValueError, match=r"Several converters fit .*convert_a.*convert_b"):
        parse_chain(["source", "sim_telarray"])


def test_parse_chain_rejects_converter_listed_elsewhere():
    with pytest.raises(ValueError, match="'corsika_to_stream' is needed between"):
        parse_chain(
            [
                "corsika",
                "obdeect_atmosphere",
                "obdeect_optics",
                "sim_telarray:bypass_optics=camera_only",
                "corsika_to_stream",
            ]
        )


@pytest.mark.parametrize(
    "items",
    [
        ["obdeect_flasher", "sim_telarray:bypass_optics=none"],
        ["corsika", "sim_telarray:bypass_optics=camera_only"],
    ],
)
def test_rejects_incompatible_photon_frame(items):
    with pytest.raises(ValueError, match=r"consumes .*, but .* produces"):
        parse_chain(items)


def test_chain_loads_no_obdeect_module():
    code = (
        "import sys\n"
        "from simtools.runners import chain_spec\n"
        "chain_spec.parse_chain(['corsika', 'sim_telarray'])\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'obdeect')\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
