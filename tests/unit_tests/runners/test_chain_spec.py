"""Tests for the chain specification."""

import subprocess
import sys

import pytest

from simtools.runners import chain_spec, stages

PHOTONS_CAMERA = "photons_camera"

OBDEECT_CHAIN = (
    "corsika",
    "corsika_to_stream",
    "obdeect_atmosphere",
    "obdeect_optics",
    "stream_to_simtel",
    "sim_telarray",
)


def _register(name, consumes, produces, applies=(), **roles):
    class FakeStage:
        streams = True

        @classmethod
        def applied_factors(cls, options, upstream=frozenset()):
            return frozenset(applies) | frozenset(options.get("extra", ()))

    FakeStage.name = name
    FakeStage.consumes = consumes
    FakeStage.produces = produces
    for role, value in roles.items():
        setattr(FakeStage, role, value)
    stages.register_stage(FakeStage)
    return FakeStage


@pytest.fixture
def fake_stages():
    names = []

    def register(name, consumes, produces, applies=(), **roles):
        names.append(name)
        return _register(name, consumes, produces, applies, **roles)

    yield register
    for name in names:
        stages.STAGES.pop(name)


def test_parse_chain_of_stage_names():
    spec = chain_spec.parse_chain(["corsika", "sim_telarray"])

    assert spec.stages == ("corsika", "sim_telarray")
    assert spec.options == {}
    assert spec.chain() == stages.SimulationChain(("corsika", "sim_telarray"))


@pytest.mark.parametrize("single", ["corsika", ["corsika"], ("corsika",)])
def test_parse_chain_of_one_stage(single):
    assert chain_spec.parse_chain(single).stages == ("corsika",)


def test_parse_chain_options_are_yaml_scalars(fake_stages):
    fake_stages("camera", "cherenkov_photons", "camera_events")
    spec = chain_spec.parse_chain(["corsika", " camera : level=3 , check=true , name=abc "])

    assert spec.stages == ("corsika", "camera")
    assert spec.stage_options("camera") == {"level": 3, "check": True, "name": "abc"}
    assert spec.stage_options("corsika") == {}


@pytest.mark.parametrize(
    ("item", "match"),
    [
        ("sim_telarray:bypass=1", r"Unknown option\(s\) \['bypass'\] of stage 'sim_telarray'"),
        ("sim_telarray:bypass_optics=5", r"Invalid bypass_optics option .* choose from"),
        ("sim_telarray:bypass_optics=yes", r"'bypass_optics' of stage 'sim_telarray' must be"),
        ("obdeect_atmosphere:wavelength_range_nm=[300]", r"of number with 2 values"),
    ],
)
def test_parse_chain_checks_options_against_the_catalog(item, match):
    with pytest.raises(ValueError, match=match):
        chain_spec.parse_chain(["corsika", item])


def test_parse_chain_options_keep_commas_inside_lists():
    spec = chain_spec.parse_chain(
        [
            "obdeect_atmosphere:wavelength_range_nm=[300,700],seed=3",
            "stream_to_simtel:allow_pending=[atmosphere, mirror],scaled_factors=[qe]",
        ]
    )

    assert spec.stage_options("obdeect_atmosphere") == {
        "wavelength_range_nm": [300, 700],
        "seed": 3,
    }
    assert spec.stage_options("stream_to_simtel") == {
        "allow_pending": ["atmosphere", "mirror"],
        "scaled_factors": ["qe"],
    }


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
        chain_spec.parse_chain(items)


def test_software_is_known_without_loading_stage_classes():
    spec = chain_spec.parse_chain(OBDEECT_CHAIN)

    assert spec.software == {"corsika", "obdeect", "sim_telarray"}
    with pytest.raises(ValueError, match="Unknown simulation stage 'nothing'"):
        chain_spec.parse_chain(["corsika", "nothing"]).software


def test_obdeect_chain_with_camera_only_sim_telarray_is_valid():
    items = [*OBDEECT_CHAIN[:-1], "sim_telarray:bypass_optics=3"]

    spec = chain_spec.parse_chain(items)

    assert spec.chain().stages == OBDEECT_CHAIN
    assert spec.stage_options("sim_telarray") == {"bypass_optics": 3}


@pytest.mark.parametrize("mode", [0])
def test_obdeect_chain_with_sim_telarray_optics_applies_a_factor_twice(mode):
    spec = chain_spec.parse_chain([*OBDEECT_CHAIN[:-1], f"sim_telarray:bypass_optics={mode}"])

    with pytest.raises(ValueError, match="twice"):
        spec.chain()


def test_chain_rejects_unknown_stage_and_mismatched_products():
    with pytest.raises(ValueError, match="Unknown simulation stage 'nothing'"):
        chain_spec.parse_chain(["corsika", "nothing"]).chain()
    with pytest.raises(ValueError, match="consumes"):
        chain_spec.parse_chain(["sim_telarray", "corsika"]).chain()


def test_chain_rejects_factor_applied_twice(fake_stages):
    fake_stages("optics_a", "cherenkov_photons", "optics_a", {"mirror_reflectivity"})
    fake_stages("optics_b", "optics_a", "camera_events", {"mirror_reflectivity", "qe"})
    spec = chain_spec.parse_chain(["corsika", "optics_a", "optics_b"])

    with pytest.raises(
        ValueError, match=r"applies 'mirror_reflectivity' twice.*optics_a.*optics_b"
    ):
        spec.chain()


def test_applied_factors_depend_on_stage_options(fake_stages):
    fake_stages("optics_a", "cherenkov_photons", "optics_a", {"mirror_reflectivity"})
    fake_stages("camera", "optics_a", "camera_events")
    spec = chain_spec.parse_chain(["corsika", "optics_a", "camera:extra=[mirror_reflectivity]"])

    assert dict(spec.applied_factors())["camera"] == {"mirror_reflectivity"}
    with pytest.raises(ValueError, match="twice"):
        spec.chain()
    assert chain_spec.parse_chain(["corsika", "optics_a", "camera"]).chain().has("camera")


@pytest.mark.parametrize(
    "items",
    [
        ["corsika", "sim_telarray"],
        [*OBDEECT_CHAIN[:-1], "sim_telarray:bypass_optics=3"],
    ],
)
def test_chains_load_no_obdeect_module(items):
    code = (
        "import sys\n"
        "from simtools.runners import chain_spec\n"
        f"chain_spec.parse_chain({items!r}).chain()\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'obdeect')\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_chain_keeps_explicit_converters():
    spec = chain_spec.parse_chain([*OBDEECT_CHAIN[:-1], "sim_telarray:bypass_optics=3"])

    assert spec.stages == OBDEECT_CHAIN
    assert spec.chain().stages == OBDEECT_CHAIN
    assert spec.stage_options("sim_telarray") == {"bypass_optics": 3}


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (
            ["corsika", "obdeect_atmosphere", "obdeect_optics", "sim_telarray:bypass_optics=3"],
            OBDEECT_CHAIN,
        ),
        (
            [
                "corsika",
                "obdeect_atmosphere",
                "obdeect_optics",
                "stream_to_simtel",
                "sim_telarray:bypass_optics=3",
            ],
            OBDEECT_CHAIN,
        ),
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
            ("obdeect_illuminator", *OBDEECT_CHAIN[2:]),
        ),
        (["corsika", "sim_telarray"], ("corsika", "sim_telarray")),
    ],
)
def test_parse_chain_inserts_missing_converters(items, expected):
    spec = chain_spec.parse_chain(items)

    assert spec.stages == expected
    assert spec.chain().stages == expected


def test_explicit_converter_keeps_its_options():
    spec = chain_spec.parse_chain(
        ["obdeect_flasher", "stream_to_simtel:allow_pending=[atmosphere]", "sim_telarray"]
    )

    assert spec.stages == ("obdeect_flasher", "stream_to_simtel", "sim_telarray")
    assert spec.stage_options("stream_to_simtel") == {"allow_pending": ["atmosphere"]}


def test_software_includes_the_programs_of_inserted_converters(fake_stages):
    fake_stages("camera_source", None, PHOTONS_CAMERA)

    spec = chain_spec.parse_chain(["camera_source", "sim_telarray"])

    assert spec.stages == ("camera_source", "stream_to_simtel", "sim_telarray")
    assert spec.software == {"obdeect", "sim_telarray"}


def test_parse_chain_rejects_several_fitting_converters(fake_stages):
    fake_stages("source", None, "raw")
    fake_stages("convert_a", "raw", "cherenkov_photons", converter=True)
    fake_stages("convert_b", "raw", "cherenkov_photons", converter=True)

    with pytest.raises(ValueError, match=r"Several converters fit .*convert_a.*convert_b"):
        chain_spec.parse_chain(["source", "sim_telarray"])


def test_parse_chain_rejects_converter_listed_elsewhere():
    with pytest.raises(ValueError, match="'corsika_to_stream' is needed between"):
        chain_spec.parse_chain(
            ["corsika", "obdeect_atmosphere", "obdeect_optics", "sim_telarray", "corsika_to_stream"]
        )


def test_chain_without_fitting_converter_is_rejected():
    spec = chain_spec.parse_chain(["obdeect_atmosphere", "corsika"])

    with pytest.raises(ValueError, match="consumes"):
        spec.chain()


@pytest.mark.parametrize(
    "items",
    [
        ["obdeect_flasher", "sim_telarray:bypass_optics=0"],
        ["corsika", "sim_telarray:bypass_optics=1"],
    ],
)
def test_rejects_incompatible_photon_frame(items):
    with pytest.raises(ValueError, match=r"photon.*bypass_optics"):
        chain_spec.parse_chain(items).chain()
