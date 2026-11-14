"""Tests for the chain specification."""

import subprocess
import sys

import pytest

from simtools.runners import chain_spec, stages

OBDEECT_CHAIN = (
    "corsika",
    "corsika_to_stream",
    "obdeect_atmosphere",
    "obdeect_optics",
    "stream_to_simtel",
    "sim_telarray",
)
# The same chain without the converters, which are inserted.
SHORT_OBDEECT_CHAIN = ["corsika", "obdeect_atmosphere", "obdeect_optics"]


def _register(name, consumes, produces, applies=()):
    class FakeStage:
        streams = True

        @classmethod
        def applied_factors(cls, options):
            return frozenset(applies) | frozenset(options.get("extra", ()))

    FakeStage.name = name
    FakeStage.consumes = consumes
    FakeStage.produces = produces
    stages.register_stage(name, lambda: FakeStage)
    return FakeStage


@pytest.fixture
def fake_stages():
    names = []

    def register(name, consumes, produces, applies=()):
        names.append(name)
        return _register(name, consumes, produces, applies)

    yield register
    for name in names:
        stages.STAGES.pop(name)
        stages.STAGE_SOFTWARE.pop(name)


def test_parse_chain_of_stage_names():
    spec = chain_spec.parse_chain(["corsika", "sim_telarray"])

    assert spec.stages == ("corsika", "sim_telarray")
    assert spec.options == {}
    assert spec.chain() == stages.SimulationChain(("corsika", "sim_telarray"))


@pytest.mark.parametrize("single", ["corsika", ["corsika"], ("corsika",)])
def test_parse_chain_of_one_stage(single):
    assert chain_spec.parse_chain(single).stages == ("corsika",)


def test_parse_chain_options_are_yaml_scalars():
    spec = chain_spec.parse_chain(
        ["corsika", " sim_telarray : bypass_optics=3 , check=true , name=abc "]
    )

    assert spec.stages == ("corsika", "sim_telarray")
    assert spec.stage_options("sim_telarray") == {"bypass_optics": 3, "check": True, "name": "abc"}
    assert spec.stage_options("corsika") == {}


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


@pytest.mark.parametrize("mode", [0, 1, 2])
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
    fake_stages("optics_a", stages.CHERENKOV_PHOTONS, "optics_a", {"mirror_reflectivity"})
    fake_stages("optics_b", "optics_a", stages.CAMERA_EVENTS, {"mirror_reflectivity", "qe"})
    spec = chain_spec.parse_chain(["corsika", "optics_a", "optics_b"])

    with pytest.raises(
        ValueError, match=r"applies 'mirror_reflectivity' twice.*optics_a.*optics_b"
    ):
        spec.chain()


def test_applied_factors_depend_on_stage_options(fake_stages):
    fake_stages("optics_a", stages.CHERENKOV_PHOTONS, "optics_a", {"mirror_reflectivity"})
    fake_stages("camera", "optics_a", stages.CAMERA_EVENTS)
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
        [*SHORT_OBDEECT_CHAIN, "sim_telarray:bypass_optics=3"],
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


def test_resolved_stages_insert_the_converters():
    spec = chain_spec.parse_chain([*SHORT_OBDEECT_CHAIN, "sim_telarray:bypass_optics=3"])

    assert spec.stages == ("corsika", "obdeect_atmosphere", "obdeect_optics", "sim_telarray")
    assert spec.resolved_stages == OBDEECT_CHAIN
    assert spec.chain().stages == OBDEECT_CHAIN
    assert spec.stage_options("sim_telarray") == {"bypass_optics": 3}


def test_resolved_stages_of_a_chain_without_mismatch_are_the_listed_stages():
    spec = chain_spec.parse_chain(["corsika", "sim_telarray"])

    assert spec.resolved_stages == spec.stages == ("corsika", "sim_telarray")


def test_software_includes_the_programs_of_the_inserted_converters():
    spec = chain_spec.parse_chain(["obdeect_flasher", "sim_telarray:bypass_optics=3"])

    assert spec.resolved_stages == ("obdeect_flasher", "stream_to_simtel", "sim_telarray")
    assert spec.software == {"obdeect", "sim_telarray"}


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_short_obdeect_chain_with_sim_telarray_optics_applies_a_factor_twice(mode):
    spec = chain_spec.parse_chain([*SHORT_OBDEECT_CHAIN, f"sim_telarray:bypass_optics={mode}"])

    with pytest.raises(ValueError, match="twice"):
        spec.chain()
