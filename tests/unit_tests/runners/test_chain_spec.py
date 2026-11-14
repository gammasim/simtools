"""Tests for chain specifications and the guarantees of the legacy chain."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from simtools.runners import chain_spec, stages


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


def test_legacy_preset_is_corsika_into_sim_telarray():
    spec = chain_spec.load_chain_spec()

    assert spec.name == chain_spec.LEGACY_PRESET
    assert spec.stages == ("corsika", "sim_telarray")
    assert spec.chain() == stages.SimulationChain.from_software("corsika_sim_telarray")
    assert chain_spec.load_chain_spec("legacy") is spec


def test_legacy_modules_do_not_load_obdeect():
    code = (
        "import sys\n"
        "from simtools.runners import chain_spec, simulator, sim_telarray_stage, corsika_stage\n"
        "from simtools.simtel import simulator_light_emission, simulator_array\n"
        "chain_spec.load_chain_spec().chain()\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'obdeect')\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_register_preset_and_load(fake_stages):
    fake_stages("adapter", stages.CHERENKOV_PHOTONS, "adapted")
    fake_stages("camera", "adapted", stages.CAMERA_EVENTS)
    chain_spec.register_preset(
        "test_preset", ("corsika", "adapter", "camera"), {"camera": {"a": 1}}
    )
    try:
        spec = chain_spec.load_chain_spec("test_preset")
        assert "test_preset" in chain_spec.preset_names()
        assert spec.chain().stages == ("corsika", "adapter", "camera")
        assert spec.stage_options("camera") == {"a": 1}
        assert spec.stage_options("adapter") == {}
    finally:
        chain_spec._PRESETS.pop("test_preset")


def test_load_chain_spec_from_file(tmp_test_directory, fake_stages):
    fake_stages("adapter", stages.CHERENKOV_PHOTONS, "adapted")
    fake_stages("camera", "adapted", stages.CAMERA_EVENTS)
    path = Path(tmp_test_directory) / "chain.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {"stage": "corsika"},
                    {"stage": "adapter", "options": {"threads": 2}},
                    {"stage": "camera"},
                ]
            }
        ),
        encoding="utf-8",
    )

    spec = chain_spec.load_chain_spec(path)

    assert spec.stages == ("corsika", "adapter", "camera")
    assert spec.stage_options("adapter") == {"threads": 2}
    assert spec.chain().stages == spec.stages


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ({"stages": []}, "non-empty 'stages' list"),
        ({}, "non-empty 'stages' list"),
        ({"stages": [{"options": {}}]}, "needs a 'stage' name"),
        ({"stages": ["corsika"]}, "needs a 'stage' name"),
    ],
)
def test_load_chain_spec_rejects_malformed_file(tmp_test_directory, content, match):
    path = Path(tmp_test_directory) / "bad.yml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        chain_spec.load_chain_spec(path)


def test_load_chain_spec_rejects_unknown_selection():
    with pytest.raises(ValueError, match=r"Unknown simulation chain 'nothing'.*legacy"):
        chain_spec.load_chain_spec("nothing")


def test_chain_rejects_unknown_stage_and_mismatched_products():
    with pytest.raises(ValueError, match="Unknown simulation stage 'nothing'"):
        chain_spec.ChainSpec("x", ("corsika", "nothing")).chain()
    with pytest.raises(ValueError, match="consumes"):
        chain_spec.ChainSpec("x", ("sim_telarray", "corsika")).chain()


def test_chain_rejects_factor_applied_twice(fake_stages):
    fake_stages("optics_a", stages.CHERENKOV_PHOTONS, "optics_a", {"mirror_reflectivity"})
    fake_stages("optics_b", "optics_a", stages.CAMERA_EVENTS, {"mirror_reflectivity", "qe"})
    spec = chain_spec.ChainSpec("twice", ("corsika", "optics_a", "optics_b"))

    with pytest.raises(
        ValueError, match=r"applies 'mirror_reflectivity' twice.*optics_a.*optics_b"
    ):
        spec.chain()


def test_applied_factors_depend_on_stage_options(fake_stages):
    fake_stages("optics_a", stages.CHERENKOV_PHOTONS, "optics_a", {"mirror_reflectivity"})
    fake_stages("camera", "optics_a", stages.CAMERA_EVENTS)
    spec = chain_spec.ChainSpec(
        "options",
        ("corsika", "optics_a", "camera"),
        {"camera": {"extra": ["mirror_reflectivity"]}},
    )

    assert dict(spec.applied_factors())["camera"] == {"mirror_reflectivity"}
    with pytest.raises(ValueError, match="twice"):
        spec.chain()
    assert chain_spec.ChainSpec("ok", spec.stages).chain().stages == spec.stages


@pytest.mark.parametrize("preset", ["obdeect", "obdeect_flasher", "obdeect_illuminator"])
def test_obdeect_presets_are_valid_chains(preset):
    spec = chain_spec.load_chain_spec(preset)

    assert spec.chain().stages == spec.stages
    assert spec.stage_options("sim_telarray") == {"bypass_optics": 3}


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_obdeect_chain_with_sim_telarray_optics_applies_a_factor_twice(mode):
    spec = chain_spec.ChainSpec(
        "twice",
        chain_spec.load_chain_spec("obdeect").stages,
        {"sim_telarray": {"bypass_optics": mode}},
    )

    with pytest.raises(ValueError, match="twice"):
        spec.chain()


def test_obdeect_presets_load_no_obdeect_module():
    code = (
        "import sys\n"
        "from simtools.runners import chain_spec\n"
        "for name in ('obdeect', 'obdeect_flasher', 'obdeect_illuminator'):\n"
        "    chain_spec.load_chain_spec(name).chain()\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'obdeect')\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
