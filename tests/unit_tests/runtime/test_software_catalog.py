"""Tests for the catalog of the simulation software and the chain stages."""

import copy

import pytest

from simtools.runners.stages import SimulationChain
from simtools.runtime import software_catalog


def test_catalog_declares_software_stages_and_default_chains():
    assert set(software_catalog.software_names()) == {"corsika", "sim_telarray", "obdeect"}
    assert {"corsika", "sim_telarray", "obdeect_optics"} <= set(software_catalog.stage_names())
    assert software_catalog.default_chain(software_catalog.AIR_SHOWER_SOURCE) == (
        "corsika",
        "sim_telarray",
    )
    assert software_catalog.default_chain(software_catalog.LIGHT_SOURCE)[0] == "light_emission"
    assert software_catalog.product("photons_camera")["frame"] == "camera"
    assert software_catalog.software("obdeect")["package"] == "obdeect"


@pytest.mark.parametrize(
    ("lookup", "name", "match"),
    [
        (software_catalog.stage_spec, "nothing", "Unknown simulation stage 'nothing'"),
        (software_catalog.software, "nothing", "Unknown simulation software 'nothing'"),
        (software_catalog.light_source, "nothing", "Unknown light source type 'nothing'"),
    ],
)
def test_unknown_names_are_rejected(lookup, name, match):
    with pytest.raises(ValueError, match=match):
        lookup(name)


def test_stage_spec_fields():
    spec = software_catalog.stage_spec("obdeect_atmosphere")

    assert spec.name == "obdeect_atmosphere"
    assert spec.software == {"obdeect"}
    assert spec.consumes == ("photons_sky", "photons_ground")
    assert spec.produces == "photons_ground"
    assert spec.thins_photons is True
    assert spec.converter is None
    assert spec.file_type("output") == "photon_stream"
    assert software_catalog.stage_spec("corsika").consumes is None
    with pytest.raises(ValueError, match="declares no log file"):
        spec.file_type("log")
    with pytest.raises(AttributeError):
        _ = spec._private


def test_checked_options_add_defaults_and_check_values():
    spec = software_catalog.stage_spec("obdeect_atmosphere")

    assert spec.checked_options({}) == {"mode": "thinned"}
    assert spec.checked_options({"seed": 4, "wavelength_range_nm": [300, 700.5]}) == {
        "mode": "thinned",
        "seed": 4,
        "wavelength_range_nm": [300, 700.5],
    }
    with pytest.raises(ValueError, match=r"Unknown option\(s\) \['threads'\]"):
        spec.checked_options({"threads": 2})
    with pytest.raises(ValueError, match="must be of type integer"):
        spec.checked_options({"seed": True})
    with pytest.raises(ValueError, match="of number with 2 values"):
        spec.checked_options({"wavelength_range_nm": [300]})
    with pytest.raises(ValueError, match="choose from"):
        spec.checked_options({"mode": "other"})


def test_list_option_accepts_one_value():
    spec = software_catalog.stage_spec("stream_to_simtel")

    assert spec.checked_options({"allow_pending": "atmosphere"}) == {
        "allow_pending": ["atmosphere"]
    }


def test_options_from_arguments_are_not_checked():
    spec = software_catalog.stage_spec("obdeect_flasher")

    assert spec.checked_options({"telescope": "LSTN-01"}) == {"telescope": "LSTN-01"}


@pytest.mark.parametrize(
    ("options", "upstream", "applied", "skipped"),
    [
        ({}, set(), {"atmospheric_extinction", "mirror_reflectivity"}, set()),
        ({"bypass_optics": 1}, set(), {"atmospheric_extinction", "quantum_efficiency"}, set()),
        (
            {"bypass_optics": 1},
            {"atmospheric_extinction"},
            {"quantum_efficiency"},
            {"atmospheric_extinction"},
        ),
        # mode 0 applies the extinction even if an upstream stage applies it (rejected later)
        ({"bypass_optics": 0}, {"atmospheric_extinction"}, {"atmospheric_extinction"}, set()),
        ({"bypass_optics": 3}, {"atmospheric_extinction"}, {"quantum_efficiency"}, set()),
    ],
)
def test_applied_and_skipped_factors(options, upstream, applied, skipped):
    spec = software_catalog.stage_spec("sim_telarray")

    assert applied <= spec.applied_factors(options, frozenset(upstream))
    assert spec.skipped_factors(options, frozenset(upstream)) == skipped


def test_simulates_camera_only():
    spec = software_catalog.stage_spec("sim_telarray")

    assert spec.simulates_camera_only({"bypass_optics": 3})
    assert not spec.simulates_camera_only({"bypass_optics": 1})
    assert not software_catalog.stage_spec("corsika").simulates_camera_only({})


def test_load_object_and_light_sources():
    assert software_catalog.load_object("simtools.runtime.software_catalog:stage_spec") is (
        software_catalog.stage_spec
    )
    assert set(software_catalog.light_source_names()) == {"flasher", "illuminator"}
    assert software_catalog.light_source("flasher")["model_types"] == ["flat_fielding"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda data: data["stages"]["corsika"].update(software="other"), "unknown software"),
        (lambda data: data["stages"]["corsika"].update(produces="other"), "unknown products"),
        (
            lambda data: data["stages"]["obdeect_atmosphere"].update(applies=["other"]),
            "unknown efficiency factors",
        ),
        (
            lambda data: data["stages"]["sim_telarray"].update(camera_only={"other": 1}),
            "unknown options",
        ),
        (
            lambda data: data["stages"]["obdeect_flasher"].update(light_source="other"),
            "unknown light sources",
        ),
        (
            lambda data: data["default_chains"].update(air_shower=["other"]),
            "Default chain 'air_shower': unknown stages",
        ),
    ],
)
def test_references_are_checked(change, match):
    data = copy.deepcopy(software_catalog.catalog())
    change(data)

    with pytest.raises(ValueError, match=match):
        software_catalog._check_references(data)  # pylint: disable=protected-access


def test_file_references_are_checked():
    data = copy.deepcopy(software_catalog.catalog())
    data["stages"]["obdeect_optics"]["files"]["log"] = "no_such_file"

    with pytest.raises(ValueError, match="Unknown file product 'no_such_file'"):
        software_catalog._check_references(data)  # pylint: disable=protected-access


def test_star_stages_are_declared_and_validated():
    assert software_catalog.stage_spec("obdeect_star").produces == "star_images"
    assert software_catalog.stage_spec("sim_telarray_star").software == {"sim_telarray"}
    assert software_catalog.product("star_images")["format"] == "table"


@pytest.mark.parametrize(
    ("items", "backend"),
    [
        (None, None),
        ([], None),
        ("sim_telarray_star", "sim_telarray"),
        (["obdeect_star"], "obdeect"),
        (["corsika", "sim_telarray"], None),
    ],
)
def test_star_backend(items, backend):
    assert software_catalog.star_backend(items) == backend


def test_star_backend_rejects_several_star_stages():
    with pytest.raises(ValueError, match="Only one ray-tracing stage"):
        software_catalog.star_backend(["sim_telarray_star", "obdeect_star"])


def test_star_stages_are_not_run_in_a_chain():
    with pytest.raises(ValueError, match="selects a ray-tracing backend"):
        SimulationChain(("obdeect_star",))
