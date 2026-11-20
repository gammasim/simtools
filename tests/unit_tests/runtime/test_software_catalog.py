"""Tests for the catalog of the simulation software and the chain stages."""

import copy

import pytest

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
        (
            {"bypass_optics": "none"},
            set(),
            {"atmospheric_extinction", "mirror_reflectivity", "quantum_efficiency"},
            set(),
        ),
        (
            {"bypass_optics": "camera_entrance"},
            set(),
            {"atmospheric_extinction", "quantum_efficiency"},
            set(),
        ),
        (
            {"bypass_optics": "camera_entrance"},
            {"atmospheric_extinction"},
            {"quantum_efficiency"},
            {"atmospheric_extinction"},
        ),
        # bypass_optics none applies the extinction even if an upstream stage applies it
        (
            {"bypass_optics": "none"},
            {"atmospheric_extinction"},
            {"atmospheric_extinction", "mirror_reflectivity"},
            set(),
        ),
        (
            {"bypass_optics": "camera_only"},
            {"atmospheric_extinction"},
            {"quantum_efficiency"},
            set(),
        ),
    ],
)
def test_applied_and_skipped_factors(options, upstream, applied, skipped):
    spec = software_catalog.stage_spec("sim_telarray")

    assert applied <= spec.applied_factors(options, frozenset(upstream))
    assert spec.skipped_factors(options, frozenset(upstream)) == skipped


def test_simulates_camera_only():
    spec = software_catalog.stage_spec("sim_telarray")

    assert spec.simulates_camera_only({"bypass_optics": "camera_only"})
    assert not spec.simulates_camera_only({"bypass_optics": "camera_entrance"})
    assert not spec.simulates_camera_only({})
    assert not software_catalog.stage_spec("corsika").simulates_camera_only({})


@pytest.mark.parametrize(
    ("options", "products"),
    [
        ({}, ("cherenkov_photons", "light_emission_photons")),
        ({"bypass_optics": "none"}, ("cherenkov_photons", "light_emission_photons")),
        ({"bypass_optics": "camera_entrance"}, ("camera_photons_iact",)),
        ({"bypass_optics": "camera_only"}, ("camera_photons_iact",)),
    ],
)
def test_consumes_with_options(options, products):
    assert software_catalog.stage_spec("sim_telarray").consumes_with(options) == products


def test_consumes_with_without_option_dependence():
    spec = software_catalog.stage_spec("obdeect_atmosphere")

    assert spec.consumes_with({}) == ("photons_sky", "photons_ground")
    with pytest.raises(ValueError, match="Unknown option"):
        spec.consumes_with({"bypass_optics": "none"})


def test_consumes_with_rejects_integer_bypass_optics():
    with pytest.raises(ValueError, match="must be of type string"):
        software_catalog.stage_spec("sim_telarray").consumes_with({"bypass_optics": 1})


def test_required_efficiency_factors():
    assert software_catalog.required_efficiency_factors("air_shower") == frozenset(
        {
            "atmospheric_extinction",
            "mirror_reflectivity",
            "telescope_transmission",
            "quantum_efficiency",
        }
    )
    assert software_catalog.required_efficiency_factors("light_source") == frozenset(
        {"quantum_efficiency"}
    )


def test_ray_tracing_backends():
    assert software_catalog.ray_tracing_backends() == {
        "sim_telarray": "simtools.ray_tracing.backends.sim_telarray:SimtelBackend",
        "obdeect": "simtools.ray_tracing.backends.obdeect:ObdeectBackend",
    }


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
        (
            lambda data: data["stages"]["sim_telarray"]["consumes_by_option"][
                "bypass_optics"
            ].update(camera_only=["photons_sky"]),
            "unknown consumed products",
        ),
        (
            lambda data: data["required_efficiency_factors"].update(air_shower=["other"]),
            "Source 'air_shower': unknown efficiency factors",
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
