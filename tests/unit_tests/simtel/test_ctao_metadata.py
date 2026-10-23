"""Tests for the CTAO metadata bridge."""

import math
from typing import get_args

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from simtools.simtel import ctao_metadata

_BASE_ARGS = {
    "site": "North",
    "primary": "gamma",
    "zenith_angle": 20 * u.deg,
    "azimuth_angle": 180 * u.deg,
    "production_id": "production-42",
    "activity_id": "12345678-1234-5678-1234-567812345678",
}
_USED_CTAO_KEYWORDS = {
    "ctaometa",
    "title",
    "created",
    "datalevl",
    "datadiv",
    "dataasso",
    "datatype",
    "dataid",
    "telescop",
    "instrume",
    "pdgid",
    "msngr",
    "author",
    "origin",
    "email",
    "license",
    "licenurl",
    "model",
    "modelver",
    "modelurl",
    "actproc",
    "activity",
    "actid",
    "actstart",
    "software",
    "softver",
    "softurl",
    "actconf",
    "simprod",
    "simzen",
    "simaz",
    "simha",
    "simdec",
    "simnsb",
}
_SKIPPED_CTAO_KEYWORDS = {
    "actdesc",
    "release",
    "referenc",
    "copyrigh",
    "actstop",
    "ae_class",
    "ae_id",
    "alt_max",
    "alt_min",
    "assembly",
    "az_mean",
    "az_range",
    "batch_id",
    "chunk_id",
    "comment",
    "datacat",
    "datasrc",
    "dec_pnt",
    "emax",
    "emin",
    "evtype",
    "fov",
    "locrunid",
    "locruntp",
    "luncycle",
    "mjdref",
    "object",
    "obs_id",
    "obsnight",
    "obsgeo-b",
    "obsgeo-h",
    "obsgeo-l",
    "period",
    "pubdate",
    "radesys",
    "ra_pnt",
    "region",
    "rights",
    "sb_id",
    "sdc_nsb",
    "sub_id",
    "sublvl",
    "tracking",
    "tstart",
    "tstop",
    "timedelt",
    "timesys",
    "timeunit",
    "trefpos",
}


def _data_model_keywords(model, ancestors=()):
    """Collect FITS keywords declared by every nested Product model."""
    if model in ancestors:
        return []
    keywords = []
    for field in model.model_fields.values():
        metadata = field.json_schema_extra or {}
        keyword = metadata.get("fits_keyword")
        if keyword:
            keywords.append(keyword.lower())
        for nested_model in _nested_models(field.annotation):
            keywords.extend(_data_model_keywords(nested_model, (*ancestors, model)))
    return keywords


def _nested_models(annotation):
    """Return CTAO model classes nested in a field annotation."""
    if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
        return [annotation]
    return [model for item in get_args(annotation) for model in _nested_models(item)]


def test_get_ctao_simtel_metadata_creates_north_product():
    metadata = ctao_metadata.get_ctao_simtel_metadata(
        _BASE_ARGS,
        created=Time("2026-01-02T03:04:05", scale="utc"),
    )

    assert metadata["DATALEVL"] == "SIM"
    assert metadata["DATADIV"] == "Event"
    assert metadata["DATAASSO"] == "Array"
    assert metadata["DATATYPE"] == "AirShowerSim"
    assert metadata["TELESCOP"] == "Simulated-CTAO"
    assert metadata["INSTRUME"] == "CTAO-North"
    assert metadata["PDGID"] == 22
    assert metadata["MSNGR"] == "photon"
    assert metadata["SIMPROD"] == "production-42"
    assert metadata["MODELVER"] == ctao_metadata.PRODUCT_MODEL_VERSION
    assert metadata["primary"] == "gamma"
    assert math.isclose(metadata["SIMZEN"], 20.0)
    assert math.isclose(metadata["SIMAZ"], 180.0)
    assert "SIMNSB" not in metadata
    assert metadata["DATAID"] != metadata["ACTID"]


@pytest.mark.parametrize(
    ("hour_angle", "declination"),
    [(30, -20), (30 * u.deg, -20 * u.deg), (2 * u.hourangle, -1200 * u.arcmin)],
)
def test_get_ctao_simtel_metadata_accepts_south_and_optional_coordinates(hour_angle, declination):
    args = {
        **_BASE_ARGS,
        "site": "South",
        "ha": hour_angle,
        "dec": declination,
        "nsb_factor": 1.25,
    }

    metadata = ctao_metadata.get_ctao_simtel_metadata(args)

    assert metadata["INSTRUME"] == "CTAO-South"
    assert math.isclose(metadata["SIMHA"], 30.0)
    assert math.isclose(metadata["SIMDEC"], -20.0)
    assert math.isclose(metadata["SIMNSB"], 1.25)


def test_optional_angle_preserves_angular_units():
    angle = ctao_metadata._optional_angle({"ha": 2 * u.hourangle}, "ha")

    assert_quantity_allclose(angle, 30 * u.deg)


def test_get_ctao_simtel_metadata_uses_contact_defaults(monkeypatch):
    for name in ("SIMTOOLS_USER_NAME", "SIMTOOLS_USER_ORGANIZATION", "SIMTOOLS_USER_EMAIL"):
        monkeypatch.delenv(name, raising=False)

    metadata = ctao_metadata.get_ctao_simtel_metadata(_BASE_ARGS)

    assert metadata["AUTHOR"] == "not set"
    assert metadata["ORIGIN"] == "not set"
    assert metadata["EMAIL"] == "not-set@example.org"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({**_BASE_ARGS, "site": "East"}, "Unsupported CTAO site"),
        ({**_BASE_ARGS, "primary": "not-a-particle"}, "Invalid particle name"),
        ({key: value for key, value in _BASE_ARGS.items() if key != "primary"}, "primary particle"),
    ],
)
def test_get_ctao_simtel_metadata_rejects_bad_input(args, message):
    with pytest.raises(ValueError, match=message):
        ctao_metadata.get_ctao_simtel_metadata(args)


def test_ctao_registry_uses_serialized_data_model_keywords():
    registry = ctao_metadata.get_ctao_metadata_registry()

    assert registry["dataid"]["source_type"] == "ctao_datamodel"
    assert registry["simnsb"]["value_schema"]["data_type"] == "number"
    assert set(registry) == _USED_CTAO_KEYWORDS
    assert not _SKIPPED_CTAO_KEYWORDS.intersection(registry)
    discovered_keywords = _data_model_keywords(ctao_metadata.Product)
    assert len(discovered_keywords) == len(set(discovered_keywords))
    assert set(discovered_keywords) == _USED_CTAO_KEYWORDS | _SKIPPED_CTAO_KEYWORDS
