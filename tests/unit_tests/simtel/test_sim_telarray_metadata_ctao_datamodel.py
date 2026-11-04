"""Compare simtools CTAO metadata with the serialization of the ctao-datamodel package.

Runs only where ctao-datamodel is installed (CI job ctao_datamodel_compatibility).
"""

import uuid
from typing import get_args

import astropy.units as u
import pytest
from astropy.time import Time

pytest.importorskip("ctao_datamodel")

# pylint: disable=wrong-import-position
from ctao_datamodel import instance_to_fits_header
from ctao_datamodel.models.common import SiteID
from ctao_datamodel.models.dataproducts import (
    Activity,
    Contact,
    Curation,
    DataAssociation,
    DataDivision,
    DataLevel,
    DataModel,
    DataModelName,
    DataType,
    FacilityName,
    InstanceIdentifier,
    ObservatoryProcess,
    Product,
    ProductType,
    Simulation,
    SimulationGridNode,
    Software,
)
from ctao_datamodel.models.dataproducts.reference import PRODUCT_MODEL_VERSION

from simtools import version
from simtools.simtel import sim_telarray_metadata
from simtools.utils.primary_particle import PrimaryParticle

_CREATED = Time("2026-01-02T03:04:05", scale="utc")
_ACTIVITY_ID = "12345678-1234-5678-1234-567812345678"
_BASE_ARGS = {
    "site": "North",
    "primary": "gamma",
    "zenith_angle": 20 * u.deg,
    "azimuth_angle": 180 * u.deg,
    "production_id": "production-42",
    "activity_id": _ACTIVITY_ID,
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


def _reference_header(args, nsb_factor=1.0):
    """Serialize the same product with the ctao-datamodel package."""
    primary = PrimaryParticle("common_name", args["primary"])
    node = {
        "zenith": args["zenith_angle"],
        "azimuth": args["azimuth_angle"],
        "nsb_factor": nsb_factor,
    }
    if args.get("ha") is not None:
        node["hour_angle"] = args["ha"]
    if args.get("dec") is not None:
        node["declination"] = args["dec"]
    production_id = args["production_id"]
    product = Product(
        description="Simulated CTAO air-shower events",
        creation_time=_CREATED,
        data=ProductType(
            level=DataLevel.SIM,
            division=DataDivision.EVENT,
            association=DataAssociation.ARRAY,
            type=DataType.AIRSHOWER_SIM,
        ),
        instance=InstanceIdentifier(
            facility_name=FacilityName.SIMULATED_CTAO,
            site_id=SiteID(f"CTAO-{args['site']}"),
            particle_pdgid=primary.pdg_id,
            messenger=sim_telarray_metadata._get_messenger(primary),
        ),
        curation=Curation(),
        model=DataModel(
            name=DataModelName.CTAO, version=PRODUCT_MODEL_VERSION, url="https://www.ctao.org"
        ),
        contact=Contact(
            name=args.get("user_name") or "not set",
            organization=args.get("user_organization") or "not set",
            email=args.get("user_email") or "not-set@example.org",
        ),
        activity=Activity(
            process=ObservatoryProcess.SIMULATION,
            name="simulate_prod",
            id=uuid.UUID(args["activity_id"]),
            start=_CREATED,
            configuration_id=production_id,
            software=Software(
                name="simtools",
                version=version.__version__,
                url="https://github.com/gammasim/simtools",
            ),
        ),
        simulation=Simulation(production_name=production_id, node=SimulationGridNode(**node)),
    )
    header = instance_to_fits_header(product, use_short=True)
    return {card.keyword: card.value for card in header.cards}


def _simtools_metadata(args):
    """Return simtools metadata for fixed creation time."""
    return sim_telarray_metadata.get_ctao_simtel_metadata(args, created=_CREATED)


def test_product_model_version_matches_package():
    """The product model version constant follows the installed package."""
    assert sim_telarray_metadata.PRODUCT_MODEL_VERSION == PRODUCT_MODEL_VERSION


@pytest.mark.parametrize(
    "extra_args",
    [
        {},
        {"site": "South", "primary": "proton"},
        {"primary": "electron", "ha": 30 * u.deg, "dec": -20 * u.deg},
        {"primary": "helium", "ha": 2 * u.hourangle, "dec": -1200 * u.arcmin},
        {
            "user_name": "A. Author",
            "user_organization": "CTAO",
            "user_email": "author@example.org",
        },
    ],
)
def test_metadata_matches_data_model_serialization(extra_args):
    """Keywords and values equal the ctao-datamodel serialization."""
    args = {**_BASE_ARGS, **extra_args}
    reference_args = {
        **args,
        **{key: args[key].to(u.deg) for key in ("zenith_angle", "azimuth_angle")},
        **{key: args[key].to(u.deg) for key in ("ha", "dec") if key in args},
    }
    expected = _reference_header(reference_args)
    actual = _simtools_metadata(args)
    expected.pop("DATAID")
    uuid.UUID(actual.pop("DATAID"))
    assert actual == expected


def test_metadata_matches_data_model_with_nsb_factor():
    """The NSB factor is serialized like the reference."""
    args = {**_BASE_ARGS, "nsb_scaling_factor": 2.5}
    expected = _reference_header(args, nsb_factor=2.5)
    actual = _simtools_metadata(args)
    assert actual["SIMNSB"] == pytest.approx(expected["SIMNSB"])


_INVALID_EMAILS = [
    "not-an-email",
    "a@b",
    "a@b..org",
    "a@-b.org",
    "a@b-.org",
    "a@b_c.org",
    "a@b.123",
    ".a@b.org",
    "a.@b.org",
    "a..b@b.org",
    "a b@c.org",
    "a@" + "b" * 64 + ".org",
    "a@b." + "o" * 260,
]
_VALID_EMAILS = ["a@b.org", "a@b.c", "first.last@sub.domain.org", "a-b@c-d.org"]


@pytest.mark.parametrize("email", _INVALID_EMAILS)
def test_email_rejected_by_data_model_is_rejected_by_simtools(email):
    """Emails rejected by the data model are rejected by simtools."""
    args = {**_BASE_ARGS, "user_email": email}
    with pytest.raises(ValueError, match=r"."):
        _reference_header(args)
    with pytest.raises(ValueError, match="Invalid contact email"):
        _simtools_metadata(args)


@pytest.mark.parametrize("email", _VALID_EMAILS)
def test_email_accepted_by_data_model_is_accepted_by_simtools(email):
    """Emails accepted by the data model are accepted by simtools."""
    args = {**_BASE_ARGS, "user_email": email}
    assert _reference_header(args)["EMAIL"] == _simtools_metadata(args)["EMAIL"] == email


def test_long_contact_name_and_organization_accepted_by_both():
    """The data model has no length limit on name and organization."""
    args = {**_BASE_ARGS, "user_name": "x" * 10000, "user_organization": "y" * 10000}
    assert _reference_header(args)["AUTHOR"] == _simtools_metadata(args)["AUTHOR"]
    assert _reference_header(args)["ORIGIN"] == _simtools_metadata(args)["ORIGIN"]


def test_registry_covers_all_data_model_keywords():
    """Every keyword of the data model is classified as used or skipped."""
    discovered_keywords = _data_model_keywords(Product)
    assert len(discovered_keywords) == len(set(discovered_keywords))
    assert set(discovered_keywords) == _USED_CTAO_KEYWORDS | _SKIPPED_CTAO_KEYWORDS
    registry = sim_telarray_metadata.get_ctao_metadata_registry()
    assert set(registry) <= _USED_CTAO_KEYWORDS
    assert not _SKIPPED_CTAO_KEYWORDS.intersection(registry)
