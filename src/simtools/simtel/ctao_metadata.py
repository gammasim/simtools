"""Create CTAO DataProducts metadata for sim_telarray configurations."""

import os
import uuid
from functools import cache

import astropy.units as u
from astropy.time import Time
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

from simtools import settings, version
from simtools.corsika.primary_particle import PrimaryParticle

_DEFAULT_AUTHOR = "not set"
_DEFAULT_EMAIL = "not-set@example.org"
_DEFAULT_ORGANIZATION = "not set"
_MODEL_URL = "https://www.ctao.org"
_SIMTOOLS_URL = "https://github.com/gammasim/simtools"
_PRODUCT_DESCRIPTION = "Simulated CTAO air-shower events"
_SIMULATION_ACTIVITY = "simulate_prod"
_MESSENGER_NAMES = {
    "electron": "electron",
    "gamma": "photon",
    "proton": "proton",
}
_REGISTRY_SAMPLE_ARGS = {
    "site": "North",
    "primary": "gamma",
    "zenith_angle": 0.0,
    "azimuth_angle": 0.0,
    "ha": 0.0,
    "dec": 0.0,
    "nsb_factor": 1.0,
    "label": "simtools-registry",
    "activity_id": "12345678-1234-5678-1234-567812345678",
}
_LEGACY_KEYS = {"primary", "zenith_angle", "azimuth_angle", "ha_angle", "dec_angle"}


def get_ctao_simtel_metadata(args, site_model=None, created=None):
    """Build CTAO and legacy simtools metadata for a sim_telarray configuration.

    Parameters
    ----------
    args : dict
        Resolved simulation arguments.
    site_model : ModelParameter, optional
        Site model used to find a resolved dimensionless NSB scaling factor.
    created : astropy.time.Time, optional
        Timestamp used for the product creation and activity start fields.

    Returns
    -------
    dict
        Metadata keys and serialized values ready for sim_telarray ``metaparam`` lines.
    """
    site = args.get("site")
    if site not in ("North", "South"):
        raise ValueError(f"Unsupported CTAO site for sim_telarray metadata: {site!r}")

    created = Time.now() if created is None else created
    primary = _get_primary(args)
    production_id = args.get("production_id") or args.get("label") or _SIMULATION_ACTIVITY
    activity_id = _get_activity_id(args)
    simulation_node = _get_simulation_node(args, site_model)
    product = Product(
        description=_PRODUCT_DESCRIPTION,
        creation_time=created,
        data=ProductType(
            level=DataLevel.SIM,
            division=DataDivision.EVENT,
            association=DataAssociation.ARRAY,
            type=DataType.AIRSHOWER_SIM,
        ),
        instance=InstanceIdentifier(
            facility_name=FacilityName.SIMULATED_CTAO,
            site_id=SiteID(f"CTAO-{site}"),
            particle_pdgid=primary.pdg_id,
            messenger=_get_messenger(primary),
        ),
        curation=Curation(),
        model=DataModel(
            name=DataModelName.CTAO,
            version=PRODUCT_MODEL_VERSION,
            url=_MODEL_URL,
        ),
        contact=Contact(
            name=os.getenv("SIMTOOLS_USER_NAME") or _DEFAULT_AUTHOR,
            organization=os.getenv("SIMTOOLS_USER_ORGANIZATION") or _DEFAULT_ORGANIZATION,
            email=os.getenv("SIMTOOLS_USER_EMAIL") or _DEFAULT_EMAIL,
        ),
        activity=Activity(
            process=ObservatoryProcess.SIMULATION,
            name=_SIMULATION_ACTIVITY,
            id=activity_id,
            start=created,
            configuration_id=production_id,
            software=Software(
                name="simtools",
                version=version.__version__,
                url=_SIMTOOLS_URL,
            ),
        ),
        simulation=Simulation(
            production_name=production_id,
            node=simulation_node,
        ),
    )
    header = instance_to_fits_header(product, use_short=True)
    metadata = {card.keyword: card.value for card in header.cards}
    _validate_short_keywords(metadata)
    metadata.update(_get_legacy_metadata(args))
    return metadata


@cache
def get_ctao_metadata_registry():
    """Build validator definitions from the installed model's FITS serialization."""
    metadata = get_ctao_simtel_metadata(
        _REGISTRY_SAMPLE_ARGS,
        created=Time("2000-01-01T00:00:00", scale="utc"),
    )
    definitions = {}
    for keyword, value in metadata.items():
        name = keyword.lower()
        if name in _LEGACY_KEYS:
            continue
        definitions[name] = {
            "name": name,
            "scope": "global",
            "mode": "set",
            "source_type": "ctao_datamodel",
            "value_schema": {"kind": "scalar", "data_type": _get_value_type(value)},
        }
    return definitions


def _get_value_type(value):
    """Infer a sim_telarray scalar type from a CTAO FITS card value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _get_primary(args):
    """Resolve the configured primary particle or raise a clear error."""
    primary_name = args.get("primary")
    if not primary_name:
        raise ValueError("A primary particle is required for CTAO sim_telarray metadata.")
    primary_type = args.get("primary_id_type") or "common_name"
    return PrimaryParticle(primary_type, primary_name)


def _get_messenger(primary):
    """Map a primary particle to the CTAO messenger vocabulary."""
    if primary.name in _MESSENGER_NAMES:
        return _MESSENGER_NAMES[primary.name]
    return f"pdgid{'+' if primary.pdg_id > 0 else '-'}{abs(primary.pdg_id)}"


def _get_activity_id(args):
    """Use the configured activity UUID or create one for this run."""
    activity_id = args.get("activity_id") or settings.config.args.get("activity_id")
    if activity_id:
        return uuid.UUID(str(activity_id))
    return uuid.uuid4()


def _get_simulation_node(args, site_model):
    """Build the phase-space node from resolved simulation arguments."""
    zenith = _get_angle(args, "zenith_angle")
    azimuth = _get_angle(args, "azimuth_angle")
    nsb_factor = _get_nsb_factor(args, site_model)
    node_values = {
        "zenith": zenith * u.deg,
        "azimuth": azimuth * u.deg,
        "hour_angle": _optional_angle(args, "ha"),
        "declination": _optional_angle(args, "dec"),
        "nsb_factor": 1.0 if nsb_factor is None else nsb_factor,
    }
    node = SimulationGridNode(**node_values)
    if nsb_factor is None:
        # The upstream field is declared as float but its documented default is None.
        object.__setattr__(node, "nsb_factor", None)
    return node


def _get_angle(args, name):
    """Return a required angle in degrees."""
    value = args.get(name)
    if value is None:
        raise ValueError(f"Missing required angle '{name}' for CTAO sim_telarray metadata.")
    return value.to_value(u.deg) if hasattr(value, "to_value") else float(value)


def _optional_angle(args, name):
    """Return an optional angle in degrees."""
    value = args.get(name)
    if value is None:
        return None
    return value.to(u.deg) if hasattr(value, "to_value") else float(value) * u.deg


def _get_nsb_factor(args, site_model):
    """Return an authoritative dimensionless NSB factor when one is available."""
    factor = args.get("nsb_factor", args.get("nsb_scaling_factor"))
    if factor is None and site_model is not None:
        parameter = site_model.parameters.get("nsb_scaling_factor")
        if isinstance(parameter, dict):
            factor = parameter.get("value")
        elif parameter is not None:
            factor = parameter
    return None if factor is None else float(factor)


def _validate_short_keywords(metadata):
    """Reject output with keys the installed model did not serialize as short cards."""
    if any(keyword.startswith("HIERARCH") for keyword in metadata):
        raise ValueError(
            "CTAO sim_telarray metadata contains a field without a short FITS keyword."
        )


def _get_legacy_metadata(args):
    """Return the existing simtools metadata aliases for the same run values."""
    metadata = {"primary": args["primary"]}
    for source, target in (
        ("zenith_angle", "zenith_angle"),
        ("azimuth_angle", "azimuth_angle"),
        ("ha", "ha_angle"),
        ("dec", "dec_angle"),
    ):
        value = args.get(source)
        if value is not None:
            metadata[target] = _get_angle(args, source)
    return metadata
