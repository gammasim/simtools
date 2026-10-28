"""Create CTAO DataProducts metadata for sim_telarray configurations."""

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

from simtools import version
from simtools.corsika.primary_particle import PrimaryParticle
from simtools.model.model_parameter import InvalidModelParameterError

_DEFAULT_CONTACT = "not set"
_DEFAULT_EMAIL = "not-set@example.org"
_MODEL_URL = "https://www.ctao.org"
_SIMTOOLS_URL = "https://github.com/gammasim/simtools"
_PRODUCT_DESCRIPTION = "Simulated CTAO air-shower events"
_SIMULATION_ACTIVITY = "simulate_prod"
_NSB_SCALING_FACTOR = "nsb_scaling_factor"
_PRIMARY = "primary"
_ZENITH_ANGLE = "zenith_angle"
_AZIMUTH_ANGLE = "azimuth_angle"
_ACTIVITY_ID = "activity_id"
# sim_telarray default for nsb_scaling_factor (see its model-parameter schema).
_DEFAULT_NSB_SCALING_FACTOR = 1.0
_MESSENGER_NAMES = {
    "electron": "electron",
    "gamma": "photon",
    "proton": "proton",
}
_REGISTRY_SAMPLE_ARGS = {
    "site": "North",
    _PRIMARY: "gamma",
    _ZENITH_ANGLE: 0.0,
    _AZIMUTH_ANGLE: 0.0,
    "ha": 0.0,
    "dec": 0.0,
    "label": "simtools-registry",
    _ACTIVITY_ID: "12345678-1234-5678-1234-567812345678",
}


def get_ctao_simtel_metadata(args, site_model=None, created=None):
    """Build CTAO metadata for a sim_telarray configuration.

    Parameters
    ----------
    args : dict
        Resolved simulation arguments.
    site_model : ModelParameter, optional
        Site model providing the NSB scaling factor when ``args`` does not override it.
        Without either, the sim_telarray default factor is used.
    created : astropy.time.Time, optional
        Timestamp used for the product creation and activity start fields.

    Returns
    -------
    dict
        FITS keywords and serialized values ready for sim_telarray ``metaparam`` lines.
    """
    site = args.get("site")
    if site not in ("North", "South"):
        raise ValueError(f"Unsupported CTAO site for sim_telarray metadata: {site!r}")

    created = Time.now() if created is None else created
    primary = _get_primary(args)
    production_id = args.get("production_id") or args.get("label") or _SIMULATION_ACTIVITY
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
            name=args.get("user_name") or _DEFAULT_CONTACT,
            organization=args.get("user_organization") or _DEFAULT_CONTACT,
            email=args.get("user_email") or _DEFAULT_EMAIL,
        ),
        activity=Activity(
            process=ObservatoryProcess.SIMULATION,
            name=_SIMULATION_ACTIVITY,
            id=uuid.UUID(str(args[_ACTIVITY_ID])) if args.get(_ACTIVITY_ID) else uuid.uuid4(),
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
            node=_get_simulation_node(args, site_model),
        ),
    )
    header = instance_to_fits_header(product, use_short=True)
    metadata = {card.keyword: card.value for card in header.cards}
    if any(keyword.startswith("HIERARCH") for keyword in metadata):
        raise ValueError(
            "CTAO sim_telarray metadata contains a field without a short FITS keyword."
        )
    return metadata


@cache
def get_ctao_metadata_registry():
    """Build validator definitions from the installed model's FITS serialization."""
    metadata = get_ctao_simtel_metadata(
        _REGISTRY_SAMPLE_ARGS,
        created=Time("2000-01-01T00:00:00", scale="utc"),
    )
    return {
        keyword.lower(): {
            "name": keyword.lower(),
            "scope": "global",
            "mode": "set",
            "source_type": "ctao_datamodel",
            "value_schema": {"kind": "scalar", "data_type": _get_value_type(value)},
        }
        for keyword, value in metadata.items()
    }


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
    primary_name = args.get(_PRIMARY)
    if not primary_name:
        raise ValueError("A primary particle is required for CTAO sim_telarray metadata.")
    primary_type = args.get("primary_id_type") or "common_name"
    return PrimaryParticle(primary_type, primary_name)


def _get_messenger(primary):
    """Map a primary particle to the CTAO messenger vocabulary."""
    if primary.name in _MESSENGER_NAMES:
        return _MESSENGER_NAMES[primary.name]
    return f"pdgid{'+' if primary.pdg_id > 0 else '-'}{abs(primary.pdg_id)}"


def _get_simulation_node(args, site_model):
    """Build the phase-space node; unknown sky coordinates keep the model defaults."""
    node_values = {
        "zenith": _get_angle(args, _ZENITH_ANGLE, required=True),
        "azimuth": _get_angle(args, _AZIMUTH_ANGLE, required=True),
        "hour_angle": _get_angle(args, "ha"),
        "declination": _get_angle(args, "dec"),
        "nsb_factor": _get_nsb_factor(args, site_model),
    }
    return SimulationGridNode(
        **{key: value for key, value in node_values.items() if value is not None}
    )


def _get_angle(args, name, required=False):
    """Return an angle in degrees as a quantity; plain numbers are taken as degrees."""
    value = args.get(name)
    if value is None:
        if required:
            raise ValueError(f"Missing required angle '{name}' for CTAO sim_telarray metadata.")
        return None
    return value.to(u.deg) if hasattr(value, "to") else float(value) * u.deg


def _get_nsb_factor(args, site_model):
    """Return the dimensionless NSB scaling factor used by sim_telarray."""
    factor = args.get(_NSB_SCALING_FACTOR)
    if factor is None and site_model is not None:
        try:
            factor = site_model.get_parameter_value(_NSB_SCALING_FACTOR)
        except InvalidModelParameterError:
            factor = None
    return _DEFAULT_NSB_SCALING_FACTOR if factor is None else float(factor)
