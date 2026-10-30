"""sim_telarray metadata: CTAO FITS metadata builder and meta-parameter registry validation."""

import re
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
from simtools.constants import (
    SIM_TELARRAY_META_PARAMETER_METASCHEMA,
    SIM_TELARRAY_META_PARAMETER_REGISTRY,
)
from simtools.corsika.primary_particle import PrimaryParticle
from simtools.data_model import parameter_catalog as catalog
from simtools.data_model import schema
from simtools.model.model_parameter import InvalidModelParameterError
from simtools.simtel.simtel_io_metadata import normalize_sim_telarray_metadata

META_PARAMETER_SOURCE_TYPES = ("all", "generated", "model_parameter", "ctao_datamodel")
# sources whose scope and values are fully described by the registry
_REGISTRY_DEFINED_SOURCE_TYPES = {"generated", "ctao_datamodel"}
_REGISTRY_SAMPLE_ARGS = {
    "site": "North",
    "primary": "gamma",
    "zenith_angle": 0.0,
    "azimuth_angle": 0.0,
    "ha": 0.0,
    "dec": 0.0,
    "label": "simtools-registry",
    "activity_id": "12345678-1234-5678-1234-567812345678",
}


def get_ctao_simtel_metadata(args, site_model=None, created=None):
    """
    Build CTAO metadata for a sim_telarray configuration.

    Parameters
    ----------
    args : dict
        Resolved simulation arguments.
    site_model : ModelParameter, optional
        Site model providing the NSB scaling factor when ``args`` does not override it.
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
    if not args.get("primary"):
        raise ValueError("A primary particle is required for CTAO sim_telarray metadata.")

    created = Time.now() if created is None else created
    primary = PrimaryParticle(args.get("primary_id_type") or "common_name", args["primary"])
    production_id = args.get("production_id") or args.get("label") or "simulate_prod"
    product = Product(
        description="Simulated CTAO air-shower events",
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
            id=uuid.UUID(str(args["activity_id"])) if args.get("activity_id") else uuid.uuid4(),
            start=created,
            configuration_id=production_id,
            software=Software(
                name="simtools",
                version=version.__version__,
                url="https://github.com/gammasim/simtools",
            ),
        ),
        simulation=Simulation(
            production_name=production_id,
            node=SimulationGridNode(
                **{
                    key: value
                    for key, value in {
                        "zenith": _get_angle(args, "zenith_angle", required=True),
                        "azimuth": _get_angle(args, "azimuth_angle", required=True),
                        "hour_angle": _get_angle(args, "ha"),
                        "declination": _get_angle(args, "dec"),
                        "nsb_factor": _get_nsb_factor(args, site_model),
                    }.items()
                    if value is not None
                }
            ),
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
        _REGISTRY_SAMPLE_ARGS, created=Time("2000-01-01T00:00:00", scale="utc")
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
    for python_type, name in ((bool, "boolean"), (int, "integer"), (float, "number")):
        if isinstance(value, python_type):
            return name
    return "string"


def _get_messenger(primary):
    """Map a primary particle to the CTAO messenger vocabulary."""
    if primary.name in ("electron", "proton"):
        return primary.name
    if primary.name == "gamma":
        return "photon"
    return f"pdgid{'+' if primary.pdg_id > 0 else '-'}{abs(primary.pdg_id)}"


def _get_angle(args, name, required=False):
    """Return an angle in degrees as a quantity; plain numbers are taken as degrees."""
    value = args.get(name)
    if value is None:
        if required:
            raise ValueError(f"Missing required angle '{name}' for CTAO sim_telarray metadata.")
        return None
    return value.to(u.deg) if hasattr(value, "to") else float(value) * u.deg


def _get_nsb_factor(args, site_model):
    """Return the dimensionless NSB scaling factor (sim_telarray default: 1.0)."""
    factor = args.get("nsb_scaling_factor")
    if factor is None and site_model is not None:
        try:
            factor = site_model.get_parameter_value("nsb_scaling_factor")
        except InvalidModelParameterError:
            factor = None
    return 1.0 if factor is None else float(factor)


# --- meta-parameter registry and validation ---------------------------------------------


def get_meta_parameter_registry(schema_version=None, validate=True, source_type="all"):
    """
    Return the expanded sim_telarray meta-parameter registry.

    Parameters
    ----------
    schema_version : str, optional
        Version of the registry schema to load.
    validate : bool, optional
        Whether to validate the loaded registry against the metaschema.
    source_type : str, optional
        Metadata source filter: ``all``, ``generated``, ``model_parameter``, or
        ``ctao_datamodel``.
    """
    registry_source = schema.load_schema(
        SIM_TELARRAY_META_PARAMETER_REGISTRY, schema_version=schema_version or "latest"
    )
    if validate:
        schema.validate_dict_using_schema(
            registry_source,
            schema_file=SIM_TELARRAY_META_PARAMETER_METASCHEMA,
            offline=True,
            ignore_software_version=True,
        )
    if source_type not in META_PARAMETER_SOURCE_TYPES:
        raise ValueError(f"Unsupported source type: {source_type}")
    registry = _build_meta_parameter_registry(registry_source)
    if source_type == "all":
        return registry
    return {
        **registry,
        "meta_parameters": {
            name: definition
            for name, definition in registry["meta_parameters"].items()
            if definition["source_type"] == source_type
        },
    }


def validate_metadata(meta_parameters):
    """
    Validate emitted sim_telarray metadata lines against the registry.

    Parameters
    ----------
    meta_parameters : list of str
        Emitted sim_telarray metadata lines.
    """
    registry = get_meta_parameter_registry(validate=False)["meta_parameters"]
    for line in meta_parameters:
        parsed = _parse_metadata_line(line)
        definition = registry.get(parsed["name"])
        if definition is None:
            raise KeyError(f"Unknown sim_telarray metadata key emitted by writer: {parsed['name']}")
        if definition["mode"] != parsed["mode"]:
            raise ValueError(
                f"sim_telarray metadata mode mismatch for {parsed['name']}: "
                f"{parsed['mode']} != {definition['mode']}"
            )
        if (
            parsed["scope"] is not None
            and definition["source_type"] in _REGISTRY_DEFINED_SOURCE_TYPES
            and definition["scope"] != parsed["scope"]
        ):
            raise ValueError(
                f"sim_telarray metadata scope mismatch for {parsed['name']}: "
                f"{parsed['scope']} != {definition['scope']}"
            )
        if parsed["value"] is None:
            if definition["mode"] in {"set", "assign"}:
                raise ValueError(
                    f"sim_telarray metadata value missing for required key {parsed['name']}"
                )
            continue
        _validate_metadata_value(parsed["name"], parsed["value"], definition["value_schema"])


def validate_metadata_values(metadata):
    """
    Validate known decoded sim_telarray metadata values against the registry.

    Parameters
    ----------
    metadata : dict
        Decoded sim_telarray metadata values, keyed by emitted name.
    """
    registry = get_meta_parameter_registry(validate=False)["meta_parameters"]
    for name, value in normalize_sim_telarray_metadata(metadata).items():
        definition = registry.get(name)
        if (
            definition is None
            or value is None
            or definition["source_type"] not in _REGISTRY_DEFINED_SOURCE_TYPES
            or definition["mode"] == "add"
        ):
            continue
        _validate_metadata_value(name, str(value), definition["value_schema"])


def _parse_metadata_line(line):
    """Parse one emitted sim_telarray metadata line (``metaparam`` or ``name = value``)."""
    parts = line.split(maxsplit=3)
    if len(parts) == 4 and parts[0] == "metaparam":
        scope, mode = parts[1], parts[2]
        name, separator, value = parts[3].partition("=")
        if (
            scope not in ("global", "telescope")
            or mode not in ("add", "set")
            or not name.strip()
            or any(char.isspace() for char in name.strip())
        ):
            raise ValueError(f"Unsupported sim_telarray metadata line: {line}")
        return {
            "scope": scope,
            "mode": mode,
            "name": name.strip().lower(),
            "value": value.strip() if separator else None,
        }

    name, separator, value = line.partition("=")
    if separator and name.strip().replace("_", "").isalnum():
        return {
            "scope": None,
            "mode": "assign",
            "name": name.strip().lower(),
            "value": value.strip(),
        }

    raise ValueError(f"Unsupported sim_telarray metadata line: {line}")


def _build_meta_parameter_registry(registry_source):
    """Build expanded sim_telarray meta_parameter registry from source overlays."""
    meta_parameters = {
        name: {"name": name, "source_type": "generated", **definition}
        for name, definition in registry_source.get("generated_meta_parameters", {}).items()
    }
    for source_name, model_schema in catalog.model_parameters().items():
        try:
            emitted_name, _ = _get_emitted_name_and_mode(model_schema)
        except KeyError:
            continue
        if emitted_name not in meta_parameters:
            definition = _build_model_parameter_definition(source_name)
            meta_parameters[definition["name"]] = definition
    meta_parameters.update(get_ctao_metadata_registry())
    return {
        **{
            key: value
            for key, value in registry_source.items()
            if key != "generated_meta_parameters"
        },
        "meta_parameters": meta_parameters,
    }


def _build_model_parameter_definition(source_name):
    """Build one sim_telarray meta_parameter definition from a model parameter schema."""
    model_schema = schema.get_model_parameter_schema(source_name)
    emitted_name, mode = _get_emitted_name_and_mode(model_schema)
    scope = "global" if model_schema.get("instrument", {}).get("class") == "Site" else "telescope"
    return {
        "name": emitted_name,
        "scope": scope,
        "mode": mode,
        "source_type": "model_parameter",
        "value_schema": (
            _get_software_definition(model_schema).get("output_value_schema")
            or _derive_value_schema(model_schema)
        ),
    }


def _get_software_definition(model_schema):
    """Return the sim_telarray simulation_software definition for a model schema."""
    for software in model_schema.get("simulation_software", []):
        if software.get("name") == "sim_telarray":
            return software
    raise KeyError(f"Model parameter without sim_telarray mapping: {model_schema['name']}")


def _get_emitted_name_and_mode(model_schema):
    """Return emitted sim_telarray metadata name and metadata mode."""
    software = _get_software_definition(model_schema)
    mode = software.get("meta_parameter_mode")
    if mode is None:
        raise KeyError(f"Model parameter without meta parameter mode: {model_schema['name']}")
    return software.get("internal_parameter_name", model_schema["name"]), mode


def _derive_value_schema(model_schema):
    """Infer value schema from model parameter schema structure."""
    data = model_schema.get("data", [])
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or len(data) == 0:
        return {"kind": "scalar", "data_type": "string"}

    if len(data) > 1 and all(_is_numeric_type(entry.get("type")) for entry in data):
        return {"kind": "fixed_numeric_tuple", "item_type": "number", "length": len(data)}

    datum = data[0]
    data_type = datum.get("type")
    if data_type == "file":
        value_schema = {"kind": "file_name"}
        if datum.get("default") is None:
            value_schema["allow_none_literal"] = True
        return value_schema
    if _is_integer_type(data_type):
        return {"kind": "scalar", "data_type": "integer"}
    if _is_numeric_type(data_type):
        return {"kind": "scalar", "data_type": "number"}
    return {"kind": "scalar", "data_type": "boolean" if data_type == "boolean" else "string"}


def _is_integer_type(data_type):
    """Check whether a schema data type is integer-like."""
    return data_type in {"int", "int64", "uint", "uint32", "uint64"}


def _is_numeric_type(data_type):
    """Check whether a schema data type is numeric-like."""
    return _is_integer_type(data_type) or data_type in {"double", "float64", "float32"}


def _validate_metadata_value(name, value, value_schema):
    """Validate one emitted metadata value against its value schema."""
    kind = value_schema["kind"]
    if kind == "scalar":
        _validate_scalar_value(name, value, value_schema["data_type"], value_schema)
    elif kind == "file_name":
        if value_schema.get("allow_none_literal") and value == "none":
            return
        if value == "" and not value_schema.get("allow_empty", False):
            raise ValueError(f"Empty file-like metadata value for {name}")
    elif kind == "fixed_numeric_tuple":
        parts = value.split()
        if len(parts) != value_schema["length"]:
            raise ValueError(
                f"sim_telarray metadata tuple length mismatch for {name}: "
                f"{len(parts)} != {value_schema['length']}"
            )
        for part in parts:
            float(part)
    elif kind == "sim_telarray_key_value_string":
        if not re.fullmatch(value_schema["regex"], value):
            raise ValueError(f"sim_telarray metadata value for {name} does not match regex")
    else:
        raise ValueError(f"Unsupported value schema kind for {name}: {kind}")


def _validate_scalar_value(name, value, data_type, value_schema):
    """Validate one scalar metadata value against its sim_telarray data type."""
    if data_type == "string":
        if value == "" and not value_schema.get("allow_empty", False):
            raise ValueError(f"Empty string metadata value for {name}")
    elif data_type == "integer":
        int(value)
    elif data_type == "number":
        float(value)
    elif data_type == "boolean":
        if value not in ("0", "1", "true", "false", "True", "False"):
            raise ValueError(f"Invalid boolean metadata value for {name}: {value}")
    else:
        raise ValueError(f"Unsupported scalar metadata data type for {name}: {data_type}")
