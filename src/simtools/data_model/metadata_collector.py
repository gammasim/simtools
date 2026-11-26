"""
Metadata collector for simtools.

This should be the only module in simtools with knowledge on the
implementation of the observatory metadata model.

"""

import getpass
import importlib
import json
import logging
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from astropy.table import Table

import simtools.utils.general as gen
import simtools.version
from simtools.constants import METADATA_JSON_SCHEMA, SCHEMA_PATH, SCHEMA_URL
from simtools.data_model import metadata_model, product_checksum, schema, schema_loader
from simtools.data_model.metadata_keys import (
    LEGACY_OBSERVATORY_KEY,
    OBSERVATORY_KEY,
    normalize_observatory_key,
)
from simtools.data_model.product_checksum import SHA256_KEY
from simtools.io import ascii_handler
from simtools.io.file_type import product_suffixes
from simtools.runtime.context import RunContext
from simtools.utils import names, value_conversion

_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?<![\w-])(?P<option>--)?"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*(?:password|passwd|pw|pwd|secret|token|api[_-]?key)"
    r"[A-Za-z0-9_-]*)(?P<separator>=|:)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s]+)",
    re.IGNORECASE,
)
_SENSITIVE_KEY = re.compile(
    r"password|passwd|(^|[_-])(pw|pwd)($|[_-])|secret|token|api[_-]?key", re.IGNORECASE
)
# format-specific readers for simulation output files (imported on demand)
_FORMAT_READERS = {
    "sim_telarray_output": "simtools.simtel.metadata_reader",
    "corsika_output": "simtools.sim_events.metadata_reader",
}
# application arguments that point to the local file system (never written to metadata)
_LOCAL_PATH_ARGUMENTS = frozenset(
    """data_path data_search_path env_file grid_output_path log_file log_file_path output_path
    runtime_environment_file simulation_models_git_path simulation_models_path value_data_path
    """.split()
)
# instrument classes of array element collections (see array_elements.yml)
_INSTRUMENT_CLASSES = {"telescopes": "telescope", "calibration_devices": "calibration_device"}


class MetadataCollector:
    """
    Collects metadata to describe the current simtools activity and its data products.

    Parameters
    ----------
    args_dict: dict
        Command line parameters.
    metadata_file_name: str
        Name of metadata file (only required when args_dict is None).
    model_parameter_name: str
        Name of model parameter.
    observatory: str
        Top-level key of the observatory metadata block (default: "ctao").
    clean_meta: bool
        Clean metadata from None values and empty lists (default: True).
    schema_version: str
        Version of the metadata schema to use (default: 'latest').
    context: RunContext, optional
        Run context providing the activity.
    """

    def __init__(
        self,
        args_dict=None,
        metadata_file_name=None,
        model_parameter_name=None,
        observatory=OBSERVATORY_KEY,
        clean_meta=True,
        schema_version="latest",
        context=None,
    ):
        """Initialize metadata collector."""
        self._logger = logging.getLogger(__name__)
        self.observatory = observatory
        self.context = context or RunContext.from_environment()
        self.args_dict = args_dict or {}
        self.model_parameter_name = model_parameter_name
        self.schema_file = None
        self.schema_dict = None
        self.top_level_meta = metadata_model.get_default_metadata_dict(
            schema_version=schema_version
        )
        self.input_metadata = self._read_input_metadata_from_file(metadata_file_name)
        self.collect_meta_data()
        if clean_meta:
            self.top_level_meta = self.clean_meta_data(self.top_level_meta)
            metadata_model.add_required_fields(
                self.top_level_meta, observatory=self.observatory, schema_version=schema_version
            )

    def collect_meta_data(self):
        """Collect and verify product metadata for each main-level metadata type."""
        for meta_type in self.top_level_meta[self.observatory].keys():
            try:
                getattr(self, f"_fill_{meta_type}_meta")(
                    self.top_level_meta[self.observatory][meta_type]
                )
            except AttributeError:
                self._logger.debug(f"Method _fill_{meta_type}_meta not implemented")

    def get_top_level_metadata(self):
        """
        Return top level metadata dictionary for output.

        The activity end time is updated. Local schema file paths are replaced by the URL
        of the schema (or removed if the schema is not available from a URL).
        """
        try:
            self.top_level_meta[self.observatory]["activity"]["end"] = self.args_dict.get(
                "activity_end", gen.now_date_time_in_isoformat()
            )
        except KeyError:
            pass
        metadata = deepcopy(self.top_level_meta)
        model = metadata.get(self.observatory, {}).get("product", {}).get("data", {}).get("model")
        if model is not None and "url" in model:
            model["url"] = _data_model_url(model["url"])
            if model["url"] is None:
                model.pop("url")
        return metadata

    @staticmethod
    def dump(args_dict, output_file, add_activity_name=False, context=None):
        """Write metadata of a new collector to file (see ``write``)."""
        MetadataCollector(args_dict, context=context).write(
            output_file, add_activity_name=add_activity_name
        )

    def write(self, yml_file=None, keys_lower_case=True, add_activity_name=False):
        """Write top level metadata to a yaml file and return its name."""
        metadata = self.get_top_level_metadata()
        activity_name = (
            metadata.get(self.observatory, {}).get("activity", {}).get("name") or ""
        ).rstrip(".")
        suffix = f".{activity_name}.meta.yml" if add_activity_name else ".meta.yml"
        if yml_file is None:
            raise TypeError("No output file for metadata defined")
        yml_file = names.file_name_with_version(yml_file, suffix)
        try:
            ascii_handler.write_data_to_file(
                data=value_conversion.change_dict_keys_case(
                    value_conversion.remove_substring_recursively_from_dict(
                        metadata, substring="\n"
                    ),
                    keys_lower_case,
                ),
                output_file=yml_file,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Error writing metadata to {yml_file}") from exc
        self._logger.info(f"Writing metadata to {yml_file}")
        return yml_file

    def set_product_checksum(self, sha256):
        """Set the SHA-256 checksum of the data product (``product.sha256``)."""
        self.top_level_meta[self.observatory]["product"][SHA256_KEY] = sha256

    def get_data_model_schema_file_name(self):
        """Return data model schema file name (command line, metadata, model name, input)."""
        if self.args_dict.get("schema_file"):
            self._logger.debug(f"Schema file from command line: {self.args_dict['schema_file']}")
            return self.args_dict["schema_file"]

        try:
            url = self.top_level_meta[self.observatory]["product"]["data"]["model"]["url"]
            if url:
                self._logger.debug(f"Schema file from product metadata: {url}")
                return url
        except KeyError:
            pass

        if self.model_parameter_name:
            try:
                schema_file = str(schema.get_model_parameter_schema_file(self.model_parameter_name))
            except FileNotFoundError:
                self._logger.debug(
                    f"No schema file for model parameter {self.model_parameter_name}"
                )
            else:
                self._logger.debug(f"Schema file from data model name: {self.model_parameter_name}")
                return schema_file

        try:
            url = self.input_metadata[0][self.observatory]["product"]["data"]["model"]["url"]
            self._logger.debug(f"Schema file from input metadata: {url}")
            return url
        except KeyError, TypeError, IndexError:
            return None

    def get_data_model_schema_dict(self):
        """Return data model schema dictionary (empty if no valid schema file is given)."""
        if self.schema_file is None:
            self._logger.debug(f"No valid schema file provided ({self.schema_file}).")
            return {}
        try:
            return schema_loader.load_schema(self.schema_file)
        except FileNotFoundError, TypeError:
            self._logger.debug(f"No valid schema file provided ({self.schema_file}).")
            return {}

    def get_site(self, from_input_meta=False):
        """Return the site name from collected (or, if requested, first input) metadata."""
        try:
            source = self.input_metadata[0] if from_input_meta else self.top_level_meta
            _site = source[self.observatory]["instrument"]["site"]
            if _site is not None:
                return names.validate_site_name(_site)
        except KeyError, TypeError, IndexError:
            pass
        return None

    def _fill_contact_meta(self, contact_dict):
        """Fill contact fields; the user name defaults to the system user."""
        contact_dict["name"] = contact_dict.get("name") or self.args_dict.get("user_name")
        if contact_dict["name"] is None:
            try:
                contact_dict["name"] = getpass.getuser()
            except Exception as exc:  # pylint: disable=broad-except
                contact_dict["name"] = "UNKNOWN_USER"
                self._logger.warning(f"Failed to get user name: {exc}, setting it to UNKNOWN_USER ")
        for key, arg in (
            ("email", "user_email"),
            ("orcid", "user_orcid"),
            ("organization", "user_organization"),
        ):
            contact_dict[key] = contact_dict.get(key) or self.args_dict.get(arg)

    def _fill_context_meta(self, context_dict):
        """Fill context fields with product metadata from input data."""
        input_metadata = (
            self.input_metadata if isinstance(self.input_metadata, list) else [self.input_metadata]
        )
        for metadata in input_metadata:
            try:  # some input files have no product metadata
                if not isinstance(metadata, Mapping):
                    raise TypeError
                metadata = dict(metadata)
                reduced = {
                    key: value
                    for key, value in metadata[self.observatory]["product"].items()
                    if key
                    in {"description", "id", "creation_time", "valid", "format", "filename", "data"}
                }
                activity_name = metadata[self.observatory].get("activity", {}).get("name")
                if activity_name:
                    reduced["activity_name"] = activity_name
                context_dict["associated_data"] = self._fill_context_sim_list(
                    context_dict["associated_data"], reduced
                )
            except KeyError, TypeError:
                self._logger.debug("No input product metadata appended to associated data.")

        associated_activities = self.args_dict.get("associated_activities")
        if associated_activities is not None and "associated_activities" in context_dict:
            context_dict["associated_activities"] = deepcopy(associated_activities)

        for entry in self.args_dict.get("associated_data") or []:
            if "associated_data" in context_dict:
                context_dict["associated_data"] = self._fill_context_sim_list(
                    context_dict["associated_data"], entry
                )

        self._append_context_note_from_value_table(context_dict)
        self._fill_application_configuration(context_dict)

    def _fill_application_configuration(self, context_dict):
        """Add the resolved, sanitized application configuration to the context."""
        if "application_configuration" not in context_dict:
            return
        arguments = {
            key: value
            for key, value in self.args_dict.items()
            if not key.startswith("_") and not _is_local_path_argument(key, value)
        }
        context_dict["application_configuration"] = {
            "schema_version": "2.0.0",
            "application": self.args_dict.get("application_label") or self.args_dict.get("label"),
            "arguments": self._sanitize_configuration(arguments),
            "sources": dict(self.args_dict.get("_metadata_configuration_sources", {})),
        }

    @classmethod
    def _sanitize_configuration(cls, value, key=None):
        """Convert configuration values to serializable types and redact secrets."""
        if key is not None and _SENSITIVE_KEY.search(key):
            return "***REDACTED***" if value is not None else None
        if isinstance(value, dict):
            return {
                str(item_key): cls._sanitize_configuration(item_value, str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._sanitize_configuration(item) for item in value]
        if isinstance(value, str):
            return _SENSITIVE_ASSIGNMENT.sub(_redact_assignment, value)
        return ascii_handler.to_builtin(value)

    def _append_context_note_from_value_table(self, context_dict):
        """Append a context note from the table metadata of the value file, if available."""
        value = self.args_dict.get("value")
        if "notes" not in context_dict or not isinstance(value, str):
            return
        value_path = Path(value)
        if not value_path.exists() and self.args_dict.get("data_path"):
            value_path = Path(self.args_dict.get("data_path")) / value
        if not value.lower().endswith(".ecsv") or not value_path.exists():
            return
        try:
            table_meta = Table.read(value_path, format="ascii.ecsv").meta
        except OSError, KeyError, AttributeError, ValueError:
            return
        context_from_simtel = table_meta.get("context_from_sim_telarray")
        if context_from_simtel in (None, "", [], {}):
            return
        note = {
            "title": "context_from_sim_telarray",
            "text": str(context_from_simtel),
            "creation_time": gen.now_date_time_in_isoformat(),
        }
        context_dict["notes"] = self._fill_context_sim_list(context_dict.get("notes"), note)

    def _read_input_metadata_from_file(self, metadata_file_name_expression=None):
        """Read and validate input metadata from file(s); None if no file is given."""
        try:
            metadata_files = gen.resolve_file_patterns(
                metadata_file_name_expression
                or self.args_dict.get("input_meta")
                or self.args_dict.get("input")
            )
        except ValueError:
            self._logger.debug("No input metadata file defined.")
            return None

        metadata = []
        for metadata_file in metadata_files:
            self._logger.debug(f"Reading meta data from {metadata_file}")
            suffix = Path(metadata_file).suffix
            product = next(
                (
                    p
                    for p in _FORMAT_READERS
                    if Path(metadata_file).name.endswith(product_suffixes(p))
                ),
                None,
            )
            if suffix in (".yaml", ".yml", ".json"):
                input_metadata = self._read_input_metadata_from_yml_or_json(metadata_file)
            elif suffix == ".ecsv":
                input_metadata = self._read_input_metadata_from_ecsv(metadata_file)
            elif product:
                input_metadata = self._format_input_metadata(product, metadata_file)
            else:
                raise ValueError(f"Unknown metadata file format: {metadata_file}")
            schema.validate_dict_using_schema(input_metadata, schema_file=METADATA_JSON_SCHEMA)
            metadata.append(
                value_conversion.change_dict_keys_case(
                    normalize_observatory_key(input_metadata), lower_case=True
                )
            )
        return metadata

    def _format_input_metadata(self, product_id, metadata_file_name):
        """Build schema-compatible input metadata for a simulation file from its reader."""
        fields = importlib.import_module(_FORMAT_READERS[product_id]).read_metadata(
            metadata_file_name
        )
        instrument_id = fields.get("instrument_id")
        format_name = fields["format_name"]
        metadata = metadata_model.get_default_metadata_dict(observatory=self.observatory)
        cta = metadata[self.observatory]
        cta["product"].update({"filename": str(metadata_file_name), "format": format_name})
        cta["product"]["data"].update({"category": "SIM", "type": "Event"})
        cta["instrument"].update(_instrument_fields(fields.get("site"), instrument_id))
        cta["process"]["type"] = "simulation"
        software = {"name": fields["software_name"], "version": fields["software_version"]}
        cta["activity"].update({"type": "software", "software": software})
        raw = json.dumps(ascii_handler.to_builtin(fields["raw_metadata"]), sort_keys=True)
        cta["context"]["notes"] = [{"title": f"{format_name} metadata", "text": raw}]
        if fields.get("associated_elements"):
            cta["context"]["associated_elements"] = fields["associated_elements"]
        return metadata

    def _read_input_metadata_from_ecsv(self, metadata_file_name):
        """Read input metadata from ecsv file."""
        try:
            table_meta = Table.read(metadata_file_name).meta
        except FileNotFoundError, AttributeError:
            self._logger.error(
                "Failed reading metadata for %s from %s", self.observatory, metadata_file_name
            )
            raise
        for name in (self.observatory, LEGACY_OBSERVATORY_KEY):
            for key in (name.upper(), name):
                if key in table_meta:
                    return {key: table_meta[key]}
        raise KeyError(f"No {self.observatory} metadata block in {metadata_file_name}")

    def _read_input_metadata_from_yml_or_json(self, metadata_file_name):
        """Read input metadata from yml or json file."""
        try:
            _input_metadata = ascii_handler.collect_data_from_file(file_name=metadata_file_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Failed reading metadata from {metadata_file_name}") from exc
        _json_type_metadata = {"Metadata", "metadata", "METADATA"}.intersection(_input_metadata)
        if len(_json_type_metadata) > 1:
            raise ValueError(f"More than one metadata entry found in {metadata_file_name}")
        if _json_type_metadata:
            _input_metadata = _input_metadata[_json_type_metadata.pop()]
        return _input_metadata

    def _fill_product_meta(self, product_dict):
        """
        Fill data product fields.

        Data model fields (name, version, url) come from the schema file if one is known;
        otherwise they are left empty (no placeholder values). The product description is
        the description of the application (if known), otherwise the schema description.
        """
        self.schema_file = self.get_data_model_schema_file_name()
        self.schema_dict = self.get_data_model_schema_dict()
        model = product_dict["data"]["model"]
        product_dict["id"] = gen.get_uuid()
        product_dict["creation_time"] = gen.now_date_time_in_isoformat()
        description = self.args_dict.get("_application_description") or self.schema_dict.get(
            "description"
        )
        product_dict["description"] = " ".join(description.split()) if description else None
        product_dict["data"].update({"category": "SIM", "level": "R1", "type": "Service"})
        try:
            product_dict["data"]["association"] = self.schema_dict["instrument"]["class"]
        except KeyError:
            pass
        model["name"] = self.schema_dict.get("name") or self.args_dict.get(
            "metadata_product_data_name"
        )
        model["version"] = self.schema_dict.get("schema_version")
        model["type"] = self.schema_dict.get("meta_schema", None)
        model["url"] = self.schema_file or self.args_dict.get("metadata_product_data_url")
        file_name = self.args_dict.get("output_file")
        product_dict["filename"] = Path(file_name).name if file_name else None
        product_dict["format"] = (
            self.args_dict.get("output_file_format")
            or (Path(file_name).suffix.lstrip(".") if file_name else None)
            or None
        )

    def _fill_instrument_meta(self, instrument_dict):
        """
        Fill instrument fields (site, class, type, id).

        The instrument is taken from 'instrument' or 'telescope'; without either, a site
        is the instrument 'OBS-<site>'.
        """
        site = self.args_dict.get("site")
        instrument_name = self.args_dict.get("instrument") or self.args_dict.get("telescope")
        if instrument_name is None and site is not None:
            instrument_name = f"OBS-{site}"
        instrument_dict.update(_instrument_fields(site, instrument_name))

    def _fill_process_meta(self, process_dict):
        """Fill process fields in metadata."""
        process_dict["type"] = "simulation"

    def _fill_activity_meta(self, activity_dict):
        """Fill activity (software) related metadata."""
        start = self.args_dict.get("activity_start", gen.now_date_time_in_isoformat())
        activity_dict.update(
            {
                "name": self.args_dict.get("label") or self.context.activity.activity_name,
                "type": "software",
                "id": self.args_dict.get("activity_id") or self.context.activity.activity_id,
                "start": start,
                "end": self.args_dict.get("activity_end", start),
            }
        )
        activity_dict["software"].update(
            {"name": "simtools", "version": simtools.version.__version__}
        )
        if "runtime_environment" in activity_dict:
            activity_dict["runtime_environment"] = deepcopy(
                self.args_dict.get("runtime_environment")
            )

    def _merge_config_dicts(self, dict_high, dict_low, add_new_fields=False):
        """Merge dict_low into dict_high (dict_high wins; None values are replaced)."""
        try:
            for k in dict_low:
                if k not in dict_high:
                    if add_new_fields:
                        dict_high[k] = dict_low[k]
                elif isinstance(dict_low[k], dict):
                    self._merge_config_dicts(dict_high[k], dict_low[k], add_new_fields)
                elif dict_high[k] is None:
                    dict_high[k] = dict_low[k]
                elif dict_low[k] is not None and dict_high[k] != dict_low[k]:
                    self._logger.debug(
                        f"Conflicting entries between dict: {dict_high[k]} vs {dict_low[k]} "
                        f"(use {dict_high[k]})"
                    )
        except TypeError as exc:
            raise TypeError("Error merging dictionaries") from exc

    def _fill_context_sim_list(self, meta_list, new_entry_dict):
        """Add an entry to a list-type metadata field (an all-None first entry is replaced)."""
        if not new_entry_dict:
            return []
        if not meta_list:
            return [new_entry_dict]
        if self._all_values_none(meta_list[0]):
            meta_list[0] = new_entry_dict
        else:
            meta_list.append(new_entry_dict)
        return meta_list

    def _all_values_none(self, input_dict):
        """Check recursively if all values in a dictionary are None."""
        if not isinstance(input_dict, dict):
            return input_dict is None
        return all(self._all_values_none(value) for value in input_dict.values())

    def clean_meta_data(self, meta_dict):
        """Remove None values and empty lists/dicts from a metadata dictionary."""
        cleaned = {}
        for key, value in meta_dict.items():
            if value in (None, []):
                continue
            if isinstance(value, dict):
                value = self.clean_meta_data(value)
            elif isinstance(value, list):
                value = [self.clean_meta_data(v) if isinstance(v, dict) else v for v in value]
                value = [v for v in value if v not in (None, "", [], {})]
            if not isinstance(value, (dict, list)) or value:
                cleaned[key] = value
        return cleaned


def write_sidecar_metadata(data_file, args_dict=None, context=None):
    """
    Write the metadata sidecar (``*.meta.yml``) with the SHA-256 checksum of a data product.

    Use for written products that cannot embed their metadata (e.g., yml, json). The
    arguments default to those of the run context. Returns the sidecar file name.
    """
    data_file = Path(data_file)
    context = context or RunContext.from_environment()
    args = args_dict if args_dict is not None else context.args
    metadata = MetadataCollector(
        args_dict={
            **args,
            "output_file": data_file.name,
            "output_file_format": data_file.suffix.lstrip("."),
        },
        context=context,
    )
    metadata.set_product_checksum(product_checksum.sha256_of_file(data_file))
    return metadata.write(data_file)


def _instrument_fields(site, instrument_name):
    """
    Return the known site, class, type and id of an instrument.

    Array elements (e.g. LSTN-01) give class 'telescope' or 'calibration_device' and their
    array element type (e.g. LSTN). Site names and 'OBS-<site>' give class 'site'. Other
    names (e.g. array configurations) give class 'array'.
    """
    instrument_class = instrument_type = None
    if instrument_name:
        try:
            collection = names.get_collection_name_from_array_element_name(instrument_name, False)
        except ValueError:
            collection = None
        if collection == "sites":
            instrument_class = "site"
            instrument_type = names.get_site_from_array_element_name(instrument_name)
            site = site or instrument_type
        elif collection in _INSTRUMENT_CLASSES:
            instrument_class = _INSTRUMENT_CLASSES[collection]
            instrument_type = names.get_array_element_type_from_name(instrument_name)
            derived_site = names.get_site_from_array_element_name(instrument_name)
            site = site or (derived_site if isinstance(derived_site, str) else None)
        else:
            instrument_class = "array"
    fields = {"site": site, "class": instrument_class, "type": instrument_type}
    fields["id"] = instrument_name or None
    return {key: value for key, value in fields.items() if value is not None}


def _data_model_url(schema_file):
    """
    Return the URL of a data model schema for the metadata (None for local files).

    Packaged schema files map to their URL in the simtools schema repository; remote
    URLs are returned unchanged. Other local paths are not written to the metadata.
    """
    if schema_file is None:
        return None
    text = str(schema_file)
    if text.startswith(("http://", "https://")):
        return text
    try:
        relative = Path(text).resolve().relative_to(Path(str(SCHEMA_PATH)).resolve())
    except ValueError:
        return None
    return f"{SCHEMA_URL}/{relative.as_posix()}"


def _is_local_path_argument(key, value):
    """Return True for arguments that refer to a local file system location."""
    if key in _LOCAL_PATH_ARGUMENTS:
        return True
    values = value if isinstance(value, (list, tuple)) else [value]
    return any(isinstance(item, str) and Path(item).is_absolute() for item in values)


def _redact_assignment(match):
    """Redact the value of a sensitive assignment, keeping key, separator and quotes."""
    value = match.group("value")
    quote = value[0] if value[0] in "\"'" else ""
    redacted = f"{quote}***REDACTED***{quote}" if quote else "***REDACTED***"
    return f"{match.group('option') or ''}{match.group('key')}{match.group('separator')}{redacted}"
