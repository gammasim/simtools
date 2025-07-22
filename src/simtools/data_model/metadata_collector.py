"""
Metadata collector for simtools.

This should be the only module in simtools with knowledge on the
implementation of the observatory metadata model.

"""

import getpass
import logging
import uuid
from pathlib import Path

import yaml

import simtools.utils.general as gen
import simtools.version
from simtools.constants import METADATA_JSON_SCHEMA
from simtools.data_model import metadata_model, schema
from simtools.io import ascii_handler, io_handler
from simtools.utils import names

__all__ = ["MetadataCollector"]


class MetadataCollector:
    """
    Collects metadata to describe the current simtools activity and its data products.

    Collect metadata from command line configuration, input data, environment,
    and schema descriptions. Depends on the CTAO top-level metadata definition.

    Two dictionaries store two different types of metadata:

    - top_level_meta: metadata for the current activity
    - input_metadata: metadata from input data

    Parameters
    ----------
    args_dict: dict
        Command line parameters
    metadata_file_name: str
        Name of metadata file (only required when args_dict is None)
    model_parameter_name: str
        Name of model parameter
    observatory: str
        Name of observatory (default: "cta")
    clean_meta: bool
        Clean metadata from None values and empty lists (default: True)
    schema_version: str
        Version of the metadata schema to use (default: 'latest')
    """

    def __init__(
        self,
        args_dict=None,
        metadata_file_name=None,
        model_parameter_name=None,
        observatory="cta",
        clean_meta=True,
        schema_version="latest",
    ):
        """Initialize metadata collector."""
        self._logger = logging.getLogger(__name__)
        self.observatory = observatory
        self.io_handler = io_handler.IOHandler()

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

    def collect_meta_data(self):
        """Collect and verify product metadata for each main-level metadata type."""
        meta_types = self.top_level_meta[self.observatory].keys()
        for meta_type in meta_types:
            try:
                fill_method = getattr(self, f"_fill_{meta_type}_meta")
                fill_method(self.top_level_meta[self.observatory][meta_type])
            except AttributeError:
                self._logger.debug(f"Method _fill_{meta_type}_meta not implemented")

    def get_top_level_metadata(self):
        """
        Return top level metadata dictionary (with updated activity end time).

        Returns
        -------
        dict
            Top level metadata dictionary.

        """
        try:
            self.top_level_meta[self.observatory]["activity"]["end"] = (
                gen.now_date_time_in_isoformat()
            )
        except KeyError:
            pass
        return self.top_level_meta

    @staticmethod
    def dump(args_dict, output_file, add_activity_name=False):
        """
        Write metadata to file (static method).

        Parameters
        ----------
        args_dict: dict
            Command line parameters
        output_file: str or Path
            Name of output file.
        add_activity_name: bool
            Add activity name to file name.
        """
        collector = MetadataCollector(args_dict)
        collector.write(output_file, add_activity_name=add_activity_name)

    def write(self, yml_file=None, keys_lower_case=True, add_activity_name=False):
        """
        Write toplevel metadata to file (yaml file format).

        Parameters
        ----------
        metadata: dict
            Metadata to be stored
        yml_file: str
            Name of output file.
        keys_lower_case: bool
            Write yaml keys in lower case.
        add_activity_name: bool
            Add activity name to file name.

        Returns
        -------
        str
            Name of output file

        Raises
        ------
        FileNotFoundError
            If yml_file not found.
        TypeError
            If yml_file is not defined.
        """
        metadata = self.get_top_level_metadata()
        activity_name = metadata.get("cta", {}).get("activity", {}).get("name", "").rstrip(".")
        suffix = f".{activity_name}.meta.yml" if add_activity_name else ".meta.yml"

        if yml_file is None:
            raise TypeError("No output file for metadata defined")

        try:
            yml_file = names.file_name_with_version(yml_file, suffix)
            with open(yml_file, "w", encoding="UTF-8") as file:
                yaml.safe_dump(
                    gen.change_dict_keys_case(
                        gen.remove_substring_recursively_from_dict(metadata, substring="\n"),
                        keys_lower_case,
                    ),
                    file,
                    sort_keys=False,
                    explicit_start=True,
                )
            self._logger.info(f"Writing metadata to {yml_file}")
            return yml_file
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Error writing metadata to {yml_file}") from exc

    def get_data_model_schema_file_name(self):
        """
        Return data model schema file name.

        The schema file name is taken (in this order) from the command line,
        from the metadata file, from the data model name, or from the input
        metadata file.

        Returns
        -------
        str
            Name of schema file.

        """
        # from command line
        if self.args_dict.get("schema"):
            self._logger.debug(f"Schema file from command line: {self.args_dict['schema']}")
            return self.args_dict["schema"]

        # from metadata
        try:
            url = self.top_level_meta[self.observatory]["product"]["data"]["model"]["url"]
            if url:
                self._logger.debug(f"Schema file from product metadata: {url}")
                return url
        except KeyError:
            pass

        # from model parameter name
        if self.model_parameter_name:
            self._logger.debug(f"Schema file from data model name: {self.model_parameter_name}")
            return str(schema.get_model_parameter_schema_file(self.model_parameter_name))

        # from first entry in input metadata (least preferred)
        try:
            url = self.input_metadata[0][self.observatory]["product"]["data"]["model"]["url"]
            self._logger.debug(f"Schema file from input metadata: {url}")
            return url
        except (KeyError, TypeError, IndexError):
            pass

        self._logger.warning("No schema file found.")
        return None

    def get_data_model_schema_dict(self):
        """
        Return data model schema dictionary.

        Returns
        -------
        dict
            Data model schema dictionary.

        """
        try:
            return ascii_handler.collect_data_from_file(file_name=self.schema_file)
        except TypeError:
            self._logger.debug(f"No valid schema file provided ({self.schema_file}).")
        return {}

    def get_site(self, from_input_meta=False):
        """
        Get site entry from metadata. Allow to get from collected or from input metadata.

        Parameters
        ----------
        from_input_meta: bool
            Get site from first entry of input metadata (default: False)

        Returns
        -------
        str
            Site name

        """
        try:
            _site = (
                self.top_level_meta[self.observatory]["instrument"]["site"]
                if not from_input_meta
                else self.input_metadata[0][self.observatory]["instrument"]["site"]
            )
            if _site is not None:
                return names.validate_site_name(_site)
        except (KeyError, TypeError, IndexError):
            pass
        return None

    def _fill_contact_meta(self, contact_dict):
        """
        Fill contact metadata fields. Get user name from system level if not given.

        Parameters
        ----------
        contact_dict: dict
            Dictionary for contact metadata fields.
        """
        contact_dict["name"] = contact_dict.get("name") or self.args_dict.get("user_name")
        if contact_dict["name"] is None:
            self._logger.warning("No user name provided, take user info from system level.")
            try:
                contact_dict["name"] = getpass.getuser()
            except Exception as exc:  # pylint: disable=broad-except
                contact_dict["name"] = "UNKNOWN_USER"
                self._logger.warning(
                    f"Failed to get user name: {exc}, setting it to {contact_dict['name']} "
                )
        meta_dict = {
            "email": "user_mail",
            "orcid": "user_orcid",
            "organization": "user_organization",
        }
        for key, value in meta_dict.items():
            contact_dict[key] = contact_dict.get(key) or self.args_dict.get(value)

    def _fill_context_meta(self, context_dict):
        """
        Fill context metadata fields with product metadata from input data.

        Parameters
        ----------
        context_dict: dict
            Dictionary for context metadata fields.

        """
        input_metadata = (
            self.input_metadata if isinstance(self.input_metadata, list) else [self.input_metadata]
        )

        for metadata in input_metadata:
            try:  # wide try..except as for some cases we expect that there is no product metadata
                reduced_product_meta = {
                    key: value
                    for key, value in metadata[self.observatory]["product"].items()
                    if key in {"description", "id", "creation_time", "valid", "format", "filename"}
                }
                if metadata[self.observatory].get("activity", {}).get("name"):
                    reduced_product_meta["activity_name"] = metadata[self.observatory][
                        "activity"
                    ].get("name")
                context_dict["associated_data"] = self._fill_context_sim_list(
                    context_dict["associated_data"], reduced_product_meta
                )
            except (KeyError, TypeError):
                self._logger.debug("No input product metadata appended to associated data.")

    def _read_input_metadata_from_file(self, metadata_file_name_expression=None):
        """
        Read and validate input metadata from file.

        In case of an ecsv file including a table, the metadata is read from the table meta data.
        Returns empty dict in case no file is given.

        Parameter
        ---------
        metadata_file_name_expression: str or Path
            Name of metadata file (regular expressions allowed).

        Returns
        -------
        dict
            Metadata dictionary.

        Raises
        ------
        ValueError:
            if metadata cannot be read from file.
        KeyError:
            if metadata does not exist

        """
        metadata_file_names = (
            metadata_file_name_expression
            or self.args_dict.get("input_meta")
            or self.args_dict.get("input")
        )

        try:
            metadata_files = gen.resolve_file_patterns(metadata_file_names)
        except ValueError:
            self._logger.debug("No input metadata file defined.")
            return None

        metadata = []
        for metadata_file in metadata_files:
            self._logger.debug(f"Reading meta data from {metadata_file}")
            if Path(metadata_file).suffix in (".yaml", ".yml", ".json"):
                _input_metadata = self._read_input_metadata_from_yml_or_json(metadata_file)
            elif Path(metadata_file).suffix == ".ecsv":
                _input_metadata = self._read_input_metadata_from_ecsv(metadata_file)
            elif Path(metadata_file).name.endswith((".simtel.zst", ".simtel")):
                self._logger.warning(
                    "Metadata extraction from sim_telarray files is not supported yet."
                )
                continue
            else:
                raise ValueError(f"Unknown metadata file format: {metadata_file}")

            schema.validate_dict_using_schema(_input_metadata, schema_file=METADATA_JSON_SCHEMA)
            metadata.append(gen.change_dict_keys_case(_input_metadata, lower_case=True))

        return metadata

    def _read_input_metadata_from_ecsv(self, metadata_file_name):
        """Read input metadata from ecsv file."""
        from astropy.table import Table  # pylint: disable=C0415

        try:
            return {
                self.observatory.upper(): Table.read(metadata_file_name).meta[
                    self.observatory.upper()
                ]
            }
        except (FileNotFoundError, KeyError, AttributeError) as exc:
            self._logger.error(
                "Failed reading metadata for %s from %s", self.observatory, metadata_file_name
            )
            raise exc

    def _read_input_metadata_from_yml_or_json(self, metadata_file_name):
        """Read input metadata from yml or json file."""
        try:
            _input_metadata = ascii_handler.collect_data_from_file(file_name=metadata_file_name)
            _json_type_metadata = {"Metadata", "metadata", "METADATA"}.intersection(_input_metadata)
            if len(_json_type_metadata) == 1:
                _input_metadata = _input_metadata[_json_type_metadata.pop()]
            if len(_json_type_metadata) > 1:
                raise ValueError(f"More than one metadata entry found in {metadata_file_name}")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Failed reading metadata from {metadata_file_name}") from exc
        return _input_metadata

    def _fill_product_meta(self, product_dict):
        """
        Fill metadata for data products fields.

        If a schema file is given for the data products, try and read product:data:model metadata
        from there.

        Parameters
        ----------
        product_dict: dict
            Dictionary describing data product.

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata dictionary.

        """
        self.schema_file = self.get_data_model_schema_file_name()
        self.schema_dict = self.get_data_model_schema_dict()

        product_dict["id"] = str(uuid.uuid4())
        product_dict["creation_time"] = gen.now_date_time_in_isoformat()
        product_dict["description"] = self.schema_dict.get("description", None)

        # DATA:CATEGORY
        product_dict["data"]["category"] = "SIM"
        product_dict["data"]["level"] = "R1"
        product_dict["data"]["type"] = "Service"
        try:
            product_dict["data"]["association"] = self.schema_dict["instrument"]["class"]
        except KeyError:
            pass

        # DATA:MODEL
        product_dict["data"]["model"]["name"] = (
            self.schema_dict.get("name")
            or self.args_dict.get("metadata_product_data_name")
            or "undefined_model_name"
        )
        product_dict["data"]["model"]["version"] = self.schema_dict.get("schema_version", "0.0.0")
        product_dict["data"]["model"]["type"] = self.schema_dict.get("meta_schema", None)
        product_dict["data"]["model"]["url"] = self.schema_file or self.args_dict.get(
            "metadata_product_data_url"
        )

        product_dict["filename"] = str(self.args_dict.get("output_file", ""))
        product_dict["format"] = (
            self.args_dict.get("output_file_format")
            or Path(product_dict["filename"]).suffix.lstrip(".")
            or None
        )

    def _fill_instrument_meta(self, instrument_dict):
        """
        Fill instrument metadata fields.

        Note inconsistency in command line arguments for 'ID',
        which is either 'instrument' or 'telescope'.

        Parameters
        ----------
        instrument_dict: dict
            Dictionary for instrument metadata fields.

        """
        instrument_dict["site"] = self.args_dict.get("site")
        instrument_dict["ID"] = self.args_dict.get("instrument") or self.args_dict.get("telescope")
        if instrument_dict["ID"]:
            instrument_dict["class"] = names.get_collection_name_from_array_element_name(
                instrument_dict["ID"], False
            )
            instrument_dict["type"] = (
                names.get_array_element_type_from_name(instrument_dict["ID"])
                if not instrument_dict.get("type")
                else instrument_dict["type"]
            )

    def _fill_process_meta(self, process_dict):
        """
        Fill process fields in metadata.

        Parameters
        ----------
        process_dict: dict
            Dictionary for process metadata fields.

        """
        process_dict["type"] = "simulation"

    def _fill_activity_meta(self, activity_dict):
        """
        Fill activity (software) related metadata.

        Parameters
        ----------
        activity_dict: dict
            Dictionary for top-level activity metadata.

        """
        activity_dict["name"] = self.args_dict.get("label", None)
        activity_dict["type"] = "software"
        activity_dict["id"] = self.args_dict.get("activity_id", "UNDEFINED_ACTIVITY_ID")
        activity_dict["start"] = gen.now_date_time_in_isoformat()
        activity_dict["end"] = activity_dict["start"]
        activity_dict["software"]["name"] = "simtools"
        activity_dict["software"]["version"] = simtools.version.__version__

    def _merge_config_dicts(self, dict_high, dict_low, add_new_fields=False):
        """
        Merge two config dicts and replace values in dict_high which are Nonetype.

        Priority to dict_high in case of conflicting entries.

        Parameters
        ----------
        dict_high: dict
            Dictionary into which values are merged.
        dict_low: dict
            Dictionary from which values are taken for merging.
        add_new_fields: bool
            If true: add fields from dict_low to dict_high, if they don't exist in dict_high

        """
        try:
            for k in dict_low:
                if k in dict_high:
                    if isinstance(dict_low[k], dict):
                        self._merge_config_dicts(dict_high[k], dict_low[k], add_new_fields)
                    elif dict_high[k] is None:
                        dict_high[k] = dict_low[k]
                    elif dict_high[k] != dict_low[k] and dict_low[k] is not None:
                        self._logger.debug(
                            f"Conflicting entries between dict: {dict_high[k]} vs {dict_low[k]} "
                            f"(use {dict_high[k]})"
                        )
                elif add_new_fields:
                    dict_high[k] = dict_low[k]
        except TypeError as exc:
            raise TypeError("Error merging dictionaries") from exc

    def _fill_context_sim_list(self, meta_list, new_entry_dict):
        """
        Fill list-type entries into metadata.

        Take into account the first list entry is the default value filled with Nones.

        Parameters
        ----------
        meta_list: list
            List of metadata entries.
        new_entry_dict: dict
            New metadata entry to be added to meta_list.

        Returns
        -------
        list
            Updated meta list.

        """
        if not new_entry_dict:
            return []
        if meta_list is None or not meta_list:
            return [new_entry_dict]
        if self._all_values_none(meta_list[0]):
            meta_list[0] = new_entry_dict
        else:
            meta_list.append(new_entry_dict)
        return meta_list

    def _process_metadata_from_file(self, meta_dict):
        """
        Process metadata from file to ensure compatibility with metadata model.

        Changes keys to lower case and removes line feeds from description fields.

        Parameters
        ----------
        meta_dict: dict
            Input metadata dictionary.

        Returns
        -------
        dict
            Metadata dictionary.

        """
        meta_dict = gen.change_dict_keys_case(meta_dict, True)
        try:
            meta_dict[self.observatory]["product"]["description"] = self._remove_line_feed(
                meta_dict[self.observatory]["product"]["description"]
            )
        except (KeyError, AttributeError):
            pass

        return meta_dict

    @staticmethod
    def _remove_line_feed(string):
        """
        Remove all line feeds from a string.

        Parameters
        ----------
        str
            input string

        Returns
        -------
        str
            with line feeds removed
        """
        return string.replace("\n", " ").replace("\r", "").replace("  ", " ")

    def _copy_list_type_metadata(self, context_dict, _input_metadata, key):
        """
        Copy list-type metadata from file.

        Very fine tuned.

        Parameters
        ----------
        context_dict: dict
            Dictionary for top level metadata (context level)
        _input_metadata: dict
            Dictionary for metadata from file.
        key: str
            Key for metadata entry.

        """
        try:
            for document in _input_metadata["context"][key]:
                context_dict[key] = self._fill_context_sim_list(context_dict[key], document)
        except KeyError:
            pass

    def _all_values_none(self, input_dict):
        """
        Check recursively if all values in a dictionary are None.

        Parameters
        ----------
        input_dict: dict
            Input dictionary.

        Returns
        -------
        bool
            True if all entries are None.

        """
        if not isinstance(input_dict, dict):
            return input_dict is None

        return all(self._all_values_none(value) for value in input_dict.values())

    def clean_meta_data(self, meta_dict):
        """
        Clean metadata dictionary from None values and empty lists.

        Parameters
        ----------
        meta_dict: dict
            Metadata dictionary.

        """

        def clean_list(value):
            nested_list = [
                self.clean_meta_data(item) if isinstance(item, dict) else item for item in value
            ]
            return [item for item in nested_list if item not in (None, "", [], {})]

        cleaned = {}
        for key, value in meta_dict.items():
            if value in (None, []):
                continue
            if isinstance(value, dict):
                nested = self.clean_meta_data(value)
                if nested:  # Only add if not empty
                    cleaned[key] = nested
            elif isinstance(value, list):
                nested_list = clean_list(value)
                if nested_list:  # Only add if not empty
                    cleaned[key] = nested_list
            else:
                cleaned[key] = value
        return cleaned
