"""
Metadata collector for simtools.

This should be the only module in simtools with knowledge on the
implementation of the metadata model.

"""
import datetime
import getpass
import logging
from pathlib import Path

from astropy.table import Table

import simtools.constants
import simtools.utils.general as gen
import simtools.version
from simtools.data_model import metadata_model
from simtools.io_operations import io_handler

__all__ = ["MetadataCollector"]


class MetadataCollector:
    """
    Collects and combines metadata associated to describe the current
    simtools activity and its data products. Collect as much metadata
    as possible from command line configuration, input data, environment,
    schema descriptions.
    Depends on the CTAO top-level metadata definition.

    Parameters
    ----------
    args_dict: dict
        Command line parameters
    metadata_file_name: str
        Name of metadata file (only required when args_dict is None)
    data_model_name: str
        Name of data model parameter

    """

    def __init__(self, args_dict, metadata_file_name=None, data_model_name=None):
        """
        Initialize metadata collector.

        """

        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()

        self.args_dict = args_dict if args_dict else {}
        self.data_model_name = data_model_name
        self.schema_file = None
        self.schema_dict = None
        self.top_level_meta = gen.change_dict_keys_case(
            data_dict=metadata_model.get_default_metadata_dict(), lower_case=True
        )
        self.input_metadata = self._read_input_metadata_from_file(
            metadata_file_name=metadata_file_name
        )
        self.collect_meta_data()

    def collect_meta_data(self):
        """
        Collect and verify product metadata from different sources.

        """

        self._fill_contact_meta(self.top_level_meta["cta"]["contact"])
        self._fill_product_meta(self.top_level_meta["cta"]["product"])
        self._fill_activity_meta(self.top_level_meta["cta"]["activity"])
        self._fill_process_meta(self.top_level_meta["cta"]["process"])
        self._fill_context_from_input_meta(self.top_level_meta["cta"]["context"])
        self._fill_associated_elements_from_args(
            self.top_level_meta["cta"]["context"]["associated_elements"]
        )

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
        try:
            if self.args_dict["schema"]:
                self._logger.debug(f"Schema file from command line: {self.args_dict['schema']}")
                return self.args_dict["schema"]
        except KeyError:
            pass

        # from metadata
        try:
            if self.top_level_meta["cta"]["product"]["data"]["model"]["url"]:
                self._logger.debug(
                    "Schema file from product metadata: "
                    f"{self.top_level_meta['cta']['product']['data']['model']['url']}"
                )
                return self.top_level_meta["cta"]["product"]["data"]["model"]["url"]
        except KeyError:
            pass

        # from data model name
        if self.data_model_name:
            self._logger.debug(f"Schema file from data model name: {self.data_model_name}")
            return f"{simtools.constants.SCHEMA_URL}{self.data_model_name}.schema.yml"

        # from input metadata
        try:
            self._logger.debug(
                "Schema file from input metadata: "
                f"{self.input_metadata['cta']['product']['data']['model']['url']}"
            )
            return self.input_metadata["cta"]["product"]["data"]["model"]["url"]
        except KeyError:
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
            return gen.collect_data_from_file_or_dict(file_name=self.schema_file, in_dict=None)
        except gen.InvalidConfigData:
            self._logger.debug(f"No valid schema file provided ({self.schema_file}).")
        return {}

    def _fill_contact_meta(self, contact_dict):
        """
        Fill contact metadata fields.

        Parameters
        ----------
        contact_dict: dict
            Dictionary for contact metadata fields.

        """

        if contact_dict.get("name", None) is None:
            contact_dict["name"] = getpass.getuser()

    def _fill_associated_elements_from_args(self, associated_elements_dict):
        """
        Append association metadata set through configurator.

        Note
        ----
        This function might go in future, as instrument
        information will not be given via command line.

        Parameters
        ----------
        associated_elements_dict: dict
            Dictionary for associated elements field.

        Raises
        ------
        TypeError, KeyError
            if error reading association metadata from args.
        KeyError
            if metadata description cannot be filled.

        """
        self._logger.debug(f"Fill metadata from args: {self.args_dict}")

        _association = {}

        try:
            if "site" in self.args_dict:
                _association["site"] = self.args_dict["site"]
            if "telescope" in self.args_dict:
                _split_telescope_name = self.args_dict["telescope"].split("-")
                _association["class"] = _split_telescope_name[0]
                _association["type"] = _split_telescope_name[1]
                _association["subtype"] = _split_telescope_name[2]
        except (TypeError, KeyError):
            self._logger.error("Error reading association metadata from args")
            raise

        self._fill_context_sim_list(associated_elements_dict, _association)

    def _fill_context_from_input_meta(self, context_dict):
        """
        Read and validate input metadata from file and fill CONTEXT metadata fields.

        Parameters
        ----------
        context_dict: dict
            Dictionary with context level metadata.

        Raises
        ------
        KeyError
            if corresponding fields cannot by accessed in the top-level or metadata dictionaries.

        """

        try:
            self._merge_config_dicts(context_dict, self.input_metadata["cta"]["context"])
            for key in ("document", "associated_elements", "associated_data"):
                self._copy_list_type_metadata(context_dict, self.input_metadata["cta"], key)
        except KeyError:
            self._logger.debug("No context metadata defined in input metadata file.")

        try:
            self._fill_context_sim_list(
                context_dict["associated_data"], self.input_metadata["cta"]["product"]
            )
        except (KeyError, TypeError):
            self._logger.debug("No input product metadata appended to associated data.")

    def _read_input_metadata_from_file(self, metadata_file_name=None, observatory="CTA"):
        """
        Read and validate input metadata from file. In case of an ecsv file including a
        table, the metadata is read from the table meta data. Returns empty dict in case
        no file is given.

        Parameter
        ---------
        metadata_file_name: str or Path
            Name of metadata file.
        observatory: str
            Observatory name.

        Returns
        -------
        dict
            Metadata dictionary.

        Raises
        ------
        gen.InvalidConfigData, FileNotFoundError
            if metadata cannot be read from file.
        KeyError:
            if metadata does not exist for the given observatory.

        """

        try:
            metadata_file_name = (
                self.args_dict.get("input_meta", None)
                if metadata_file_name is None
                else metadata_file_name
            )
        except TypeError:
            pass

        if metadata_file_name is None:
            self._logger.debug("No input metadata file defined.")
            return {}

        # metadata from yml or json file
        if Path(metadata_file_name).suffix in (".yaml", ".yml", ".json"):
            try:
                self._logger.debug("Reading meta data from %s", metadata_file_name)
                _input_metadata = gen.collect_data_from_file_or_dict(
                    file_name=metadata_file_name, in_dict=None
                )
                _json_type_metadata = {"Metadata", "metadata", "METADATA"}.intersection(
                    _input_metadata
                )
                if len(_json_type_metadata) == 1:
                    _input_metadata = _input_metadata[_json_type_metadata.pop()]
                elif len(_json_type_metadata) > 1:
                    self._logger.error(
                        "More than one metadata entry found in %s", metadata_file_name
                    )
                    raise gen.InvalidConfigData
            except (gen.InvalidConfigData, FileNotFoundError):
                self._logger.error("Failed reading metadata from %s", metadata_file_name)
                raise
        # metadata from table meta in ecsv file
        elif Path(metadata_file_name).suffix == ".ecsv":
            try:
                _input_metadata = {observatory: Table.read(metadata_file_name).meta[observatory]}
            except (FileNotFoundError, KeyError):
                self._logger.error(
                    "Failed reading metadata for %s from %s", observatory, metadata_file_name
                )
                raise
        else:
            self._logger.error("Unknown metadata file format: %s", metadata_file_name)
            raise gen.InvalidConfigData

        metadata_model.validate_schema(_input_metadata, None)

        return gen.change_dict_keys_case(
            self._process_metadata_from_file(_input_metadata),
            lower_case=True,
        )

    def _fill_product_meta(self, product_dict):
        """
        Fill metadata for data products fields. If a schema file is given for the data products,
        try and read product:data:model metadata from there.

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

        product_dict["id"] = self.args_dict.get("activity_id", "UNDEFINED_ACTIVITY_ID")
        product_dict["creation_time"] = datetime.datetime.now().isoformat(timespec="seconds")
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
        helper_dict = {"name": "name", "version": "version", "type": "base_schema"}
        for key, value in helper_dict.items():
            product_dict["data"]["model"][key] = self.schema_dict.get(value, None)
        product_dict["data"]["model"]["url"] = self.schema_file

        product_dict["format"] = self.args_dict.get("output_file_format", None)
        product_dict["filename"] = str(self.args_dict.get("output_file", None))

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
        Fill activity (software) related metadata

        Parameters
        ----------
        activity_dict: dict
            Dictionary for top-level activity metadata.

        """

        activity_dict["name"] = self.args_dict.get("label", None)
        activity_dict["type"] = "software"
        activity_dict["id"] = self.args_dict.get("activity_id", "UNDEFINED_ACTIVITY_ID")
        activity_dict["start"] = datetime.datetime.now().isoformat(timespec="seconds")
        activity_dict["end"] = activity_dict["start"]
        activity_dict["software"]["name"] = "simtools"
        activity_dict["software"]["version"] = simtools.version.__version__

    def _merge_config_dicts(self, dict_high, dict_low, add_new_fields=False):
        """
        Merge two config dicts and replace values in dict_high which are Nonetype. Priority to \
         dict_high in case of conflicting entries.

        Parameters
        ----------
        dict_high: dict
            Dictionary into which values are merged.
        dict_low: dict
            Dictionary from which values are taken for merging.
        add_new_fields: bool
            If true: add fields from dict_low to dict_high, if they don't exist in dict_high

        """

        if dict_high is None and dict_low:
            dict_high = dict_low
            return

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
        except (KeyError, TypeError):
            self._logger.error("Error merging dictionaries")
            raise

    def _fill_context_sim_list(self, meta_list, new_entry_dict):
        """
        Fill list-type entries into metadata. Take into account the first list entry is the default
        value filled with Nones.

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

        if len(new_entry_dict) == 0:
            return []
        try:
            if self._all_values_none(meta_list[0]):
                meta_list[0] = new_entry_dict
            else:
                meta_list.append(new_entry_dict)
        except (TypeError, IndexError):
            meta_list = [new_entry_dict]
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
            meta_dict["cta"]["product"]["description"] = self._remove_line_feed(
                meta_dict["cta"]["product"]["description"]
            )
        except (KeyError, AttributeError):
            pass

        return meta_dict

    @staticmethod
    def _remove_line_feed(string):
        """
        Remove all line feeds from a string

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
                self._fill_context_sim_list(context_dict[key], document)
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
            True if all entries are None, False otherwise.

        """

        if not isinstance(input_dict, dict):
            return input_dict is None

        return all(self._all_values_none(value) for value in input_dict.values())
