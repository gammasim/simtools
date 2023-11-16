import datetime
import logging
from pathlib import Path

import simtools.utils.general as gen
import simtools.version
from simtools import io_handler
from simtools.data_model import metadata_model
from simtools.utils import names

__all__ = ["MetadataCollector"]


class MetadataCollector:
    """
    Collects and combines metadata associated with the current activity
    (e.g., the execution of an application).
    Depends on and fine tuned to CTAO top-level metadata definition.

    Parameters
    ----------
    args: argparse.Namespace
        Command line parameters

    """

    def __init__(self, args_dict):
        """
        Initialize metadata collector.

        """

        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()

        self.args_dict = args_dict
        self.top_level_meta = gen.change_dict_keys_case(
            data_dict=metadata_model.top_level_reference_schema(), lower_case=True
        )
        self.collect_product_meta_data()

    def collect_product_meta_data(self):
        """
        Collect and verify product metadata from different sources.

        """

        self._fill_association_meta_from_args(
            self.top_level_meta["cta"]["context"]["associated_elements"]
        )
        self._fill_product_meta(self.top_level_meta["cta"]["product"])
        self._fill_top_level_meta_from_file(self.top_level_meta["cta"])
        self._fill_association_id(self.top_level_meta["cta"]["context"]["associated_elements"])
        self._fill_activity_meta(self.top_level_meta["cta"]["activity"])

    def _fill_association_meta_from_args(self, association_dict):
        """
        Append association metadata set through configurator.

        Parameters
        ----------
        association_dict: dict
            Dictionary for association metadata field.

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

        self._fill_context_sim_list(association_dict, _association)

    def _fill_top_level_meta_from_file(self, top_level_dict):
        """
        Read and validate metadata from file. Fill metadata into top-level template.

        Parameters
        ----------
        top_level_dict: dict
            Dictionary for top level metadata.

        Raises
        ------
        KeyError
            if corresponding fields cannot by accessed in the top-level or metadata dictionaries.

        """

        if self.args_dict.get("input_meta", None) is None:
            self._logger.debug("Skipping metadata reading; no metadata file defined.")
            return

        try:
            self._logger.debug(f"Reading meta data from {self.args_dict['input_meta']}")
            _input_meta = gen.collect_data_from_yaml_or_dict(
                in_yaml=self.args_dict.get("input_meta", None), in_dict=None
            )
        except gen.InvalidConfigData:
            self._logger.debug("Failed reading metadata from file.")
            raise

        metadata_model.validate_schema(_input_meta, None)
        _input_meta = self._process_metadata_from_file(_input_meta)

        self._merge_config_dicts(top_level_dict, _input_meta["cta"])
        for key in ("document", "associated_elements"):
            self._copy_metadata_context_lists(top_level_dict, _input_meta["cta"], key)

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

        product_dict["id"] = self.args_dict.get("activity_id", "UNDEFINED_ACTIVITY_ID")
        self._logger.debug(f"Reading activity UUID {product_dict['id']}")

        product_dict["data"]["category"] = "SIM"
        product_dict["data"]["level"] = "R1"
        product_dict["data"]["type"] = "service"

        _schema_dict = self._collect_schema_dict()
        product_dict["data"]["model"]["name"] = _schema_dict.get("name", "simpipe-schema")
        product_dict["data"]["model"]["version"] = _schema_dict.get("version", "0.0.0")
        product_dict["format"] = self.args_dict.get("output_file_format", None)
        product_dict["filename"] = str(self.args_dict.get("output_file", None))

    def _collect_schema_dict(self):
        """
        Read schema from file.

        The schema configuration parameter points to a directory or a file.
        For the case of a directory, the schema file is assumed to be named
        <parameter_name>.schema.yml.

        Returns
        -------
        dict
            Dictionary containing schema metadata.

        """

        _schema = self.args_dict.get("schema", "")
        if Path(_schema).is_dir():
            try:
                _data_dict = gen.collect_data_from_yaml_or_dict(
                    in_yaml=self.args_dict.get("input", None), in_dict=None, allow_empty=True
                )
                return gen.collect_dict_from_file(
                    file_path=_schema,
                    file_name=f"{_data_dict['name']}.schema.yml",
                )
            except (TypeError, KeyError):
                return {}
        return gen.collect_dict_from_file(_schema)

    @staticmethod
    def _fill_association_id(association_dict):
        """
        Fill association id from site and telescope class, type, subtype.

        Parameters
        ----------
        association_dict: dict
            Association dictionary.

        """
        for association in association_dict:
            try:
                association["id"] = names.simtools_instrument_name(
                    site=association["site"],
                    telescope_class_name=association["class"],
                    sub_system_name=association["type"],
                    telescope_id_name=association.get("subtype", "D"),
                )
            except ValueError:
                association["id"] = None

    def _fill_activity_meta(self, activity_dict):
        """
        Fill activity (software) related metadata

        Parameters
        ----------
        activity_dict: dict
            Dictionary for top-level activity metadata.

        """

        activity_dict["name"] = self.args_dict.get("label", None)
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

    def input_data_file_name(self):
        """
        Return input data file (full path).

        Returns
        -------
        str
            Input data file (full path).

        Raises
        ------
        KeyError
            if missing description of INPUT_DATA
        """

        try:
            return self.args_dict["input_data"]
        except KeyError:
            self._logger.error("Missing description of INPUT_DATA")
            raise

    @staticmethod
    def _fill_context_sim_list(product_list, new_entry_dict):
        """
        Fill list-type entries into metadata. Take into account the first list entry is the default
        value filled with Nones.

        Returns
        -------
        list
            Updated product list.

        """

        if len(new_entry_dict) == 0:
            return []
        try:
            if any(v is not None for v in product_list[0].values()):
                product_list.append(new_entry_dict)
            else:
                product_list[0] = new_entry_dict
        except (TypeError, IndexError):
            product_list = [new_entry_dict]
        return product_list

    def _process_metadata_from_file(self, meta_dict):
        """
        Process metadata from file to ensure compatibility
        with metadata model.

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
        except KeyError:
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

        return string.replace("\n", " ").replace("\r", "")

    def _copy_metadata_context_lists(self, top_level_dict, _input_meta, key):
        """
        Copy list-type metadata from file.
        Very fine tuned.

        Parameters
        ----------
        top_level_dict: dict
            Dictionary for top level metadata.
        meta_dict: dict
            Dictionary for metadata from file.
        key: str
            Key for metadata entry.

        """

        try:
            for document in _input_meta["context"][key]:
                self._fill_context_sim_list(top_level_dict["context"][key], document)
        except KeyError:
            top_level_dict["context"].pop(key)
