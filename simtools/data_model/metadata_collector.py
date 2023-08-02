import datetime
import logging

import simtools.util.general as gen
import simtools.version
from simtools import io_handler
from simtools.data_model import metadata_model, validate_schema
from simtools.util import names

__all__ = ["MetadataCollector"]


class MetadataCollector:
    """
    Collects and combines metadata associated with the current activity
    (e.g., the executation of an application).
    Follows CTAO top-level metadata definition.

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
            metadata_model.top_level_reference_schema(), True
        )
        self.collect_product_meta_data()

    def collect_product_meta_data(self):
        """
        Collect and verify product metadata from different sources.

        """

        self._fill_association_meta_from_args(
            self.top_level_meta["cta"]["context"]["sim"]["association"]
        )

        self._fill_product_meta(self.top_level_meta["cta"]["product"])

        self._fill_top_level_meta_from_file(self.top_level_meta["cta"])

        self._fill_association_id(self.top_level_meta["cta"]["context"]["sim"]["association"])

        self._fill_activity_meta(self.top_level_meta["cta"]["activity"])

    def _fill_association_meta_from_args(self, association_dict):
        """
        Append association metadata set through configurator.

        Parameters
        ----------
        association_dict: dict
            Dictionary for assocation metadata field.

        Raises
        ------
        AttributeError
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

        _schema_validator = validate_schema.SchemaValidator()
        _input_meta = _schema_validator.validate_and_transform(
            meta_file_name=self.args_dict["input_meta"],
        )

        try:
            self._merge_config_dicts(top_level_dict, _input_meta)
        except (KeyError, TypeError):
            self._logger.error("Error reading input metadata")
            raise
        # list entry copies
        for association in _input_meta["product"]["association"]:
            self._fill_context_sim_list(
                top_level_dict["context"]["sim"]["association"], association
            )
        try:
            for document in _input_meta["product"]["document"]:
                self._fill_context_sim_list(top_level_dict["context"]["sim"]["document"], document)
        except KeyError:
            top_level_dict["context"]["sim"].pop("document")

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
        self._logger.debug(f"Reading activitiy UUID {product_dict['id']}")

        product_dict["data"]["category"] = "SIM"
        product_dict["data"]["level"] = "R0"
        product_dict["data"]["type"] = "service"
        _schema_dict = (
            gen.collect_data_from_yaml_or_dict(
                in_yaml=self.args_dict.get("schema", None), in_dict=None, allow_empty=True
            )
            or {}
        )
        product_dict["data"]["model"]["name"] = _schema_dict.get("name", "simpipe-schema")
        product_dict["data"]["model"]["version"] = _schema_dict.get("version", "0.0.0")
        product_dict["format"] = self.args_dict.get("output_file_format", None)
        product_dict["filename"] = str(self.args_dict.get("output_file", None))

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
                    association["site"],
                    association["class"],
                    association["type"],
                    association["subtype"],
                )
            except ValueError:
                association["id"] = None

    def _fill_activity_meta(self, activity_dict):
        """
        Fill activity (software) related metadata

        Parameters
        ----------
        activity_dict: dict
            Dictionary for top-level activitiy metadata.

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata
            dictionary

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
        Fill list-type entries into metadata. Take into account the first list entry is the default\
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
