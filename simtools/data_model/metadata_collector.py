import datetime
import logging
import os

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
    args_dict: Dictionary
        Command line parameters
    data_model_name: str
        Name of simulation model parameter

    """

    def __init__(self, args_dict, data_model_name=None):
        """
        Initialize metadata collector.

        """

        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()

        self.args_dict = args_dict
        self.data_model_name = data_model_name
        self.top_level_meta = gen.change_dict_keys_case(
            data_dict=metadata_model.get_default_metadata_dict(), lower_case=True
        )
        self.input_meta = self._read_input_meta_from_file()
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

    def get_data_model_schema(self):
        """
        Return name of schema file and schema dict.
        The schema file name is taken (in this order) from the command line,
        from the metadata file, from the data model name, or from the input
        metadata file.

        Returns
        -------
        dict
            Schema dictionary.
        str
            Name of schema file.

        """

        _schema_file = self.args_dict.get("schema", None)
        if _schema_file is not None:
            self._logger.info(f"From command line: {_schema_file}")
            return _schema_file

        try:
            _schema_file = self.top_level_meta["cta"]["product"]["data"]["model"]["url"]
            if _schema_file is None:
                raise TypeError
            self._logger.info(f"From metadata: {_schema_file}")
            return _schema_file
        except (KeyError, TypeError):
            pass

        # TODO - questionable that this is hardwired
        if self.data_model_name is not None:
            _schema_file = (
                "https://raw.githubusercontent.com/gammasim/workflows/main/schemas/"
                + self.data_model_name
                + ".schema.yml"
            )
            self._logger.info(f"From data model name: {_schema_file}")
        else:
            try:
                _schema_file = self.input_meta["cta"]["product"]["data"]["model"]["url"]
                self._logger.info(f"From input meta data : {_schema_file}")
            except KeyError:
                _schema_file = None

        try:
            return (
                gen.collect_data_from_yaml_or_dict(in_yaml=_schema_file, in_dict=None),
                _schema_file,
            )
        except gen.InvalidConfigData:
            self._logger.debug(f"Failed reading schema file from {_schema_file}.")
        return {}, None

    def _fill_contact_meta(self, contact_dict):
        """
        Fill contact metadata fields.

        Parameters
        ----------
        contact_dict: dict
            Dictionary for contact metadata fields.

        """

        if contact_dict.get("name", None) is None:
            contact_dict["name"] = os.getlogin()

    def _fill_associated_elements_from_args(self, associated_elements_dict):
        """
        Append association metadata set through configurator.

        TODO - this function might go in future, as instrument
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
            self._merge_config_dicts(context_dict, self.input_meta["cta"]["context"])
            for key in ("document", "associated_elements", "associated_data"):
                self._copy_list_type_metadata(context_dict, self.input_meta["cta"], key)
        except KeyError:
            self._logger.debug("No context metadata defined in input metadata file.")

        self._fill_context_sim_list(
            context_dict["associated_data"], self.input_meta["cta"]["product"]
        )

    def _read_input_meta_from_file(self):
        """
        Read and validate input metadata from file.

        Returns
        -------
        dict
            Metadata dictionary.

        Raises
        ------
        gen.InvalidConfigData
            if metadata cannot be read from file.

        """

        if self.args_dict is None or self.args_dict.get("input_meta", None) is None:
            self._logger.debug("No input metadata file defined.")
            return {}

        try:
            self._logger.debug(f"Reading meta data from {self.args_dict['input_meta']}")
            _input_meta = gen.collect_data_from_yaml_or_dict(
                in_yaml=self.args_dict["input_meta"], in_dict=None
            )
        except gen.InvalidConfigData:
            self._logger.error("Failed reading metadata from file.")
            raise

        metadata_model.validate_schema(_input_meta, None)

        return self._process_metadata_from_file(_input_meta)

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

        _schema_dict, _schema_file = self.get_data_model_schema()

        product_dict["id"] = self.args_dict.get("activity_id", "UNDEFINED_ACTIVITY_ID")
        product_dict["creation_time"] = datetime.datetime.now().isoformat(timespec="seconds")
        product_dict["description"] = _schema_dict.get("description", None)

        # DATA:CATEGORY
        product_dict["data"]["category"] = "SIM"
        product_dict["data"]["level"] = "R1"
        product_dict["data"]["type"] = "service"
        # TODO - introduce consistent naming of DL3 data model and model parameter schema files
        try:
            product_dict["data"]["association"] = _schema_dict["instrument"]["class"]
        except KeyError:
            pass

        # DATA:MODEL
        product_dict["data"]["model"]["name"] = _schema_dict.get("name", None)
        product_dict["data"]["model"]["version"] = _schema_dict.get("version", None)
        product_dict["data"]["model"]["url"] = _schema_file
        product_dict["data"]["model"]["type"] = _schema_dict.get("base_schema", None)

        product_dict["format"] = self.args_dict.get("output_file_format", None)
        product_dict["filename"] = str(self.args_dict.get("output_file", None))

    def _fill_process_meta(self, process_dict):
        """
        Fill metadata for process fields.

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

        return string.replace("\n", " ").replace("\r", "").replace("  ", " ")

    def _copy_list_type_metadata(self, context_dict, _input_meta, key):
        """
        Copy list-type metadata from file.
        Very fine tuned.

        Parameters
        ----------
        context_dict: dict
            Dictionary for top level metadata (context level)
        _input_meta: dict
            Dictionary for metadata from file.
        key: str
            Key for metadata entry.

        """

        try:
            for document in _input_meta["context"][key]:
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
