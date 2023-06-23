import datetime
import logging
import uuid
from pathlib import Path

import simtools.util.general as gen
import simtools.version
from simtools import io_handler
from simtools.data_model import meta_data_model, validate_schema
from simtools.util import names

__all__ = ["WorkflowDescription"]


class WorkflowDescription:
    """
    Workflow description, configuration and metadata class.
    Assigns uuid to workflow in ACIVITY:ID

    Parameters
    ----------
    args: argparse.Namespace
        Command line parameters

    """

    def __init__(self, args_dict):
        """
        Initialize workflow configuration.
        """

        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()

        self.args_dict = args_dict
        self.workflow_config = meta_data_model.workflow_configuration_schema()
        self.workflow_config["activity"]["name"] = args_dict["label"]
        self.workflow_config["activity"]["id"] = str(uuid.uuid4())

        self._read_workflow_configuration(self.args_dict.get("workflow_config", None))
        self.workflow_config["configuration"] = self.args_dict

        self.top_level_meta = gen.change_dict_keys_case(
            meta_data_model.top_level_reference_schema(), True
        )
        self.collect_product_meta_data()

    def collect_product_meta_data(self):
        """
        Collect and verify product metadata.

        """

        self._fill_association_meta_from_args(
            self.top_level_meta["cta"]["context"]["sim"]["association"]
        )

        self._fill_product_meta(self.top_level_meta["cta"]["product"])

        self._fill_top_level_meta_from_file(self.top_level_meta["cta"])

        self._fill_association_id(self.top_level_meta["cta"]["context"]["sim"]["association"])

        self._fill_activity_meta(self.top_level_meta["cta"]["activity"])

    def set_configuration_parameter(self, key, value):
        """
        Set value of workflow configuration parameter.

        Parameters
        ----------
        key: (str,required)
            Key of the workflow configuration dict.
        value: (str,required)
            Value of the workflow configuration dict associated to 'key'.

        Raises
        ------
        KeyError
            if configuration does not exist in workflow.

        """
        try:
            self.workflow_config["configuration"][key] = value
        except KeyError:
            self._logger.error(f"Missing key {key} in configuration")
            raise

    def get_configuration_parameter(self, key):
        """
        Return value of workflow configuration parameter.

        Parameters
        ----------
        key: (str,required)
            Key of the workflow configuration dict.

        Returns
        -------
        configuration  value
           value of configuration parameter

        Raises
        ------
        KeyError
            if configuration does not exist in workflow
        """

        try:
            return self.workflow_config["configuration"][key]
        except KeyError:
            self._logger.error(f"Missing key {key} in configuration")
            raise

    def reference_data_columns(self):
        """
        Return reference data column definition expected in input data.

        Returns
        -------
        data_columns dict
            Reference data columns

        Raises
        ------
        KeyError
            if data_columns does not exist in workflow configuration.

        """

        try:
            return self.workflow_config["data_columns"]
        except KeyError:
            self._logger.error("Missing data_columns entry in workflow configuration")
            print(self.workflow_config)
            raise

    def product_data_file_name(self, suffix=None, full_path=True):
        """
        Return name of product data file.

        File name is the combination of activity id (or 'TEST' if CONFIGURATION:TEST is set) and:
        a. Top-level meta ['product']['name']
        or
        b. Top-level meta ['activity']['name']

        (depending which one is set)

        Parameters
        ----------
        suffix: str
            file name extension (if None: use product_data_file_format())
        full_path: bool
            if True: return path + file name, otherwise file name only.

        Returns
        -------
        Path
            data file path and name

        Raises
        ------
        KeyError
            if data file name is not defined in workflow configuration or in product metadata dict.
        TypeError
            if activity:name and product:filename is None.

        """

        try:
            if self.workflow_config["configuration"]["test"]:
                _filename = "TEST"
            else:
                _filename = self.workflow_config["activity"]["id"]
            if self.workflow_config["product"]["filename"]:
                _filename += "-" + self.workflow_config["product"]["filename"]
            else:
                _filename += "-" + self.workflow_config["activity"]["name"]
        except KeyError:
            self._logger.error("Missing cta:product:id in metadata")
            raise
        except TypeError:
            self._logger.error("Missing activity:name in metadata")
            raise

        if not suffix:
            suffix = "." + self.product_data_file_format(suffix=True)

        if full_path:
            return Path(self.product_data_directory()).joinpath(_filename + suffix)
        return Path(_filename + suffix)

    def product_data_file_format(self, suffix=False):
        """
        Return file format for product data.

        Parameters
        ----------
        suffix: bool
            Return the ecsv suffix (if format is ascii.ecsv),
            Return file format (if false)

        Returns
        -------
        str
            File format of data product; default file format is 'ascii.ecsv'.

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata dictionary.
        """

        _file_format = "ascii.ecsv"
        try:
            if self.workflow_config["product"]["format"] is not None:
                _file_format = self.workflow_config["product"]["format"]
        except KeyError:
            self._logger.info("Using default file format for model file: ascii.ecsv")

        if suffix and _file_format == "ascii.ecsv":
            _file_format = "ecsv"

        return _file_format

    def product_data_directory(self):
        """
        Output directory for data products.

        Returns
        -------
        path
            output directory for data products

        """

        _output_dir = self.io_handler.get_output_directory(
            self.workflow_config["activity"]["name"], "product-data"
        )
        self._logger.debug(f"Outputdirectory {_output_dir}")
        return _output_dir

    def _fill_association_meta_from_args(self, association_dict):
        """
        Append association meta data set through configurator.

        Parameters
        ----------
        association_dict: dict
            Dictionary for assocation metadata field.

        Raises
        ------
        AttributeError
            if error reading association meta data from args.
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
        except KeyError:
            self._logger.error("Error reading association meta data from args")
            raise
        except AttributeError as e:
            self._logger.debug(f"Missing parameter on command line, use defaults ({e})")

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

        _schema_validator = validate_schema.SchemaValidator()
        try:
            _input_meta = _schema_validator.validate_and_transform(
                meta_file_name=self.workflow_config["configuration"]["input_meta"],
                lower_case=True,
            )
        except KeyError:
            self._logger.debug("No input metadata file defined")
            return

        try:
            self._merge_config_dicts(top_level_dict, _input_meta)
        except KeyError:
            self._logger.error("Error reading input meta data")
            raise
        # list entry copies
        for association in _input_meta["product"]["association"]:
            self._fill_context_sim_list(
                top_level_dict["context"]["sim"]["association"], association
            )
        try:
            for document in _input_meta["context"]["document"]:
                self._fill_context_sim_list(top_level_dict["context"]["sim"]["document"], document)
        except KeyError:
            top_level_dict["context"]["sim"].pop("document")

    def _fill_product_meta(self, product_dict):
        """
        Fill metadata for data products fields.

        Parameters
        ----------
        product_dict: dict
            Dictionary describing data product.

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata dictionary.

        """

        product_dict["id"] = self.workflow_config["activity"]["id"]
        self._logger.debug(f"Assigned ACTIVITY UUID {product_dict['id']}")

        product_dict["data"]["category"] = "SIM"
        product_dict["data"]["level"] = "R0"
        product_dict["data"]["type"] = "service"
        product_dict["data"]["model"]["name"] = "simpipe-table"
        product_dict["data"]["model"]["version"] = "0.1.0"
        product_dict["format"] = self.product_data_file_format()
        product_dict["filename"] = str(self.product_data_file_name(full_path=False))

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
        Fill activity (software) related meta data

        Parameters
        ----------
        activity_dict: dict
            Dictionary for top-level activitiy meta data.

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata
            dictionary

        """
        try:
            activity_dict["name"] = self.workflow_config["activity"]["name"]
            activity_dict["start"] = datetime.datetime.now().isoformat(timespec="seconds")
            activity_dict["end"] = activity_dict["start"]
            activity_dict["software"]["name"] = "simtools"
            activity_dict["software"]["version"] = simtools.version.__version__
        except KeyError:
            self._logger.error("Error ACTIVITY meta from input meta data")
            raise

    def _read_workflow_configuration(self, workflow_config_file):
        """
        Read workflow configuration from file and merge it with existing workflow config. Keys are \
         changed to lower case.

        Parameters
        ----------
        workflow_config_file
            name of configuration file describing this workflow.

        """

        if workflow_config_file:
            try:
                _workflow_from_file = gen.change_dict_keys_case(
                    gen.collect_data_from_yaml_or_dict(workflow_config_file, None)["CTASIMPIPE"],
                    True,
                )
                self._logger.debug(f"Reading workflow configuration from {workflow_config_file}")
            except KeyError:
                self._logger.debug("Error reading CTASIMPIPE workflow configuration")

            self._merge_config_dicts(self.workflow_config, _workflow_from_file, True)

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
            if missing description of CONFIGURATON:INPUT_DATA
        """

        try:
            return self.workflow_config["configuration"]["input_data"]
        except KeyError:
            self._logger.error("Missing description of CONFIGURATON:INPUT_DATA")
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
