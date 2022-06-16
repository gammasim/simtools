import datetime
import logging
import os
from pathlib import Path
import uuid

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.data_model as data_model
import simtools.util.general as gen
import simtools.util.validate_schema as vs
from simtools.util import names
import simtools.version


class WorkflowDescription:
    """
    Workflow description, configuration and metadata class

    Attributes
    ----------
    label: str
        workflow (activity) name
    args: argparse.Namespace
        command line parameters

    Methods
    -------
    collect_workflow_configuration()
        Collect configuration parameters from command line or file
    collect_product_meta_data()
        Collect product meta data information and add activity information
    get_configuration_parameter()
        Returns workflow configuration parameter (entry of CONFIGURATION:key)
    set_configuration_parameter()
        Sets workflow configuration parameter (entry of CONFIGURATION:key)
    label()
        Return workflow name
    product_data_directory()
        Return product data directory
    product_data_file_format()
        Return product data file format
    product_data_file_name()
        Return product data file name
    reference_data_columns()
        Return reference data columns expected in input data


    """

    def __init__(self,
                 label=None,
                 args=None):
        """
        Initialize workflow configuration class

        Parameters
        ----------
        label: str
            workflow label
        args: argparse.Namespace
            command line parameters

        """

        self._logger = logging.getLogger(__name__)

        self.args = args
        self.workflow_config = self._default_workflow_config()
        self.workflow_config['ACTIVITY']['NAME'] = label
        self.workflow_config['ACTIVITY']['ID'] = str(uuid.uuid4())

        if self.args:
            self.collect_workflow_configuration()

        self.top_level_meta = data_model.top_level_reference_schema()

        if self.args:
            self.collect_product_meta_data()

    def collect_workflow_configuration(self):
        """
        Collect configuration parameters from command
        line and/or configuration file and fill it into
        workflow configuration dict.

        Priority is given to arguments given through the
        command line (if they are given in both workflow
        configuration file and command line), with the
        exception of 'None' values.

        """

        self._read_workflow_configuration(
            self._from_args('workflow_config_file'))

        self.workflow_config['INPUT']['METAFILE'] = \
            self._from_args(
                'input_meta_file',
                self.workflow_config['INPUT']['METAFILE'])

        self.workflow_config['INPUT']['DATAFILE'] = \
            self._from_args(
                'input_data_file',
                self.workflow_config['INPUT']['DATAFILE'])

        for arg in vars(self.args):
            self.workflow_config['CONFIGURATION'][str(arg)] = getattr(self.args, arg)

        if self.workflow_config['CONFIGURATION']['configFile']:
            cfg.setConfigFileName(self.workflow_config['CONFIGURATION']['configFile'])

    def collect_product_meta_data(self):
        """
        Collect product meta data and verify the
        schema

        """

        self._fill_top_level_meta_from_args()

        if self.workflow_config['INPUT']['METAFILE']:
            self._fill_top_level_meta_from_file()

        self._fill_product_meta()
        self._fill_product_association_identifier()
        self._fill_activity_meta()

    def label(self):
        """
        Return workflow name
        (often set as label)

        Returns
        -------
        label str
           activity name

        """

        return self.workflow_config['ACTIVITY']['NAME']

    def set_configuration_parameter(self, key, value):
        """
        Set value of workflow configuration parameter.

        Raises
        ------
        KeyError
            if CONFIGURATION does not exist in workflow

        """
        try:
            self.workflow_config['CONFIGURATION'][key] = value
        except KeyError:
            self._logger.error("Missing key {} in CONFIGURATION".format(key))
            raise

    def get_configuration_parameter(self, key):
        """
        Return value of workflow configuration parameter.

        Returns
        -------
        configuration  value
           value of CONFIGURATION parameter

        Raises
        ------
        KeyError
            if CONFIGURATION does not exist in workflow

        """

        try:
            return self.workflow_config['CONFIGURATION'][key]
        except KeyError:
            self._logger.error("Missing key {} in CONFIGURATION".format(key))
            raise

    def reference_data_columns(self):
        """
        Return reference data column definition expected
        in input data

        Returns
        -------
        DATA_COLUMNS dict
            reference data columns

        Raises
        ------
        KeyError
            if DATA_COLUMNS does not exist in workflow
            configuration

        """

        try:
            return self.workflow_config["DATA_COLUMNS"]
        except KeyError:
            self._logger.error(
                "Missing DATA_COLUMNS entry in workflow configuration")
            raise

    def product_data_file_name(self, suffix=None):
        """
        Return full path and name of product data file

        file name is determined by:
        a. Top-level meta ['PRODUCT']['DATA']
        b. Top-level meta ['PRODUCT']['ID'] + label

        File name always used CTA:PRODUCT:ID for unique identification
        (not applied when CONFIGURATION:test is true)

        Parameters
        ----------
        suffix str
           file name extension (if none: use product_data_file_format()


        Returns
        -------
        Path
            data file path and name

        Raises
        ------
        KeyError
            if data file name is not defined in workflow configuration
            or in product metadata dict

        """

        _directory = self.product_data_directory()
        try:
            if self.workflow_config['CONFIGURATION']['test']:
                _filename = 'TEST'
            else:
                _filename = self.workflow_config['ACTIVITY']['ID']
            if self.workflow_config['PRODUCT']['FILENAME']:
                _filename += '-' + self.workflow_config['PRODUCT']['FILENAME']
            else:
                _filename += '-' + self.workflow_config['ACTIVITY']['NAME']
        except KeyError:
            self._logger.error("Missing CTA:PRODUCT:ID in metadata")
            raise
        except TypeError:
            self._logger.error("Missing ACTIVITY:NAME in metadata")
            raise

        if not suffix:
            suffix = '.' + self.product_data_file_format(suffix=True)

        return Path(_directory).joinpath(_filename+suffix)

    def product_data_file_format(self, suffix=False):
        """
        Return file format for data file

        Parameter
        ---------
        suffix: bool
            return just the ecsv suffix (if format is ascii.ecsv)
            return file format (if false)

        Returns
        -------
        str
            file format of data product; default file format is 'ascii.ecsv'

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata
            dictionary

        """

        _file_format = "ascii.ecsv"
        try:
            _file_format = self.workflow_config['PRODUCT']['FORMAT']
        except KeyError:
            self._logger.info(
                "Using default file format for model file: ascii.ecsv")

        if suffix and _file_format == "ascii.ecsv":
            _file_format = "ecsv"

        return _file_format

    def product_data_directory(self):
        """
        Return output directory for data products.
        Create directory if necessary.

        Output directory is determined following this sorted
        list:
        1. PRODUCT:DIRECTORY is set (e.g., through workflow file)
        2. gammasim-tools output location

        Returns
        -------
        path
            output directory for data products

        Raises
        ------
        KeyError
            if data file name is not defined in workflow configuration
            or in product metadata dict

        """

        _output_label = self.workflow_config['ACTIVITY']['NAME']

        if self.workflow_config['PRODUCT']['DIRECTORY']:
            path = Path(self.workflow_config['PRODUCT']['DIRECTORY'])
            path.mkdir(parents=True, exist_ok=True)
            _output_dir = path.absolute()
        else:
            _output_dir = cfg.get("outputLocation")

        _output_dir = io.getApplicationOutputDirectory(
            _output_dir, _output_label)

        self._logger.info("Outputdirectory {}".format(_output_dir))
        return _output_dir

    def _from_args(self, key, default_return=None):
        """
        Return argparser argument.
        No errors raised if argument does not exist

        """

        try:
            return self.args.__dict__[key]
        except KeyError:
            pass

        return default_return

    def _fill_top_level_meta_from_args(self):
        """
        Fill metadata available through command line into top-level template

        Raises
        ------
        KeyError
            if metadata description cannot be filled

        """

        try:
            _association = {}
            _association['SITE'] = self.args.site
            _split_telescope_name = self.args.telescope.split("-")
            _association['CLASS'] = _split_telescope_name[0]
            _association['TYPE'] = _split_telescope_name[1]
            _association['SUBTYPE'] = _split_telescope_name[2]
            self.top_level_meta['CTA']['CONTEXT']['SIM']['ASSOCIATION'][0] = _association
        except KeyError:
            self._logger.error("Error reading user input meta data from args")
            raise
        except AttributeError as e:
            self._logger.debug(
                'Missing parameter on command line, use defaults ({})'.format(e))
            pass

    def _fill_top_level_meta_from_file(self):
        """
        Read and validate user-provided metadata from file.
        Fill metadata into top-level template.

        Raises
        ------
        KeyError
            if corresponding fields cannot by accessed in the
            user top-level or user metadata dictionaries

        """

        _schema_validator = vs.SchemaValidator()
        _user_meta = _schema_validator.validate_and_transform(
            self.workflow_config['INPUT']['METAFILE'])

        try:
            self.top_level_meta['CTA']['CONTACT'] = _user_meta['CONTACT']
            self.top_level_meta['CTA']['INSTRUMENT'] = _user_meta['INSTRUMENT']
            self.top_level_meta['CTA']['PRODUCT']['DESCRIPTION'] = \
                _user_meta['PRODUCT']['DESCRIPTION']
            self.top_level_meta['CTA']['PRODUCT']['CREATION_TIME'] = \
                _user_meta['PRODUCT']['CREATION_TIME']
            if 'VALID' in _user_meta['PRODUCT']:
                if 'START' in _user_meta['PRODUCT']['VALID']:
                    self.top_level_meta['CTA']['PRODUCT']['VALID']['START'] = \
                        _user_meta['PRODUCT']['VALID']['START']
                if 'END' in _user_meta['PRODUCT']['VALID']:
                    self.top_level_meta['CTA']['PRODUCT']['VALID']['END'] = \
                        _user_meta['PRODUCT']['VALID']['END']
            self.top_level_meta['CTA']['PROCESS'] = _user_meta['PROCESS']
            self.top_level_meta['CTA']['CONTEXT']['SIM']['ASSOCIATION'] = \
                _user_meta['PRODUCT']['ASSOCIATION']
            try:
                self.top_level_meta['CTA']['CONTEXT']['SIM']['DOCUMENT'] = \
                    _user_meta['CONTEXT']['DOCUMENT']
            except KeyError:
                pass
        except KeyError:
            self._logger.error("Error reading user input meta data")
            raise

        try:
            self.workflow_config['PRODUCT']['FILENAME'] = os.path.splitext(
                _user_meta['PRODUCT']['DATA'])[0]
        except KeyError:
            pass

    def _fill_product_meta(self):
        """
        Fill metadata for data product

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata
            dictionary

        """

        self.top_level_meta['CTA']['PRODUCT']['ID'] = \
            self.workflow_config['ACTIVITY']['ID']
        self._logger.debug("Assigned ACTIVITE UUID {}".format(
            self.top_level_meta['CTA']['PRODUCT']['ID']))

        try:
            self.top_level_meta['CTA']['PRODUCT']['FORMAT'] = \
                self.product_data_file_format()
        except KeyError:
            self._logger.error("Missing CTA:PRODUCT:FORMAT key in user input meta data")
            raise

    def _fill_product_association_identifier(self):
        """
        Fill list of associations in top-level data model

        Raises
        ------
        KeyError
            if CONTEXT:SIM:ASSOCIATION is not found

        """

        try:
            for association in self.top_level_meta['CTA']['CONTEXT']['SIM']['ASSOCIATION']:
                association['ID'] = names.simtoolsInstrumentName(
                    association['SITE'],
                    association['CLASS'],
                    association['TYPE'],
                    association['SUBTYPE'])
        except KeyError:
            self._logger.error('Error reading CONTEXT:SIM:ASSOCIATION')
            raise

    def _fill_activity_meta(self):
        """
        Fill activity (software) related meta data

        Raises
        ------
        KeyError
            if relevant fields are not defined in top level metadata
            dictionary

        """
        try:
            self.top_level_meta['CTA']['ACTIVITY']['NAME'] = \
                self.workflow_config['ACTIVITY']['NAME']
            self.top_level_meta['CTA']['ACTIVITY']['START'] = \
                datetime.datetime.now().isoformat(timespec='seconds')
            self.top_level_meta['CTA']['ACTIVITY']['END'] = \
                self.top_level_meta['CTA']['ACTIVITY']['START']
            self.top_level_meta['CTA']['ACTIVITY']['SOFTWARE']['VERSION'] = \
                simtools.version.__version__
        except KeyError:
            self._logger.error("Error ACTIVITY meta from user input meta data")
            raise

    def _read_workflow_configuration(self, workflow_config_file):
        """
        Read configuration parameter file and return it as a single dict
        (simplifies processing)

        Parameters
        ----------
        workflow_config_file
            configuration file describing this workflow

        Returns
        -------
        workflow_config: dict
            workflow configuration

        """

        if workflow_config_file:
            try:
                _workflow_from_file = gen.collectDataFromYamlOrDict(
                    workflow_config_file, None)['CTASIMPIPE']
                self._logger.debug("Reading workflow configuration from {}".format(
                    workflow_config_file))
            except KeyError:
                self._logger.debug("Error reading CTASIMPIPE workflow configuration")

            self._merge_config_dicts(self.workflow_config, _workflow_from_file)

    def _merge_config_dicts(self, dict_high, dict_low):
        """
        Merge two config dicts and replace values which are Nonetype.
        Priority to dict_high in case of conflicting entries.


        """

        for k in dict_low:
            if k in dict_high:
                if isinstance(dict_low[k], dict):
                    self._merge_config_dicts(dict_high[k], dict_low[k])
                elif dict_high[k] is None:
                    dict_high[k] = dict_low[k]
                elif dict_high[k] != dict_low[k] and dict_low[k] is not None:
                    self._logger.debug("Conflicting entries between dict: {} vs {}".format(
                        dict_high[k], dict_low[k]))
            else:
                dict_high[k] = dict_low[k]

    def user_input_data_file_name(self):
        """
        Return user input data file
        (full path)
        """

        try:
            return self.workflow_config['INPUT']['DATAFILE']
        except KeyError:
            self._logger.error("Missing description of INPUT:DATAFILE")
            raise

    @staticmethod
    def _default_workflow_config():
        """
        Setup default dictionary for workflow config

        CONFIGURATION collects all argparse argument

        """

        return {
            'REFERENCE': {
                'VERSION': '0.1.0'
            },
            'ACTIVITY': {
                'NAME': None,
                'ID': None,
                'DESCRIPTION': None,
            },
            'DATAMODEL': {
                'USERINPUTSCHEMA': None,
            },
            'INPUT': {
                'METAFILE': None,
                'DATAFILE': None,
            },
            'PRODUCT': {
                'DIRECTORY': None,
                'FILENAME': None,
            },
            'CONFIGURATION': {
                'configFile': './config.yml',
                'logLevel': 'INFO',
                'test': False,
            }
        }
