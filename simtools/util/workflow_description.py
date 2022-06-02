import datetime
import logging
import os
from pathlib import Path
import uuid

import simtools.config as cfg
import simtools.io_handler as io
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
    toplevel_meta: dict
        top-level metadata definition

    Methods
    -------
    collect_workflow_configuration()
        Collect configuration parameters from command line or file
    collect_product_meta_data()
        Collect product meta data information and add activity information
    configuration(key)
        Returns entry of CONFIGURATION:key
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
    userinput_schema_file_name()
        Return userinput schema file name


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

        self.toplevel_meta = self._collect_toplevel_template()

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

        self.workflow_config['DATAMODEL']['TOPLEVELMODEL'] = \
            self._from_args(
                'toplevel_metadata_schema',
                self.workflow_config['DATAMODEL']['TOPLEVELMODEL']
            )

        self.workflow_config['PRODUCT']['DIRECTORY'] = \
            self._from_args(
                'product_data_directory',
                self.workflow_config['PRODUCT']['DIRECTORY']
            )
        if self.workflow_config['PRODUCT']['DIRECTORY']:
            self.workflow_config['PRODUCT']['DIRECTORY'] = \
                Path(self.workflow_config['PRODUCT']['DIRECTORY']).absolute()

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

    def collect_product_meta_data(self):
        """
        Collect product meta data and verify the
        schema

        """

        self._fill_toplevel_meta_from_args()

        if self.workflow_config['INPUT']['METAFILE']:
            self._fill_toplevel_meta_from_file()

        self._fill_product_meta()
        self._fill_product_association_identifier()
        self._fill_activity_meta()

    def label(self):
        """
        Return workflow name
        (often set as label)

        """

        return self.workflow_config['ACTIVITY']['NAME']

    def configuration(self, key, value=None):
        """
        Set or Return workflow configuration parameter.

        Usually filled from argparser.

        """

        try:
            if value is None:
                return self.workflow_config['CONFIGURATION'][key]
            else:
                self.workflow_config['CONFIGURATION'][key] = value
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
        a. Toplevel meta ['PRODUCT']['DATA']
        b. Toplevel meta ['PRODUCT']['ID'] + label

        File name always used CTA:PRODUCT:ID for unique identification

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
            _filename = self.workflow_config['ACTIVITY']['ID']
            if self.workflow_config['PRODUCT']['FILENAME']:
                _filename += '-' + self.workflow_config['PRODUCT']['FILENAME']
            else:
                _filename += '-' + self.workflow_config['ACTIVITY']['NAME']
        except KeyError:
            self._logger.error(
                "Missing CTA:PRODUCT:ID in metadata")
            print(self.toplevel_meta)
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
            return file suffix (if true)
            return file format (if false)

        Returns
        -------
        str
            file format of data product; default file format is 'ascii.ecsv'

        Raises
        ------
        KeyError
            if relevant fields are not defined in toplevel metadata
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
        1. self.product_data_dir is set (e.g., through command line)
        2. gammasim-tools output location

        Returns
        -------
        str
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
            _output_dir = io.getApplicationOutputDirectory(
                cfg.get("outputLocation"),
                _output_label)

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

    def _fill_toplevel_meta_from_args(self):
        """
        Fill metadata available through command line into top-level template

        Parameters
        ----------
        args: argparse.Namespace
            command line parameters

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
            self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION'][0] = _association
        except KeyError:
            self._logger.error("Error reading user input meta data from args")
            raise
        except AttributeError:
            pass

    def _fill_toplevel_meta_from_file(self):
        """
        Read and validate user-provided metadata from file.
        Fill metadata into top-level template.

        Raises
        ------
        KeyError
            if corresponding fields cannot by accessed in the
            user top-level or user metadata dictionaries

        """

        _schema_validator = vs.SchemaValidator(self.userinput_schema_file_name())
        _user_meta = _schema_validator.validate_and_transform(
            self.workflow_config['INPUT']['METAFILE'])

        try:
            self.toplevel_meta['CTA']['CONTACT'] = _user_meta['CONTACT']
            self.toplevel_meta['CTA']['INSTRUMENT'] = _user_meta['INSTRUMENT']
            self.toplevel_meta['CTA']['PRODUCT']['DESCRIPTION'] = \
                _user_meta['PRODUCT']['DESCRIPTION']
            self.toplevel_meta['CTA']['PRODUCT']['CREATION_TIME'] = \
                _user_meta['PRODUCT']['CREATION_TIME']
            if 'CONTEXT' in _user_meta['PRODUCT']:
                self.toplevel_meta['CTA']['PRODUCT']['CONTEXT'] = \
                    _user_meta['PRODUCT']['CONTEXT']
            if 'VALID' in _user_meta['PRODUCT']:
                if 'START' in _user_meta['PRODUCT']['VALID']:
                    self.toplevel_meta['CTA']['PRODUCT']['VALID']['START'] = \
                        _user_meta['PRODUCT']['VALID']['START']
                if 'END' in _user_meta['PRODUCT']['VALID']:
                    self.toplevel_meta['CTA']['PRODUCT']['VALID']['END'] = \
                        _user_meta['PRODUCT']['VALID']['END']
            self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION'] = \
                _user_meta['PRODUCT']['ASSOCIATION']
            self.toplevel_meta['CTA']['PROCESS'] = _user_meta['PROCESS']
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
            if relevant fields are not defined in toplevel metadata
            dictionary

        """

        self.toplevel_meta['CTA']['PRODUCT']['ID'] = \
            self.workflow_config['ACTIVITY']['ID']
        self._logger.debug("Assigned ACTIVITE UUID {}".format(
            self.toplevel_meta['CTA']['PRODUCT']['ID']))

        try:
            self.toplevel_meta['CTA']['PRODUCT']['FORMAT'] = \
                self.product_data_file_format()
        except KeyError:
            self._logger.error("Error PRODUCT meta from user input meta data")
            raise

    def _fill_product_association_identifier(self):
        """
        Fill list of associations in top-level data model

        Raises
        ------
        KeyError
            if PRODUCT::ASSOCIATION is not found

        """

        try:
            for association in self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION']:
                association['ID'] = self._read_instrument_name(association)
        except KeyError:
            self._logger.error('Error reading PRODUCT:ASSOCIATION')
            raise

    def _read_instrument_name(self, association):
        """
        Returns a string defining the instrument following
        the gammasim-tools naming convention derived from
        PRODUCT:ASSOCIATION entry

        """

        try:
            _instrument = \
                names.validateSiteName(association['SITE']) \
                + "-" + \
                names.validateName(association['CLASS'], names.allTelescopeClassNames) \
                + "-" + \
                names.validateSubSystemName(association['TYPE']) \
                + "-" + \
                names.validateTelescopeIDName(association['SUBTYPE'])
        except KeyError:
            self._logger.error('Error reading PRODUCT:ASSOCIATION')
            raise
        except ValueError:
            self._logger.error('Error reading naming in PRODUCT:ASSOCIATION')
            raise

        return _instrument

    def _fill_activity_meta(self):
        """
        Fill activity (software) related meta data

        Raises
        ------
        KeyError
            if relevant fields are not defined in toplevel metadata
            dictionary

        """
        try:
            self.toplevel_meta['CTA']['ACTIVITY']['NAME'] = \
                self.workflow_config['ACTIVITY']['NAME']
            self.toplevel_meta['CTA']['ACTIVITY']['START'] = \
                datetime.datetime.now().isoformat(timespec='seconds')
            self.toplevel_meta['CTA']['ACTIVITY']['END'] = \
                self.toplevel_meta['CTA']['ACTIVITY']['START']
            self.toplevel_meta['CTA']['ACTIVITY']['SOFTWARE']['VERSION'] = \
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

            self._merge_config_dicts(_workflow_from_file, self.workflow_config)

    def _merge_config_dicts(self, dict1, dict2):
        """
        Merge two config dicts and replace values which are Nonetype.
        Priority to dict2 in case of conflicting entries.

        """

        for k in dict1:
            if k in dict2:
                if isinstance(dict1[k], dict):
                    self._merge_config_dicts(dict1[k], dict2[k])
                elif dict2[k] is None:
                    dict2[k] = dict1[k]
                elif dict2[k] != dict1[k] and dict1[k] is not None:
                    self._logger.debug("Conflicting entries between dict: {} vs {}".format(
                        dict2[k], dict1[k]))
            else:
                dict2[k] = dict1[k]

    def _collect_toplevel_template(self):
        """
        Fill toplevel data model template from schema file

        Returns
        -------
        dict
            top-level meta data template

        """

        _toplevel_meta = None
        try:
            if self.workflow_config['DATAMODEL']['TOPLEVELMODEL']:
                _workflow_config_file = Path(
                    self._read_schema_directory(),
                    self.workflow_config['DATAMODEL']['TOPLEVELMODEL']
                )
                self._logger.debug(
                    "Reading top-level metadata template from {}".format(
                        _workflow_config_file))
                _toplevel_meta = gen.collectDataFromYamlOrDict(
                    _workflow_config_file, None)
        except KeyError:
            self._logger.error('Error reading DATAMODEL:TOPLEVELMODEL')
            raise

        return _toplevel_meta

    def userinput_schema_file_name(self):
        """
        Return user meta file name.
        (full path)
        """

        try:
            if self.workflow_config['DATAMODEL']['USERINPUTSCHEMA']:
                return Path(
                    self._read_schema_directory(),
                    self.workflow_config['DATAMODEL']['USERINPUTSCHEMA'])
        except KeyError:
            self._logger.error("Missing description of DATAMODEL:USERINPUTSCHEMA")
            raise

    def userinput_data_file_name(self):
        """
        Return user input data file
        (full path)
        """

        try:
            return self.workflow_config['INPUT']['DATAFILE']
        except KeyError:
            self._logger.error("Missing description of INPUT:DATAFILE")
            raise

    def _read_schema_directory(self):
        """
        Return directory for metadata schema file


        """

        if 'reference_schema_directory' in self.workflow_config['CONFIGURATION'] \
                and self.workflow_config['CONFIGURATION']['reference_schema_directory']:
            return self.workflow_config['CONFIGURATION']['reference_schema_directory']

        return ""

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
                'TOPLEVELMODEL': None,
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
                'logLevel': 'INFO',
                'test': False,
            }
        }
