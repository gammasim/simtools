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
    Workflow configuration and metadata class

    Methods
    -------
    collect_configuration(args)
        Collect configuration parameters

    """

    def __init__(self,
                 label=None,
                 args=None,
                 toplevel_meta={}):
        """
        Initialize workflow configuration class

        Parameters
        ----------
        args: argparse.Namespace
            command line parameters
        toplevel_meta: dict
            top-level metadata definition
            (default behaviour: read from template file)

        """

        self._logger = logging.getLogger(__name__)
        self.workflow_config = {}
        self.label = label

        self.product_data_dir = None
        self.product_data_filename = None

        self.input_meta_file = None
        self.input_data_file = None

        self.toplevel_meta = toplevel_meta

        self.collect_configuration(args)
        self.collect_product_meta_data(args)

    def collect_configuration(self, args):
        """
        Collect configuration parameters from command
        line and configuration file

        Parameters
        -----------
        args: argparse.Namespace
            command line parameters

        """

        if not args:
            return

        self._read_workflow_configuration(args.workflow_config_file)
        self._set_reference_schema_directory(args.reference_schema_directory)

        if args.product_data_directory:
            self.product_data_dir = Path(args.product_data_directory).absolute()

        try:
            self.input_meta_file = args.input_meta_file
        except AttributeError:
            pass
        try:
            self.input_data_file = args.input_data_file
        except AttributeError:
            pass

        self.toplevel_meta = self._get_toplevel_template()

    def collect_product_meta_data(self, args):
        """
        Collect product meta data and verify the
        schema

        Parameters
        -----------
        args: argparse.Namespace
            command line parameters

        """

        self._fill_toplevel_meta_from_args(args)

        if self.input_meta_file:
            self._fill_toplevel_meta_from_file()

        self._fill_product_meta()
        self._fill_activity_meta()

    def reference_data_columns(self):
        """
        Return reference data column definition

        Returns
        -------

        """

        try:
            return self.workflow_config["CTASIMPIPE"]["DATA_COLUMNS"]
        except KeyError:
            self._logger.error(
                "Missing CTASIMPIPE:DATA_COLUMNS entry in workflow configuration")
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
        str
            data file path and name

        Raises
        ------
        KeyError
            if data file name is not defined in workflow configuration
            or in product metadata dict

        """

        _directory = self.product_data_directory()
        try:
            _filename = self.toplevel_meta['CTA']['PRODUCT']['ID']
            if self.product_data_filename:
                _filename += '-' + self.product_data_filename
            else:
                _filename += '-' + self.label
        except KeyError:
            self._logger.error(
                "Missing CTA:PRODUCT:ID in metadata")
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
            file format of data product

        Raises
        ------
        KeyError
            if relevant fields are not defined in toplevel metadata
            dictionary

        """

        _file_format = "ascii.ecsv"
        try:
            _file_format = self.workflow_config['CTASIMPIPE']['PRODUCT']['FORMAT']
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

        _output_label = self.label

        if self.product_data_dir:
            path = Path(self.product_data_dir)
            path.mkdir(parents=True, exist_ok=True)
            _output_dir = path.absolute()
        else:
            _output_dir = io.getApplicationOutputDirectory(
                cfg.get("outputLocation"),
                _output_label)

        self._logger.info("Outputdirectory {}".format(_output_dir))
        return _output_dir

    def _fill_toplevel_meta_from_args(self, args):
        """
        Fill metadata available through command line into top-level template

        Parameters
        -----------
        args: argparse.Namespace
            command line parameters

        """

        try:
            _association = {}
            _association['SITE'] = args.site
            _split_telescope_name = args.telescope.split("-")
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

        _schema_validator = vs.SchemaValidator(self.workflow_config)
        _user_meta = _schema_validator.validate_and_transform(
            self.input_meta_file)

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
            self.product_data_filename = os.path.splitext(
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

        self.toplevel_meta['CTA']['PRODUCT']['ID'] = str(uuid.uuid4())
        self._logger.debug("Issued UUID {}".format(
            self.toplevel_meta['CTA']['PRODUCT']['ID']))

        try:
            self.toplevel_meta['CTA']['PRODUCT']['FORMAT'] = \
                self.product_data_file_format()
        except KeyError:
            self._logger.error("Error PRODUCT meta from user input meta data")
            raise

        self._fill_product_association_identifier()

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
                self.workflow_config['CTASIMPIPE']['ACTIVITY']['NAME']
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

        self.workflow_config = gen.collectDataFromYamlOrDict(
            workflow_config_file, None)
        self._logger.debug("Reading workflow configuration from {}".format(
            workflow_config_file))

    def _set_reference_schema_directory(self, reference_schema_directory=None):
        """
        Set reference schema directory

        Parameters
        ----------
        reference_schema_directory
            directory to reference schema

        Raises
        ------
        KeyError
            workflow configuration does not include SCHEMADIRECTORY key


        """

        if reference_schema_directory:
            try:
                self.workflow_config['CTASIMPIPE']['DATAMODEL']['SCHEMADIRECTORY'] = \
                    reference_schema_directory
            except KeyError as error:
                self._logger.error("Workflow configuration incomplete")
                raise KeyError from error

    def _get_toplevel_template(self):
        """
        Read toplevel data model template from file

        Returns
        -------
        dict
            top-level meta data definition

        """

        _toplevel_metadata_file = self._read_toplevel_metadata_file()
        if _toplevel_metadata_file:
            self._logger.debug(
                "Reading top-level metadata template from {}".format(
                    _toplevel_metadata_file))
            return gen.collectDataFromYamlOrDict(
                _toplevel_metadata_file, None)
        return None

    def _read_toplevel_metadata_file(self):
        """
        Return full path and name of top level data model schema file

        Raises
        ------
        KeyError
            if relevant fields are not defined in toplevel metadata
            dictionary
        TypeError
            if workflow directory cannot be converted to type str

        """
        try:
            return str(
                self.workflow_config['CTASIMPIPE']['DATAMODEL']['SCHEMADIRECTORY']
                + '/' +
                self.workflow_config['CTASIMPIPE']['DATAMODEL']['TOPLEVELMODEL'])
        except (KeyError, TypeError):
            self._logger.error(
                "Missing description of DATAMODEL:SCHEMADIRECTORY/TOPLEVELMODEL")
            raise
