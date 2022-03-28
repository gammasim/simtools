import datetime
import logging
import os
import uuid
import yaml

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen
import simtools.util.names as names
import simtools.version


class ModelData:
    """
    Simulation model data class.

    Includes metadata enrichment and model data writing.

    Attributes:
    -----------
    workflow_config: dict
        workflow configuration
    toplevel_meta: dict
        top-level meta data definition (default: read from template file)

    Methods:
    --------
    write_model_file()
        Write model and model metadata to file.

    """

    def __init__(self, workflow_config=None, toplevel_meta=None):
        """
        Initialize model data

        """

        self._logger = logging.getLogger(__name__)

        self.workflow_config = workflow_config
        if toplevel_meta:
            self.toplevel_meta = toplevel_meta
        else:
            self.toplevel_meta = self._get_toplevel_template()
        self._user_meta = None
        self._user_data = None

    def write_model_file(self, user_meta, user_data):
        """
        Write model data consisting of metadata and data files

        Parameters
        ----------
        user_meta: dict
            User meta data.
        user_data: astropy Table
            Model data.

        """

        self._user_meta = user_meta
        self._user_data = user_data

        self._prepare_metadata()
        self._write_metadata()
        self._write_data()

    def _get_toplevel_template(self):
        """
        Read and return toplevel data model template

        """

        _toplevel_metadata_file = self._read_toplevel_metadata_file()
        if _toplevel_metadata_file:
            self._logger.debug(
                "Reading top-level metadata template from {}".format(
                    _toplevel_metadata_file))
            return gen.collectDataFromYamlOrDict(
                _toplevel_metadata_file,
                None)
        return None

    def _prepare_metadata(self):
        """
        Prepare metadata for model data file according
        to top-level data model

        """

        self._fill_user_meta()
        self._fill_product_meta()
        self._fill_activity_meta()

    def _fill_user_meta(self):
        """
        Fill user-provided meta data

        """

        try:
            self.toplevel_meta['CTA']['CONTACT'] = self._user_meta['CONTACT']
            self.toplevel_meta['CTA']['INSTRUMENT'] = self._user_meta['INSTRUMENT']
            self.toplevel_meta['CTA']['PRODUCT']['DESCRIPTION'] = \
                self._user_meta['PRODUCT']['DESCRIPTION']
            self.toplevel_meta['CTA']['PRODUCT']['CREATION_TIME'] = \
                self._user_meta['PRODUCT']['CREATION_TIME']
            if 'CONTEXT' in self._user_meta['PRODUCT']:
                self.toplevel_meta['CTA']['PRODUCT']['CONTEXT'] = \
                    self._user_meta['PRODUCT']['CONTEXT']
            if 'VALID' in self._user_meta['PRODUCT']:
                if 'START' in self._user_meta['PRODUCT']['VALID']:
                    self.toplevel_meta['CTA']['PRODUCT']['VALID']['START'] = \
                        self._user_meta['PRODUCT']['VALID']['START']
                if 'END' in self._user_meta['PRODUCT']['VALID']:
                    self.toplevel_meta['CTA']['PRODUCT']['VALID']['END'] = \
                        self._user_meta['PRODUCT']['VALID']['END']
            self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION'] = \
                self._user_meta['PRODUCT']['ASSOCIATION']
            self.toplevel_meta['CTA']['PROCESS'] = self._user_meta['PROCESS']
        except KeyError:
            self._logger.debug("Error reading user input meta data")
            print(self._user_meta)
            print(self.toplevel_meta)
            raise

    def _fill_product_meta(self):
        """
        Fill product related meta data

        """

        self.toplevel_meta['CTA']['PRODUCT']['ID'] = str(uuid.uuid4())

        try:
            self.toplevel_meta['CTA']['PRODUCT']['FORMAT'] = \
                self._read_data_file_format()
        except KeyError:
            self._logger.debug("Error PRODUCT meta from user input meta data")
            raise

        self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION']['ID'] = \
            self._read_instrument_name()

    def _fill_activity_meta(self):
        """
        Fill activity (software) related meta data

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
            self._logger.debug("Error ACTIVITY meta from user input meta data")
            raise

    def _read_toplevel_metadata_file(self):
        """
        Return full path and name of top level data file

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

    def _read_data_file_format(self):
        """
        Return file format for data file

        """
        try:
            return self.workflow_config['CTASIMPIPE']['PRODUCT']['FORMAT']
        except KeyError:
            self._logger.info(
                "Using default file format for model file: ascii.ecsv")

        return "ascii.ecsv"

    def _read_data_file_name(self, suffix=None):
        """
        Return full path and name of data file

        file name is determined by:
        a. workflow_config['CTASIMPIPE']['PRODUCT']['NAME'] (preferred)
        b. _user_meta['PRODUCT']['NAME']

        """

        _directory = self._read_data_directory()

        # Filename
        try:
            _filename = str(
                self.workflow_config['CTASIMPIPE']['PRODUCT']['NAME'])
        except KeyError:
            _filename = None

        if not _filename:
            try:
                _filename = os.path.splitext(
                    self._user_meta['PRODUCT']['DATA'])[0]
            except KeyError:
                self._logger.error(
                    "Missing description in user meta of PRODUCT:NAME")
                raise
        # Suffix
        if not suffix:
            suffix = '.' + self._read_data_file_format()

        return _directory+'/'+_filename+suffix

    def _read_data_directory(self):
        """
        Return output directory for data products.
        Create directory if necessary.

        """

        try:
            _output_location = self.workflow_config['CTASIMPIPE']['PRODUCT']['DIRECTORY']
            _output_label = self.workflow_config["CTASIMPIPE"]["ACTIVITY"]["NAME"]
        except KeyError:
            pass

        if not _output_location:
            _output_location = cfg.get("outputLocation")

        if not _output_label:
            _output_label = ''

        _output_dir = io.getApplicationOutputDirectory(
            _output_location,
            _output_label)

        self._logger.info("Outputdirectory {}".format(_output_dir))

        return str(_output_dir)

    def _write_metadata(self):
        """
        Write model metadata file

        """

        if self.toplevel_meta:
            ymlfile = self._read_data_file_name('.yml')
            self._logger.debug(
                "Writing metadata to {}".format(ymlfile))
            with open(ymlfile, 'w', encoding='UTF-8') as file:
                yaml.dump(
                    self.toplevel_meta,
                    file,
                    sort_keys=False)

    def _read_instrument_name(self):
        """
        Returns a string defining the instrument following
        the gammasim-tools naming convention

        """

        # FIXME
        # - works only for camera names (requires adding equivalent of
        #   allCameraNames to names.py
        # - no validation / generation of SUBTYPE
        try:
            _instrument = \
                names.validateSiteName(
                    self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION']['SITE']) \
                + "-" + \
                names.validateName(
                    self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION']['CLASS'],
                    names.allTelescopeClassNames) \
                + "-" + \
                names.validateName(
                    self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION']['TYPE'],
                    names.allCameraNames) \
                + "-" + \
                self.toplevel_meta['CTA']['PRODUCT']['ASSOCIATION']['SUBTYPE']
        except KeyError:
            self._logger.error('Error reading PRODUCT:ASSOCIATION')
            raise

        return _instrument

    def _write_data(self):
        """
        Write model data file

        """

        if self._user_data:
            ecsvfile = self._read_data_file_name()
            self._logger.debug(
                "Writing data to {}".format(ecsvfile))
            self._user_data.write(
                ecsvfile,
                format=self._read_data_file_format(),
                overwrite=True)
