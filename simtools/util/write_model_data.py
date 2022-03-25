import datetime
import logging
import uuid
import yaml

import simtools.util.general as gen
import simtools.version


class ModelData:
    """
    Simulation model data class.

    Includes metadata enrichment and model data writing.

    Limitations:
    - allows only for writing of ascii.ecsv format

    Attributes:
    -----------
    workflow_config: dict
        workflow configuration
    toplevel_meta: dict
        top-level meta data definition

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
        Write a model data file including a complete
        set of metadata

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

    def _write_metadata(self):
        """
        Write model metadata file

        """

        if self.toplevel_meta:
            ymlfile = self._read_data_file_name('.yml')
            self._logger.debug(
                "Writing metadata to {}".format(ymlfile))
            with open(ymlfile, 'w') as file:
                yaml.dump(
                    self.toplevel_meta,
                    file,
                    sort_keys=False)

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
        Fill all user-related meta data

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
            self.toplevel_meta['CTA']['PROCESS'] = self._user_meta['PROCESS']
        except KeyError:
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
            raise

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
            raise

    def _get_toplevel_template(self):
        """
        Read and return toplevel data model template

        """

        if self._read_toplevel_metadata_file():
            return gen.collectDataFromYamlOrDict(
                self._read_toplevel_metadata_file(),
                None)

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
                "using default file format for model file: ascii.ecsv")

        return "ascii.ecsv"

    def _read_data_file_name(self, suffix=None):
        """
        Return full path and name of data file

        """

        # FIXME
        try:
            _model_file = str(
                str(self.workflow_config["CTASIMPIPE"]['PRODUCT']['DIRECTORY'])
                + '/' + 'tt')
        except KeyError:
            self._logger.error(
                "Missing description of PRODUCT:DIRECTORY")
            raise

        if suffix:
            return _model_file + suffix

        return str(_model_file + '.' + self._read_data_file_format())
