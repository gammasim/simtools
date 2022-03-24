import datetime
import logging
import uuid
import yaml


class ModelData:
    """
    Simulation model data class.

    Includes metadata enrichment and model data writing.

    Limitations:
    - allows only for writing of ascii.ecsv format

    Attributes:
    -----------

    Methods:
    --------
    write_model_file()
        Write model and model metadata to file.

    """

    def __init__(self):
        """
        Initialize model data

        """

        self._logger = logging.getLogger(__name__)

        self.toplevel_meta = self._get_toplevel_template()

        self._modelfile = None
        self._modelfile_data_format = 'ascii.ecsv'
        self._workflow_config = None
        self._user_meta = None

    def write_model_file(self,
                         workflow_config,
                         user_meta,
                         user_data,
                         output_dir):
        """
        Write a model data file including a complete
        set of metadata

        Parameters
        ----------
        workflow_config: dict
            Workflow configuration.
        user_meta: dict
            User given meta data.
        user_data: astropy Table
            Model data.
        output_dir: str
            Ouput directory for model and meta data writing.


        """
        # FIXME where do we state the name of the output file?
        self._modelfile = str(output_dir) + '/tt'

        self._workflow_config = workflow_config
        self._user_meta = user_meta

        self._prepare_metadata()

        self._write(user_data)

    def _write(self, data=None):
        """
        Write model metadata and data files

        """

        if self.toplevel_meta:
            ymlfile = str(self._modelfile+'.yml')
            self._logger.debug(
                "Writing metadata to %s", ymlfile)
            with open(ymlfile, 'w') as file:
                yaml.dump(
                    self.toplevel_meta,
                    file,
                    sort_keys=False)

        if data:
            ecsvfile = self._modelfile + '.' + self._modelfile_data_format
            self._logger.debug(
                "Writing data to %s", ecsvfile)
            data.write(
                ecsvfile,
                format=self._modelfile_data_format,
                overwrite=True)

    def _prepare_metadata(self):
        """
        Prepare metadata for model data file according
        to top-level data model

        """

        self._fill_user_meta()

        self.toplevel_meta['CTA']['PRODUCT']['ID'] = str(uuid.uuid4())

        try:
            self.toplevel_meta['CTA']['PRODUCT']['FORMAT'] = self._modelfile_data_format
        except KeyError:
            raise

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

    def _fill_activity_meta(self):
        """
        Fill activity (software) related meta data

        """
        try:
            self.toplevel_meta['CTA']['ACTIVITY']['NAME'] = \
                self._workflow_config['CTASIMPIPE']['ACTIVITY']['NAME']
            self.toplevel_meta['CTA']['ACTIVITY']['START'] = \
                datetime.datetime.now().isoformat()
            self.toplevel_meta['CTA']['ACTIVITY']['END'] = \
                self.toplevel_meta['CTA']['ACTIVITY']['START']
        except KeyError:
            raise

    def _get_toplevel_template(self):
        """
        Return toplevel data model template

        """

        return {
            'CTA': {
                'REFERENCE': {
                    'VERSION': '1.0.0'},
                'PRODUCT': {
                    'DESCRIPTION': None,
                    'CONTEXT': None,
                    'CREATION_TIME': None,
                    'ID': None,
                    'DATA': {
                        'CATEGORY': 'SIM',
                        'LEVEL': 'R0',
                        'ASSOCIATION': None,
                        'TYPE': 'service',
                        'MODEL': {
                            'NAME': 'simpipe-table',
                            'VERSION': '0.1.0',
                            'URL': None},
                    },
                    'FORMAT': None
                },
                'INSTRUMENT': {
                    'SITE': None,
                    'CLASS': None,
                    'TYPE': None,
                    'SUBTYPE': None,
                    'ID': None
                },
                'PROCESS': {
                    'TYPE': None,
                    'SUBTYPE': None,
                    'ID': None
                },
                'CONTACT': {
                    'ORGANIZATION': None,
                    'NAME': None,
                    'EMAIL': None
                },
                'ACTIVITY': {
                    'NAME': None,
                    'TYPE': 'software',
                    'ID': None,
                    'START': None,
                    'END': None,
                    'SOFTWARE': {
                        'NAME': 'gammasim-tools',
                        'VERSION': None}
                }
            }
        }
