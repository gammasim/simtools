import logging
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

        self._modelfile = None
        self._modelfile_data_format = 'ascii.ecsv'

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

        toplevel_meta = self._prepare_metadata(user_meta)

        self._write(toplevel_meta, user_data)

    def _write(self, data_meta=None, data=None):
        """
        Write model metadata and data files

        """

        if data_meta:
            ymlfile = str(self._modelfile+'.yml')
            self._logger.debug(
                "Writing metadata to %s", ymlfile)
            with open(ymlfile, 'w') as file:
                yaml.dump(
                    data_meta,
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

    def _prepare_metadata(self, user_meta):
        """
        Prepare metadata for model data file according
        to top-level data model

        """

        toplevel_meta = self._get_toplevel_template()
        try:
            toplevel_meta['CTA']['CONTACT'] = user_meta['CONTACT']
            toplevel_meta['CTA']['INSTRUMENT'] = user_meta['INSTRUMENT']

            toplevel_meta['CTA']['PRODUCT']['DATA']['FORMAT'] = self._modelfile_data_format

        except KeyError:
            raise

        return toplevel_meta

    def _get_toplevel_template(self):
        """
        Return toplevel data model template

        """

        return {
            'CTA': {
                'REFERENCE': {
                    'VERSION': '1.0.0'},
                'CONTACT': {
                    'ORGANIZATION': None,
                    'NAME': None,
                    'EMAIL': None
                },
                'INSTRUMENT': {
                    'SITE': None,
                    'CLASS': None,
                    'TYPE': None,
                    'SUBTYPE': None,
                    'ID': None
                },
                'PRODUCT': {
                    'DESCRIPTION': None,
                    'CREATION_TIME': None,
                    'ID': None,
                    'DATA': {
                        'CATEGORY': None,
                        'LEVEL': None,
                        'ASSOCIATION': None,
                        'TYPE': 'service',
                        'MODEL': {
                            'NAME': None,
                            'VERSION': '0.1.0',
                            'URL': None},
                        'FORMAT': None,
                        'PROCESS': {
                            'TYPE': None,
                            'SUBTYPE': None,
                            'ID': None},
                        'ACTIVITY': {
                            'NAME': None,
                            'TYPE': None,
                            'ID': None,
                            'START': None,
                            'END': None,
                            'SOFTWARE': {
                                'NAME': 'gammasim-tools',
                                'VERSION': None}
                        }
                    }
                }
            }
        }
