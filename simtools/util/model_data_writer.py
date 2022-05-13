import logging
import yaml


class ModelDataWriter:
    """
    Simulation model data writer

    Includes writing of metadata and model data.

    Attributes:
    -----------
    workflow_config: WorkflowDescription
        workflow configuration

    Methods:
    --------
    write_data()
        Write model data to file.
    write_metadata()
        Write metadata to file.

    """

    def __init__(self, workflow_config=None):
        """
        Initialize model data

        Parameters
        ----------
        workflow_config: WorkflowDescription
            workflow configuration

        """

        self._logger = logging.getLogger(__name__)
        self._product_data_filename = None
        self.workflow_config = workflow_config

    def write_data(self,
                   product_data):
        """
        Write model data consisting of metadata and data files

        Parameters
        ----------
        product_data: astropy Table
            Model data.

        """

        _file = self.workflow_config.product_data_file_name()
        try:
            self._logger.debug("Writing data to {}".format(_file))
            product_data.write(
                _file,
                format=self.workflow_config.product_data_file_format(),
                overwrite=True)
        except AttributeError:
            self._logger.error("Error writing model data to {}".format(
                _file))
            raise

    def write_metadata(self):
        """
        Write model metadata file
        (yaml file format)

        """

        if self.workflow_config.toplevel_meta:
            ymlfile = self.workflow_config.product_data_file_name('.yml')
            self._logger.debug(
                "Writing metadata to {}".format(ymlfile))
            with open(ymlfile, 'w', encoding='UTF-8') as file:
                yaml.dump(
                    self.workflow_config.toplevel_meta,
                    file,
                    sort_keys=False)
        else:
            self._logger.debug("No metadata defined for write")
