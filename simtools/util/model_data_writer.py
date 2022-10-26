import logging

import yaml

from simtools.util import workflow_description


class ModelDataWriter:
    """
    Writer for simulation model data and metadata.

    Attributes
    ----------
    workflow_config: WorkflowDescription
        workflow configuration

    Methods
    -------
    write_data()
        Write model data to file.
    write_metadata()
        Write metadata to file.

    """

    def __init__(self, workflow_config=None, args_dict=None):
        """
        Initialize model data

        Parameters
        ----------
        workflow_config: WorkflowDescription
            Workflow configuration
        args_dict: Dictionary
            Dictionary with configuration parameters.

        """

        self._logger = logging.getLogger(__name__)
        self._product_data_filename = None
        self.workflow_config = self._get_workflow_config(workflow_config, args_dict)

    def write_data(self, product_data):
        """
        Write model data consisting of metadata and data files

        Parameters
        ----------
        product_data: astropy Table
            Model data.

        """

        _file = self.workflow_config.product_data_file_name()
        try:
            self._logger.info("Writing data to {}".format(_file))
            product_data.write(
                _file, format=self.workflow_config.product_data_file_format(), overwrite=True
            )
        except FileNotFoundError:
            self._logger.error("Error writing model data to {}".format(_file))
            raise

    def write_metadata(self, ymlfile=None):
        """
        Write model metadata file
        (yaml file format)

        Attributes
        ----------
        ymlfile str
            name of output file (default=None)

        """

        try:
            if not ymlfile:
                ymlfile = self.workflow_config.product_data_file_name(".yml")
            self._logger.info("Writing metadata to {}".format(ymlfile))
            with open(ymlfile, "w", encoding="UTF-8") as file:
                yaml.safe_dump(self.workflow_config.top_level_meta, file, sort_keys=False)
        except FileNotFoundError:
            self._logger.error("Error writing model data to {}".format(ymlfile))
            raise
        except AttributeError:
            self._logger.error("No metadata defined for writing")
            raise

    @staticmethod
    def _get_workflow_config(workflow_config=None, args_dict=None):
        """
        Return workflow config, if needed from command line parameter dictionary.

        Parameters
        ----------
        workflow_config: WorkflowDescription
            Workflow configuration
        args_dict: Dictionary
            Dictionary with configuration parameters.

        Returns
        -------
        WorkflowDescription
            Workflow configuration

        """
        if workflow_config:
            return workflow_config

        return workflow_description.WorkflowDescription(args_dict=args_dict)
