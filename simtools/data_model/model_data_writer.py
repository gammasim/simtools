import logging

import yaml

import simtools.util.general as gen
from simtools.data_model import workflow_description

__all__ = ["ModelDataWriter"]


class ModelDataWriter:
    """
    Writer for simulation model data and metadata.

    Parameters
    ----------
    workflow_config: WorkflowDescription
        Workflow configuration.
    args_dict: Dictionary
        Dictionary with configuration parameters.
    """

    def __init__(self, workflow_config=None, args_dict=None):
        """
        Initialize model data
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

        Raises
        ------
        FileNotFoundError
            if Workflow configuration file not found.
        """

        _file = self.workflow_config.product_data_file_name()
        try:
            self._logger.info(f"Writing data to {_file}")
            product_data.write(
                _file, format=self.workflow_config.product_data_file_format(), overwrite=True
            )
        except FileNotFoundError:
            self._logger.error(f"Error writing model data to {_file}")
            raise

    def write_metadata(self, ymlfile=None, keys_lower_case=False):
        """
        Write model metadata file (yaml file format).

        Parameters
        ----------
        ymlfile: str
            Name of output file.
        keys_lower_case: bool
            Write yaml key in lower case.

        Returns
        -------
        str
            Name of output file

        Raises
        ------
        FileNotFoundError
            If ymlfile not found.
        AttributeError
            If no metadata defined for writing.
        """

        try:
            if not ymlfile:
                ymlfile = self.workflow_config.product_data_file_name(".yml")
            self._logger.info(f"Writing metadata to {ymlfile}")
            with open(ymlfile, "w", encoding="UTF-8") as file:
                yaml.safe_dump(
                    gen.change_dict_keys_case(self.workflow_config.top_level_meta, keys_lower_case),
                    file,
                    sort_keys=False,
                )
            return ymlfile
        except FileNotFoundError:
            self._logger.error(f"Error writing model data to {ymlfile}")
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
