import logging

import simtools.util.general as gen

class WorkflowConfiguration:
    """
    Workflow configuration class

    """

    def __init__(self):
        """
        Initialize workflow configuration class

        """

        self._logger = logging.getLogger(__name__)
        self.configuration = {}

    def collect_configuration(self,
                              workflow_config_file,
                              reference_schema_directory=None):
        """
        Collect configuration parameter into a single dict
        (simplifies processing)

        Parameters:
        -----------
        workflow_config_file
            configuration file describing this workflow
        reference_schema_directory
            directory to reference schema

        Return:
        -------
        workflow_config: dict
            workflow configuration

        """

        _workflow_config = gen.collectDataFromYamlOrDict(
            workflow_config_file, None)
        self._logger.debug("Reading workflow configuration from {}".format(
            workflow_config_file))

        if reference_schema_directory:
            try:
                _workflow_config['CTASIMPIPE']['DATAMODEL']['SCHEMADIRECTORY'] = \
                    reference_schema_directory
            except KeyError as error:
                self._logger.error("Workflow configuration incomplete")
                raise KeyError from error

        self.configuration = _workflow_config
        return self.configuration
