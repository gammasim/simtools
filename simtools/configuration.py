import logging
import os
import sys

import yaml

import simtools.util.commandline_parser as argparser


class Configurator:
    """
    Configuration handling application configuration.

    Allow to set configuration parameters by
    - command line arguments
    - configuration file (yml file)
    - environmental variables
    - configuration dict when calling the class

    """

    def __init__(self, config=None, label=None):

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Configuration")

        self.config = config
        self.parser = argparser.CommandLineParser(label)

    def initialize(self, add_workflow_config=False):
        """
        Parse args, check environmental variables, and return configuration.

        Returns
        -------
        dict
            Configuration arguments

        """

        self.parser.initialize_default_arguments(add_workflow_config=add_workflow_config)

        _list_args = sys.argv[1:]
        _list_args += self._arglistFromConfig(self.config)

        self.config = vars(self.parser.parse_args(_list_args))

        self._fillFromEnvironmentalVariables()
        self._fillFromConfigFile()

        return self.config

    def _fillFromConfigFile(self):
        """
        Read and fill configuration parameters from yaml file.
        Use argparse config parameter checker (parse all config parameters again)

        """

        if "config_file" not in self.config:
            return

        with open(self.config["config_file"], "r") as stream:
            _config = yaml.safe_load(stream)

        _list_args = self._arglistFromConfig(self.config) + self._arglistFromConfig(_config)

        self.config = vars(self.parser.parse_args(_list_args))

    @staticmethod
    def _arglistFromConfig(input_dict):
        """
        Convert input dict to list of strings; add argument double dashes; \
        handle boolean parameters.

        Parameters
        ----------
        input_dict: dict
           Dictionary of commands to convert to list.

        Returns
        -------
        list
            Dict keys and values as dict.

        """

        _list_args = []

        if input_dict is None:
            return _list_args

        for key, value in input_dict.items():
            if not isinstance(value, bool):
                _list_args.append("--" + key)
                _list_args.append(str(value))
            elif value:
                _list_args.append("--" + key)

        return _list_args

    def _fillFromEnvironmentalVariables(self):
        """
        Fill any unconfigured configuration parameters (parameter is None) \
        from environmental variables.

        """

        for key, value in self.config.items():
            if value is None:
                self.config[key] = os.environ.get(key.upper())
