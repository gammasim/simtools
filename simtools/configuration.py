import logging
import os
import sys

import yaml

import simtools.io_handler as io_handler
import simtools.util.commandline_parser as argparser


class InvalidConfigurationParameter(Exception):
    pass


class Configurator:
    """
    Configuration handling application configuration.

    Allow to set configuration parameters by
    - command line arguments
    - configuration file (yml file)
    - environmental variables
    - configuration dict when calling the class

    Methods
    -------
    initialize()
       Initialize configuration from command line, configuration file, class config, or env.

    """

    def __init__(self, config=None, label=None, description=None):
        """
        Configurator init.

        Parameters
        ----------
        config: dict
           Configuration parameters as dict.
        label: str
           Class label.

        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Configuration")

        self.configClassInit = config
        self.label = label
        self.config = {}
        self.parser = argparser.CommandLineParser(self.label, description)

    def default_config(self, arg_list=None, add_db_config=False):
        """
        Returns dictionary of default configuration

        """
        self.parser.initialize_default_arguments()
        if arg_list and "--site" in arg_list:
            self.parser.initialize_telescope_model_arguments(True, "--telescope" in arg_list)
        if add_db_config:
            self.parser.initialize_db_config_arguments()

        self._fillConfig(arg_list)
        return self.config

    def initialize(
        self, telescope_model=False, workflow_config=False, db_config=False, job_submission=False
    ):
        """
        Initialize configuration from command line, configuration file, class config, \
        or environmental variable.

        Priorities in parameter settings.
        1. command line; 2. yaml file; 3. class init; 4. env variables.

        Conflicting configuration settings raise an Exception, with the exception of settings \
        from environmental variables, which are only done when the configuration parameter \
        is None.

        Parameters
        ----------
        telescope_model: bool
            Add telescope model configuration to list of args.
        workflow_config: bool
            Add workflow configuration to list of args.
        db_config: bool
            Add database configuration parameters to list of args.
        job_submission: bool
            Add job submission configuration to list of args.

        Returns
        -------
        dict
            Configuration parameters as dict.

        Raises
        ------
        InvalidConfigurationParameter
           if parameter has already been defined with a different value.

        """

        self.parser.initialize_default_arguments(
            telescope_model=telescope_model,
            workflow_config=workflow_config,
            db_config=db_config,
            job_submission=job_submission,
        )

        self._fillFromCommandLine()
        self._fillFromConfigFile()
        self._fillFromConfigDict(self.configClassInit)
        self._fillFromEnvironmentalVariables()
        self._initializeIOHandler()
        _db_dict = self._getDBParameters()

        if self.config["label"] is None:
            self.config["label"] = self.label

        return self.config, _db_dict

    def _fillFromCommandLine(self, arg_list=None):
        """
        Fill configuration parameters from command line arguments.

        """

        if arg_list is None:
            arg_list = sys.argv[1:]

        self._fillConfig(arg_list)

    def _fillFromConfigDict(self, _input_dict):
        """
        Fill configuration parameters from dictionary.

        Parameters
        ----------
        _input_dict: dict
           dictionary with configuration parameters

        """
        _tmp_config = {}
        try:
            for key, value in _input_dict.items():
                self._check_parameter_configuration_status(key, value)
                _tmp_config[key] = value
        except AttributeError:
            pass

        self._fillConfig(_tmp_config)

    def _check_parameter_configuration_status(self, key, value):
        """
        Check if a parameter is already configured and not still set to the default value.
        Allow configuration of None values.

        Parameters
        ----------
        key, value
           parameter key, value to be checked


        Raises
        ------
        InvalidConfigurationParameter
           if parameter has already been defined with a different value.


        """

        # parameter not changed or None
        if self.parser.get_default(key) == self.config[key] or self.config[key] is None:
            return

        # parameter already set
        if key in self.config and self.config[key] != value:
            self._logger.error(
                "Inconsistent configuration parameter ({}) definition ({} vs {})".format(
                    key, self.config[key], value
                )
            )
            raise InvalidConfigurationParameter

    def _fillFromConfigFile(self):
        """
        Read and fill configuration parameters from yaml file.

        Raises
        ------
        FileNotFoundError
           if configuration file has not been found.

        """

        try:
            self._logger.debug("Reading configuration from {}".format(self.config["config_file"]))
            with open(self.config["config_file"], "r") as stream:
                _config_dict = yaml.safe_load(stream)
            self._fillFromConfigDict(_config_dict)
        except TypeError:
            pass
        except FileNotFoundError:
            self._logger.error(
                "Configuration file not found: {}".format(self.config["config_file"])
            )
            raise

    def _fillFromEnvironmentalVariables(self):
        """
        Fill any unconfigured configuration parameters (parameter is None) \
        from environmental variables.

        """

        _env_dict = {}
        try:
            for key, value in self.config.items():
                if value is None:
                    _env_dict[key] = os.environ.get(key.upper())
        except AttributeError:
            pass

        self._fillFromConfigDict(_env_dict)

    def _initializeIOHandler(self):
        """
        Initialize IOHandler with input and output paths.

        """
        _io_handler = io_handler.IOHandler()
        _io_handler.setPaths(
            output_path=self.config["output_path"],
            data_path=self.config["data_path"],
            model_path=self.config["model_path"],
        )

    @staticmethod
    def _arglistFromConfig(input_var):
        """
        Convert input list of strings as needed by argparse.

        Special cases:
        - boolean are expected to be handled as action="store_true" or "store_false"
        - None values or zero length values are ignored (this means setting a parameter \
            to none or "" is not allowed.


        Ignore values which are None or of zero length.

        Parameters
        ----------
        input_var: dict, list, None
           Dictionary/list of commands to convert to list.

        Returns
        -------
        list
            Dict keys and values as dict.

        """

        if isinstance(input_var, dict):
            _list_args = []
            for key, value in input_var.items():
                if not isinstance(value, bool) and value is not None and len(str(value)) > 0:
                    _list_args.append("--" + key)
                    _list_args.append(str(value))
                elif value:
                    _list_args.append("--" + key)
            return _list_args

        try:
            return [str(value) for value in list(input_var) if value != "None"]
        except TypeError:
            return []

    @staticmethod
    def _convert_stringnone_to_none(input_dict):
        """
        Convert string type 'None' to type None (argparse returns None as str).

        Parameters
        ----------
        input_dict
            Dictionary with values to be converted.

        """

        for key, value in input_dict.items():
            input_dict[key] = None if value == "None" else value

        return input_dict

    def _fillConfig(self, input_container):
        """
        Fill configuration dictionary.

        Parameters
        ----------
        input_container
            List or dictionary with configuration updates.

        """

        self.config = self._convert_stringnone_to_none(
            vars(
                self.parser.parse_args(
                    self._arglistFromConfig(self.config) + self._arglistFromConfig(input_container)
                )
            )
        )

    def _getDBParameters(self):
        """
        Return parameters for DB configuration

        Parameters
        ----------
        dict
            Dictionary with DB parameters


        """

        _db_dict = {}
        _db_para = ("db_api_user", "db_api_pw", "db_api_port", "db_api_name")
        try:
            for _para in _db_para:
                _db_dict[_para] = self.config[_para]
        except KeyError:
            pass

        return _db_dict
