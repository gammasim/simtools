"""Application configuration."""

import argparse
import logging
import sys

import astropy.units as u

import simtools.configuration.commandline_parser as argparser
from simtools.db.mongo_db import jsonschema_db_dict
from simtools.io import ascii_handler, io_handler
from simtools.utils import general as gen


class InvalidConfigurationParameterError(Exception):
    """Exception for Invalid configuration parameter."""


class Configurator:
    """
    Application configuration.

    Allow to set configuration parameters by

    - command line arguments
    - configuration file (yml file)
    - configuration dict when calling the class
    - environmental variables

    Assigns unique ACTIVITY_ID to this configuration (uuid).

    Configuration parameter names are converted always to lower case.

    Parameters
    ----------
    config: dict
        Configuration parameters as dict.
    label: str
        Class label.
    usage: str
        Application usage description.
    description: str
        Text displayed as description.
    epilog: str
        Text display after all arguments.
    """

    def __init__(self, config=None, label=None, usage=None, description=None, epilog=None):
        """Initialize Configurator."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Configuration")

        self.config_class_init = config
        self.label = label
        self.config = {}
        self.parser = argparser.CommandLineParser(
            prog=self.label,
            usage=usage,
            description=description,
            epilog=epilog,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def default_config(self, arg_list=None, add_db_config=False):
        """
        Return dictionary of default configuration.

        Parameters
        ----------
        arg_list: list
            List of arguments.
        add_db_config: bool
            Add DB configuration file.

        Returns
        -------
        dict
            Configuration parameters as dict.
        """
        self.parser.initialize_default_arguments()
        simulation_model = None
        if arg_list and "--site" in arg_list:
            simulation_model = ["site"]
        if arg_list and "--telescope" in arg_list:
            simulation_model = ["site", "telescope"]
        self.parser.initialize_simulation_model_arguments(simulation_model)
        if add_db_config:
            self.parser.initialize_db_config_arguments()

        self._fill_config(arg_list)
        return self.config

    def initialize(
        self,
        require_command_line=True,
        paths=True,
        output=False,
        simulation_model=None,
        simulation_configuration=None,
        db_config=False,
    ):
        """
        Initialize application configuration.

        Configure from command line, configuration file, class config, or environmental variable.

        Configuration sources applied in increasing priority order:
        env variables < class init < yaml file < command line.

        Parameters
        ----------
        require_command_line: bool
            Require at least one command line argument.
        paths: bool
            Add path configuration to list of args.
        output: bool
            Add output file configuration to list of args.
        simulation_model: list
            List of simulation model configuration parameters to add to list of args
        simulation_configuration: dict
            Dict of simulation software configuration parameters to add to list of args.
        db_config: bool
            Add database configuration parameters to list of args.

        Returns
        -------
        dict
            Configuration parameters as dict.
        dict
            Dictionary with DB parameters

        """
        self.parser.initialize_default_arguments(
            paths=paths,
            output=output,
            simulation_model=simulation_model,
            simulation_configuration=simulation_configuration,
            db_config=db_config,
        )

        _cli_arglist = self._get_cli_arglist(require_command_line=require_command_line)
        _cli_config = vars(self.parser.parse_args(_cli_arglist))
        _config_file = _cli_config.get("config") or (self.config_class_init or {}).get("config")
        _env_file = _cli_config.get("env_file") or (self.config_class_init or {}).get("env_file")

        self._reset_required_arguments()
        self.config = vars(self.parser.parse_args([]))
        self.config.update(self._config_from_env(_env_file))
        self.config.update(gen.change_dict_keys_case(self.config_class_init or {}))
        self.config.update(self._config_from_file(_config_file))
        self._fill_config(_cli_arglist)

        if self.config.get("activity_id", None) is None:
            self.config["activity_id"] = gen.get_uuid()
        if self.config["label"] is None:
            self.config["label"] = self.label
        self._initialize_model_versions()

        self._initialize_io_handler()
        if output:
            self._initialize_output()
        _db_dict = self._get_db_parameters()

        self.config["application_label"] = self.config.get("application_label") or self.label
        return self.config, _db_dict

    def _get_cli_arglist(self, arg_list=None, require_command_line=True):
        """
        Return CLI arguments as a list without modifying the configuration.

        Parameters
        ----------
        arg_list: list
            List of arguments.
        require_command_line: bool
            Require at least one command line argument.

        Returns
        -------
        list
            Command-line arguments.

        """
        if arg_list is None:
            arg_list = sys.argv[1:]

        if require_command_line and len(arg_list) == 0:
            self._logger.debug("No command line arguments given, printing help.")
            arg_list = ["--help"]

        if "--config" in arg_list:
            self._reset_required_arguments()

        return arg_list

    def _reset_required_arguments(self):
        """
        Reset required parser arguments (i.e., arguments added with "required=True").

        Includes also mutually exclusive groups.
        Access protected attributes of parser (no public method available).

        """
        for group in self.parser._mutually_exclusive_groups:  # pylint: disable=protected-access
            group.required = False
        for action in self.parser._actions:  # pylint: disable=protected-access
            action.required = False

    def _config_from_file(self, config_file):
        """
        Read configuration from yaml file and return as dictionary.

        Parameters
        ----------
        config_file: str
            Name of configuration file.

        Returns
        -------
        dict
            Configuration parameters.

        Raises
        ------
        FileNotFoundError
            If configuration file has not been found.
        """
        try:
            self._logger.debug(f"Reading configuration from {config_file}")
            _config_dict = (
                ascii_handler.collect_data_from_file(file_name=config_file) if config_file else None
            )
            _config_dict = gen.remove_substring_recursively_from_dict(_config_dict, substring="\n")
            if "configuration" in _config_dict.get("applications", [{}])[0]:
                return gen.change_dict_keys_case(_config_dict["applications"][0]["configuration"])
            return gen.change_dict_keys_case(_config_dict)
        except (TypeError, AttributeError):
            return {}
        except FileNotFoundError:
            self._logger.error(f"Configuration file not found: {config_file}")
            raise

    def _config_from_env(self, env_file=None):
        """
        Return configuration parameters from environment variables.

        Parameters
        ----------
        env_file: str
            Path to the .env file.

        Returns
        -------
        dict
            Configuration parameters from environment variables.
        """
        _env_list = [action.dest for action in self.parser._actions]  # pylint: disable=protected-access
        return gen.load_environment_variables(env_file=env_file, env_list=_env_list)

    def _fill_from_command_line(self, arg_list=None, require_command_line=True):
        """
        Fill configuration parameters from command line arguments.

        Triggers a print of the help if no command line arguments are given and
        require_command_line is set.

        Parameters
        ----------
        arg_list: list
            List of arguments.
        require_command_line: bool
            Require at least one command line argument.

        """
        self._fill_config(self._get_cli_arglist(arg_list, require_command_line))

    def _fill_from_config_dict(self, input_dict, overwrite=False):
        """
        Fill configuration parameters from dictionary.

        Enforce that configuration parameter names are lower case.

        Parameters
        ----------
        input_dict: dict
            dictionary with configuration parameters.
        overwrite: bool
            overwrite existing configuration parameters.

        """
        _tmp_config = {}
        try:
            for key, value in input_dict.items():
                if not overwrite:
                    self._check_parameter_configuration_status(key, value)
                _tmp_config[key.lower()] = value
        except AttributeError:
            pass

        self._fill_config(_tmp_config)

    def _check_parameter_configuration_status(self, key, value):
        """
        Check if a parameter is already configured and not still set to the default value.

        Allow configuration with None values.

        Parameters
        ----------
        key, value
           parameter key, value to be checked


        Raises
        ------
        InvalidConfigurationParameterError
           if parameter has already been defined with a different value.
        """
        # parameter not changed or None
        if self.parser.get_default(key) == self.config[key] or self.config[key] is None:
            return

        # parameter already set
        if key in self.config and self.config[key] != value:
            self._logger.error(
                f"Inconsistent configuration parameter ({key}) definition "
                f"({self.config[key]} vs {value})"
            )
            raise InvalidConfigurationParameterError

    def _fill_from_config_file(self, config_file):
        """
        Fill configuration parameters from yaml file.

        Parameters
        ----------
        config_file: str
            Name of configuration file.

        Raises
        ------
        FileNotFoundError
            If configuration file has not been found.
        """
        self._fill_from_config_dict(self._config_from_file(config_file), overwrite=True)

    def _fill_from_environmental_variables(self):
        """
        Fill any configuration parameters from environmental variables or from file (e.g., ".env").

        Only parameters which are not already configured are changed (i.e., parameter is None).

        """
        _all_env_dict = gen.load_environment_variables(
            env_file=self.config.get("env_file", None), env_list=self.config.keys()
        )

        _env_dict = {}
        for key, value in self.config.items():
            if value is None:
                env_value = _all_env_dict.get(key)
                if env_value is not None:
                    _env_dict[key] = env_value

        self._fill_from_config_dict(_env_dict)

    def _initialize_model_versions(self):
        """Initialize model versions."""
        if (
            self.config.get("model_version", None)
            and isinstance(self.config["model_version"], list)
            and len(self.config["model_version"]) == 1
        ):
            self.config["model_version"] = self.config["model_version"][0]

    def _initialize_io_handler(self):
        """Initialize IOHandler with input and output paths."""
        _io_handler = io_handler.IOHandler()
        _io_handler.set_paths(
            output_path=self.config.get("output_path", None),
            model_path=self.config.get("model_path", None),
        )

    def _initialize_output(self):
        """Initialize default output file names (in case output_file is not configured)."""
        if self.config.get("output_file", None) is None:
            self.config["output_file_from_default"] = True
            prefix = "TEST"
            label = extension = ""
            if not self.config.get("test", False):
                prefix = self.config["activity_id"]
                if self.config.get("label", "") and len(self.config.get("label", "")) > 0:
                    label = f"-{self.config['label']}"
            if len(self.config.get("output_file_format", "")) > 0:
                extension = f".{self.config['output_file_format']}"

            self.config["output_file"] = f"{prefix}{label}{extension}"

    @staticmethod
    def _arglist_from_config(input_var):
        """
        Convert input list of strings as needed by argparse.

        Special cases:
        - lists as arguments (using e.g., nargs="+") are expanded
        - boolean are expected to be handled as action="store_true" or "store_false"
        - None values or zero length values are ignored (this means setting a parameter
          to none or "" is not allowed).


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
                if isinstance(value, list):
                    _list_args.append("--" + key)
                    _list_args.extend(map(str, value))
                elif isinstance(value, u.Quantity) or (
                    not isinstance(value, bool) and value is not None and len(str(value)) > 0
                ):
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
    def _convert_string_none_to_none(input_dict):
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

    def _fill_config(self, input_container):
        """
        Fill configuration dictionary.

        Parameters
        ----------
        input_container
            List or dictionary with configuration updates.
        """
        self.config = self._convert_string_none_to_none(
            vars(
                self.parser.parse_args(
                    self._arglist_from_config(self.config)
                    + self._arglist_from_config(input_container)
                )
            )
        )

    def _get_db_parameters(self):
        """
        Return parameters for DB configuration.

        Returns
        -------
        dict
            Dictionary with DB parameters.
        """
        db_params = jsonschema_db_dict["properties"].keys()
        return {param: self.config.get(param) for param in db_params if param in self.config}
