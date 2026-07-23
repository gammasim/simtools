"""Application configuration."""

import argparse
import logging
import shlex
import sys

import astropy.units as u

import simtools.configuration.commandline_parser as argparser
import simtools.version as simtools_version
from simtools.db.mongo_db import jsonschema_db_dict
from simtools.io import ascii_handler, io_handler
from simtools.utils import general as gen


class Configurator:
    """
    Application configuration.

    Allow to set configuration parameters by

    - command line arguments
    - configuration file
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
    description: str
        Text displayed as description.
    usage: str
        Application usage description.
    """

    def __init__(self, config=None, label=None, description=None, usage=None):
        """Initialize Configurator."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Configuration")

        self.config_class_init = config
        self.label = label
        self.config = {}
        self.config_sources = {
            "defaults": set(),
            "environment": set(),
            "constructor": set(),
            "yaml": set(),
            "cli": set(),
        }
        self.parser = argparser.CommandLineParser(
            prog=self.label,
            description=description,
            usage=usage,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def configure(
        self,
        initialize_output=False,
    ):
        """Initialize configuration from an already populated parser.

        Parameters
        ----------
        initialize_output : bool
            Generate a default output filename when none is configured.

        Returns
        -------
        tuple
            Application configuration and database configuration dictionaries.
        """
        cli_arglist = self._get_cli_arglist()
        config_file = self._option_value(cli_arglist, "--config") or (
            self.config_class_init or {}
        ).get("config")
        env_file = self._option_value(cli_arglist, "--env_file") or (
            self.config_class_init or {}
        ).get("env_file", ".env")

        env_config = self._config_from_env(env_file)
        file_config = self._config_from_file(config_file)
        constructor_config = gen.change_dict_keys_case(self.config_class_init or {})
        default_config = self._parser_defaults()
        cli_keys = self._explicit_cli_keys(cli_arglist)
        self.config_sources = {
            "defaults": set(default_config),
            "environment": set(env_config),
            "constructor": set(constructor_config),
            "yaml": set(file_config),
            "cli": cli_keys,
        }
        merged_config = {
            **default_config,
            **env_config,
            **constructor_config,
            **file_config,
        }
        merged_config = self._without_cli_overrides(merged_config, cli_keys)
        self.config = self._convert_string_none_to_none(
            vars(
                self.parser.parse_args(
                    self._arglist_from_config(merged_config, parser=self.parser) + cli_arglist
                )
            )
        )

        if self.config.get("activity_id") is None:
            self.config["activity_id"] = gen.get_uuid()
        if self.config["label"] is None:
            self.config["label"] = self.label
        self._initialize_model_versions()
        self._initialize_io_handler()
        if initialize_output:
            self._initialize_output()
        db_dict = self._get_db_parameters()
        self.config["application_label"] = self.config.get("application_label") or self.label
        return self.config, db_dict

    def _parser_defaults(self):
        """Return parser defaults as a configuration source mapping."""
        return {
            action.dest: action.default
            for action in self.parser._actions  # pylint: disable=protected-access
            if action.default is not argparse.SUPPRESS
        }

    @staticmethod
    def _option_value(arg_list, option_name):
        """Return the last explicit value for one CLI option without parsing other options."""
        value = None
        for index, argument in enumerate(arg_list):
            if argument == option_name and index + 1 < len(arg_list):
                value = arg_list[index + 1]
            elif argument.startswith(f"{option_name}="):
                value = argument.split("=", maxsplit=1)[1]
        return value

    def _get_cli_arglist(self):
        """Return CLI arguments, or help when the application was called without arguments."""
        arg_list = sys.argv[1:]
        if not arg_list:
            self._logger.debug("No command line arguments given, printing help.")
            arg_list = ["--help"]
        return arg_list

    def _without_cli_overrides(self, config, cli_keys):
        """Remove values superseded by CLI options, including exclusive alternatives."""
        superseded = set(cli_keys)
        for group in self.parser._mutually_exclusive_groups:  # pylint: disable=protected-access
            group_destinations = {
                action.dest
                for action in group._group_actions  # pylint: disable=protected-access
            }
            if group_destinations & cli_keys:
                superseded.update(group_destinations)
        return {key: value for key, value in config.items() if key not in superseded}

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
                _config_dict = _config_dict["applications"][0]["configuration"]

            if _config_dict:
                _config_dict = io_handler.resolve_test_resource_paths(_config_dict)

            _config_dict = simtools_version.resolve_by_version(
                _config_dict,
                _config_dict.get("model_version"),
                preserve_inconsistent_keys=self.parser.preserve_by_version,
            )
            return gen.change_dict_keys_case(_config_dict)
        except TypeError, AttributeError:
            self._logger.debug("No YAML configuration update applied to configuration dictionary.")
            return {}
        except FileNotFoundError:
            self._logger.error(f"Configuration file not found: {config_file}")
            raise

    def _explicit_cli_keys(self, arg_list):
        """Return parser destination names explicitly present in a CLI argument list."""
        explicit_keys = set()
        for arg in arg_list:
            option = arg.split("=", maxsplit=1)[0]
            action = self.parser._option_string_actions.get(option)  # pylint: disable=protected-access
            if action is not None:
                explicit_keys.add(action.dest)
        return explicit_keys

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
    def _arglist_from_config(input_var, parser=None):
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
        parser: argparse.ArgumentParser, optional
            Parser used to detect fixed-arity arguments (``nargs=<int>``). When provided,
            scalar string values for those arguments are split into multiple CLI tokens.

        Returns
        -------
        list
            Dict keys and values as dict.

        """
        if isinstance(input_var, dict):
            return Configurator._arglist_from_dict(input_var, parser=parser)

        try:
            return [str(value) for value in list(input_var) if value != "None"]
        except TypeError:
            return []

    @staticmethod
    def _arglist_from_dict(input_dict, parser=None):
        """Convert a configuration dictionary into CLI-style argument tokens."""
        list_args = []
        for key, value in input_dict.items():
            list_args.extend(Configurator._arg_tokens_for_item(key, value, parser=parser))
        return list_args

    @staticmethod
    def _arg_tokens_for_item(key, value, parser=None):
        """Return CLI argument tokens for a single configuration entry."""
        option = f"--{key}"
        action = (
            parser._option_string_actions.get(option)  # pylint: disable=protected-access
            if parser is not None
            else None
        )

        boolean_tokens = Configurator._boolean_arg_tokens(option, value, action)
        if boolean_tokens is not None:
            return boolean_tokens

        if isinstance(value, list):
            return [option, *map(str, value)]

        if Configurator._is_scalar_config_value(value):
            return [option, *Configurator._normalize_scalar_config_value(key, value, parser=parser)]

        if value:
            return [option]

        return []

    @staticmethod
    def _boolean_arg_tokens(option, value, action):
        """Return tokens for a boolean parser action, or None for non-boolean values."""
        if not isinstance(value, bool):
            return None
        if action is not None and action.__class__.__name__ == "_StoreFalseAction":
            return [option] if not value else []
        return [option] if value else []

    @staticmethod
    def _is_scalar_config_value(value):
        """Return True for scalar values that should produce an option and one or more tokens."""
        return isinstance(value, u.Quantity) or (
            not isinstance(value, bool) and value is not None and len(str(value)) > 0
        )

    @staticmethod
    def _normalize_scalar_config_value(key, value, parser=None):
        """Normalize a scalar config value to one or more argument tokens."""
        expected_nargs = Configurator._get_fixed_nargs(parser, key)
        if not (isinstance(value, str) and isinstance(expected_nargs, int) and expected_nargs > 1):
            return [str(value)]

        split_values = shlex.split(value)
        if len(split_values) != expected_nargs:
            raise ValueError(
                f"Configuration value for '{key}' must provide "
                f"{expected_nargs} entries, got {len(split_values)}: {value}"
            )
        return split_values

    @staticmethod
    def _get_fixed_nargs(parser, key):
        """Return fixed integer nargs for an option key if configured."""
        if parser is None:
            return None
        action = parser._option_string_actions.get(f"--{key}")  # pylint: disable=protected-access
        if action is None:
            return None
        return action.nargs if isinstance(action.nargs, int) else None

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
