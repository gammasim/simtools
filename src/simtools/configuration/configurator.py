"""Application configuration."""

import argparse
import logging
import sys

import simtools.configuration.commandline_parser as argparser
from simtools import dependency_versions
from simtools.configuration.argument_helpers import (
    coerce,
    configuration_from_environment,
    configuration_from_file,
    normalize_argument_aliases,
    validate_site_telescope_consistency,
)
from simtools.utils import general as gen


class Configurator:
    """
    Application configuration.

    Allow to set configuration parameters by

    - command line arguments
    - configuration file
    - environmental variables

    Assigns unique ACTIVITY_ID to this configuration (uuid).

    Configuration parameter names are converted always to lower case.

    Parameters
    ----------
    label: str
        Class label.
    description: str
        Text displayed as description.
    usage: str
        Application usage description.
    """

    def __init__(self, label=None, description=None, usage=None, use_dependency_defaults=True):
        """Initialize Configurator."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Configuration")

        self.label = label
        self.config = {}
        self.use_dependency_defaults = use_dependency_defaults
        self.config_sources = {}
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
        dict
            Parsed application configuration.
        """
        cli_arglist = self._get_cli_arglist()
        self._validate_cli_aliases(cli_arglist)
        cli_config = self.parser.parse_explicit_args(cli_arglist)
        config_file = cli_config.get("config") or self._option_value(cli_arglist, "--config")
        env_file = cli_config.get("env_file", ".env")
        env_config = normalize_argument_aliases(self._config_from_env(env_file), self.parser)
        file_config = normalize_argument_aliases(self._config_from_file(config_file), self.parser)
        default_config = self.parser.declared_defaults()
        if self.use_dependency_defaults:
            default_config.update(self._dependency_defaults(default_config))
        self._merge_sources(
            (
                ("default", default_config),
                ("environment", env_config),
                ("yaml", file_config),
                ("cli", cli_config),
            )
        )
        self._validate_merged_required()
        validate_site_telescope_consistency(self.config, self.parser)
        if self.config.get("activity_id") is None:
            self.config["activity_id"] = gen.get_uuid()
        if self.config.get("label") is None:
            self.config["label"] = self.label
        self._initialize_model_versions()
        if initialize_output:
            self._initialize_output()
        self.config["application_label"] = self.config.get("application_label") or self.label
        return self.config

    def _merge_sources(self, sources):
        """Merge source values in increasing priority and record the source of each value."""
        definitions = self.parser.argument_definitions
        destinations = {action.dest for action in self.parser._actions}  # pylint: disable=protected-access
        self.config = {}
        self.config_sources = {}
        superseded = self._superseded_by_cli(dict(sources).get("cli", {}))
        for source, values in sources:
            for key, value in values.items():
                if key not in destinations:
                    self.parser.error(f"unrecognized arguments: --{key}")
                if source not in ("default", "cli") and key in superseded:
                    continue
                self.config[key] = self._source_value(definitions.get(key), value, source)
                self.config_sources[key] = source

    def _source_value(self, definition, value, source):
        """Return a source value converted by its argument declaration."""
        if value == "None":
            return None
        if source == "cli" or definition is None:
            return value
        return self._convert_external_value(definition, value, source)

    def _superseded_by_cli(self, cli_config):
        """Return keys whose environment and file values a CLI option replaces.

        These are the CLI keys and the other members of their mutually exclusive groups.
        """
        superseded = set(cli_config)
        for group in self.parser._mutually_exclusive_groups:  # pylint: disable=protected-access
            members = {action.dest for action in group._group_actions}  # pylint: disable=protected-access
            if members & cli_config.keys():
                superseded.update(members)
        return superseded

    def _convert_external_value(self, definition, value, source):
        """Convert a value from defaults, environment or configuration file.

        Arguments with a custom argparse action are parsed by that action, so that a value
        means the same from every source.
        """
        if source == "default" and value == "":
            return value
        action = definition.kwargs.get("action")
        if not isinstance(action, type) or value is None:
            return coerce(definition, value, source)
        if isinstance(value, bool):
            return value
        tokens = [str(item) for item in value] if isinstance(value, list | tuple) else [str(value)]
        return self.parser.parse_explicit_args([f"--{definition.name}", *tokens])[definition.name]

    def _validate_merged_required(self):
        """Validate required arguments and exclusive groups after source merging."""
        missing = [
            f"--{definition.name}"
            for definition in self.parser.argument_definitions.values()
            if definition.kwargs.get("required") and self.config.get(definition.name) is None
        ]
        groups = {}
        for definition in self.parser.argument_definitions.values():
            if definition.exclusive_group is not None:
                group_key = (definition.group, definition.exclusive_group)
                groups.setdefault(group_key, []).append(definition)
        for definitions in groups.values():
            selected = [
                definition.name
                for definition in definitions
                if self.config.get(definition.name) not in (None, False)
            ]
            if len(selected) > 1:
                self.parser.error(
                    f"argument --{selected[1]}: not allowed with argument --{selected[0]}"
                )
            if definitions[0].exclusive_group_required and not selected:
                missing.append("/".join(f"--{item.name}" for item in definitions))
        if missing:
            self.parser.error("the following arguments are required: " + ", ".join(missing))

    @staticmethod
    def _dependency_defaults(parser_defaults):
        """Return catalog-managed defaults supported by this parser."""
        if not {"simulation_models_git_revision"} & parser_defaults.keys():
            return {}
        catalog = dependency_versions.load_dependency_catalog()
        environment = dependency_versions.dependency_catalog_environment(catalog)
        return {
            "simulation_models_git_revision": environment["SIMTOOLS_SIMULATION_MODELS_GIT_REVISION"]
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

    def _validate_cli_aliases(self, arg_list):
        """Reject conflicting canonical and deprecated CLI values."""
        for action in self.parser._actions:  # pylint: disable=protected-access
            values = {
                value
                for option in action.option_strings
                if (value := self._option_value(arg_list, option)) is not None
            }
            if len(values) > 1:
                raise ValueError(f"Conflicting values for {action.dest} command-line aliases.")

    def _config_from_file(self, config_file):
        """Read the YAML configuration through the shared conversion helper."""
        return configuration_from_file(config_file, self.parser)

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
        return configuration_from_environment(self.parser, env_file)

    def _initialize_model_versions(self):
        """Initialize model versions."""
        if (
            self.config.get("model_version", None)
            and isinstance(self.config["model_version"], list)
            and len(self.config["model_version"]) == 1
        ):
            self.config["model_version"] = self.config["model_version"][0]

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
