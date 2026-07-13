"""Command line parser for applications."""

import argparse

from simtools.configuration import commandline_parameters

DEFAULT_ARGUMENT_GROUPS = {
    "configuration": "CONFIGURATION_ARGS",
    "database configuration": "DB_CONFIG_ARGS",
    "execution": "EXECUTION_ARGS",
    "output": "OUTPUT_ARGS",
    "paths": "PATH_ARGS",
    "run time": "RUN_TIME_ARGS",
    "user": "USER_ARGS",
}

SIMULATION_CONFIGURATION_GROUPS = (
    ("simulation configuration", commandline_parameters.get_corsika_configuration_args),
    ("shower parameters", lambda: commandline_parameters.PARAMETER_DEFINITIONS["SHOWER_ARGS"]),
    ("corsika configuration", lambda: commandline_parameters.PARAMETER_DEFINITIONS["CORSIKA_ARGS"]),
)

SIMULATION_MODEL_BASE_PARAMETERS = (
    "model_version",
    "parameter_version",
    "updated_parameter_version",
)
SIMULATION_MODEL_SITE_DEPENDENCIES = {"layout", "layout_file", "telescope", "telescopes"}
SIMULATION_MODEL_LAYOUT_OPTIONS = {"layout", "layout_file"}
SIMULATION_MODEL_LAYOUT_BASE_PARAMETERS = ("array_layout_name", "array_element_list")
SIMULATION_MODEL_LAYOUT_OPTIONAL_PARAMETERS = {
    "layout_file": ("array_layout_file",),
    "plot_all_layouts": ("plot_all_layouts",),
}
SIMULATION_MODEL_LAYOUT_POST_PARAMETERS = {
    "layout_parameter_file": ("array_layout_parameter_file",),
}


class CommandLineParser(argparse.ArgumentParser):
    """
    Command line parser for applications.

    Wrapper around standard python argparse.ArgumentParser.

    Command line arguments should be given in snake_case, e.g. 'input_meta'.

    Parameters
    ----------
    argparse.ArgumentParser class
        Object for parsing command line strings into Python objects. For a list of keywords, please\
        refer to argparse.ArgumentParser documentation.
    """

    def initialize_default_arguments(
        self,
        paths=True,
        output=False,
        simulation_model=None,
        simulation_configuration=None,
        db_config=False,
    ):
        """
        Initialize default arguments used by all applications (e.g., log level or test flag).

        Parameters
        ----------
        paths: bool
            Add path configuration to list of args.
        output: bool
            Add output file configuration to list of args.
        simulation_model: list
            List of simulation model configuration parameters to add to list of args
            (use: 'model_version', 'telescope', 'site')
        simulation_configuration: dict
            Dict of simulation software configuration parameters to add to list of args.
        db_config: bool
            Add database configuration parameters to list of args.
        """
        self.initialize_simulation_model_arguments(simulation_model)
        self.initialize_simulation_configuration_arguments(simulation_configuration)

        if db_config:
            self.initialize_named_argument_group("database configuration")
        if paths:
            self.initialize_named_argument_group("paths")
        if output:
            self.initialize_named_argument_group("output")

        for group_name in ("configuration", "execution", "run time", "user"):
            self.initialize_named_argument_group(group_name)

    def initialize_simulation_model_arguments(self, model_options):
        """
        Initialize default arguments for simulation model definition.

        Note that the model version is always required.

        Parameters
        ----------
        model_options: list
            Options to be set: "telescope", "site", "layout", "layout_file",
            "updated_parameter_version"
        """
        if model_options is None:
            return

        requested = set(model_options)
        definitions = commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"]
        group = self.add_argument_group("simulation model")

        self._add_parameters(
            group,
            self._simulation_model_direct_parameters(requested),
            definitions,
        )

        if requested & SIMULATION_MODEL_LAYOUT_OPTIONS:
            self._add_simulation_model_layout_parameters(group, requested, definitions)

        self._add_parameters(group, ["ignore_missing_design_model"], definitions)

    def initialize_simulation_configuration_arguments(self, simulation_configuration):
        """
        Initialize default arguments for simulation configuration and simulation software.

        Parameters
        ----------
        simulation_configuration: dict
            Dict of simulation software configuration parameters.
        """
        if simulation_configuration is None:
            return

        if "software" in simulation_configuration:
            self._initialize_named_parameters_group(
                "simulation software",
                ["simulation_software"],
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_SOFTWARE_ARGS"],
            )

        if "corsika_configuration" in simulation_configuration:
            selected = simulation_configuration["corsika_configuration"]
            for group_name, definitions_factory in SIMULATION_CONFIGURATION_GROUPS:
                self._initialize_named_parameters_group(
                    group_name,
                    selected,
                    definitions_factory(),
                )

        if "sim_telarray_configuration" in simulation_configuration:
            self._initialize_named_parameters_group(
                "sim_telarray configuration",
                simulation_configuration["sim_telarray_configuration"],
                commandline_parameters.PARAMETER_DEFINITIONS["SIMTEL_ARGS"],
            )

    def add_parameter_from_definition(self, container, name, definition):
        """Add one argument from a parameter-definition dictionary."""
        argparse_kwargs, doc_metadata = _split_argument_metadata(definition)
        action = container.add_argument(f"--{name}", **argparse_kwargs)
        action.simtools_doc = doc_metadata["doc"]
        action.simtools_doc_hidden = doc_metadata["doc_hidden"]
        action.simtools_scopes = doc_metadata["scopes"]
        return action

    def initialize_named_argument_group(self, group_name):
        """Initialize one predefined argument group by its display name."""
        self._initialize_named_parameters_group(
            group_name,
            ["all"],
            commandline_parameters.PARAMETER_DEFINITIONS[DEFAULT_ARGUMENT_GROUPS[group_name]],
        )

    def initialize_application_argument_group(self, selected_parameters, available_parameters=None):
        """Initialize application-specific arguments."""
        self.initialize_argument_group("application", selected_parameters, available_parameters)

    def initialize_argument_group(self, group_name, selected_parameters, available_parameters=None):
        """Initialize a group of arguments from a parameter-definition dictionary."""
        definitions = (
            available_parameters or commandline_parameters.PARAMETER_DEFINITIONS["APPLICATION_ARGS"]
        )
        self._initialize_named_parameters_group(group_name, selected_parameters, definitions)

    def _initialize_named_parameters_group(self, group_name, selected_parameters, definitions):
        """Create one argparse group and add the selected parameters to it."""
        group = self.add_argument_group(group_name)
        self._add_parameters(group, selected_parameters, definitions)
        return group

    def _add_parameters(self, container, selected_parameters, definitions):
        """Add selected parameters to an argparse container."""
        parameter_names = (
            definitions.keys() if "all" in selected_parameters else selected_parameters
        )
        for parameter_name in parameter_names:
            definition = definitions.get(parameter_name)
            if definition is not None:
                self.add_parameter_from_definition(container, parameter_name, definition)

    def _simulation_model_direct_parameters(self, requested):
        """Return the ordered direct simulation-model parameters to add."""
        direct_parameters = [name for name in SIMULATION_MODEL_BASE_PARAMETERS if name in requested]
        direct_parameters.append("overwrite_model_parameters")
        if requested & SIMULATION_MODEL_SITE_DEPENDENCIES or "site" in requested:
            direct_parameters.append("site")
        if "telescope" in requested:
            direct_parameters.append("telescope")
        if "telescopes" in requested:
            direct_parameters.append("telescopes")
        return direct_parameters

    def _add_simulation_model_layout_parameters(self, group, requested, definitions):
        """Add the layout-selection arguments to the simulation-model group."""
        layout_group = group.add_mutually_exclusive_group(
            required="--list_available_layouts" not in self._option_string_actions
        )
        layout_parameters = list(SIMULATION_MODEL_LAYOUT_BASE_PARAMETERS)
        for option_name, parameter_names in SIMULATION_MODEL_LAYOUT_OPTIONAL_PARAMETERS.items():
            if option_name in requested:
                layout_parameters.extend(parameter_names)
        self._add_parameters(layout_group, layout_parameters, definitions)

        for option_name, parameter_names in SIMULATION_MODEL_LAYOUT_POST_PARAMETERS.items():
            if option_name in requested:
                self._add_parameters(group, parameter_names, definitions)


def _split_argument_metadata(definition):
    """Split a parameter definition into argparse kwargs and documentation metadata."""
    doc_metadata = {
        "doc": definition.get("doc", definition.get("help")),
        "doc_hidden": definition.get("doc_hidden", definition.get("help") is argparse.SUPPRESS),
        "scopes": definition.get("scopes"),
    }
    argparse_kwargs = {k: v for k, v in definition.items() if k not in doc_metadata}
    return argparse_kwargs, doc_metadata
