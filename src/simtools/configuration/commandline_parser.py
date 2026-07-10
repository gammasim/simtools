"""Command line parser for applications."""

import argparse

from simtools.configuration import commandline_parameters


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
            self.initialize_db_config_arguments()
        if paths:
            self.initialize_path_arguments()
        if output:
            self.initialize_output_arguments()

        self.initialize_config_files()
        self.initialize_application_execution_arguments()
        self.initialize_run_time()
        self.initialize_user_arguments()

    def initialize_config_files(self):
        """Initialize configuration files."""
        self.initialize_argument_group(
            "configuration",
            ["all"],
            commandline_parameters.PARAMETER_DEFINITIONS["CONFIGURATION_ARGS"],
        )

    def initialize_path_arguments(self):
        """Initialize paths."""
        self.initialize_argument_group(
            "paths", ["all"], commandline_parameters.PARAMETER_DEFINITIONS["PATH_ARGS"]
        )

    def initialize_output_arguments(self):
        """Initialize application output files(s)."""
        self.initialize_argument_group(
            "output", ["all"], commandline_parameters.PARAMETER_DEFINITIONS["OUTPUT_ARGS"]
        )

    def initialize_run_time(self):
        """Initialize run time arguments."""
        self.initialize_argument_group(
            "run time", ["all"], commandline_parameters.PARAMETER_DEFINITIONS["RUN_TIME_ARGS"]
        )

    def initialize_application_execution_arguments(self):
        """Initialize application execution arguments."""
        self.initialize_argument_group(
            "execution", ["all"], commandline_parameters.PARAMETER_DEFINITIONS["EXECUTION_ARGS"]
        )

    def initialize_user_arguments(self):
        """Initialize user arguments."""
        self.initialize_argument_group(
            "user", ["all"], commandline_parameters.PARAMETER_DEFINITIONS["USER_ARGS"]
        )

    def initialize_db_config_arguments(self):
        """Initialize DB configuration parameters."""
        self.initialize_argument_group(
            "database configuration",
            ["all"],
            commandline_parameters.PARAMETER_DEFINITIONS["DB_CONFIG_ARGS"],
        )

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

        _job_group = self.add_argument_group("simulation model")
        if "model_version" in model_options:
            self.add_parameter_from_definition(
                _job_group,
                "model_version",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                    "model_version"
                ],
            )
        if "parameter_version" in model_options:
            self.add_parameter_from_definition(
                _job_group,
                "parameter_version",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                    "parameter_version"
                ],
            )
        if "updated_parameter_version" in model_options:
            self.add_parameter_from_definition(
                _job_group,
                "updated_parameter_version",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                    "updated_parameter_version"
                ],
            )
        self.add_parameter_from_definition(
            _job_group,
            "overwrite_model_parameters",
            commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                "overwrite_model_parameters"
            ],
        )

        if any(
            option in model_options
            for option in ["site", "telescope", "telescopes", "layout", "layout_file"]
        ):
            self._add_model_option_site(_job_group)

        if "telescope" in model_options:
            self.add_parameter_from_definition(
                _job_group,
                "telescope",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"]["telescope"],
            )
        if "telescopes" in model_options:
            self.add_parameter_from_definition(
                _job_group,
                "telescopes",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"]["telescopes"],
            )
        if "layout" in model_options or "layout_file" in model_options:
            self._add_model_option_layout(
                job_group=_job_group,
                model_options=model_options,
                # layout info is always required for layout related tasks with the exception
                # of listing the available layouts in the DB
                required="--list_available_layouts" not in self._option_string_actions,
            )

        self.add_parameter_from_definition(
            _job_group,
            "ignore_missing_design_model",
            commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                "ignore_missing_design_model"
            ],
        )

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
            _grp = self.add_argument_group("simulation software")
            self.add_parameter_from_definition(
                _grp,
                "simulation_software",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_SOFTWARE_ARGS"][
                    "simulation_software"
                ],
                group_name="simulation software",
            )
        if "corsika_configuration" in simulation_configuration:
            self.initialize_argument_group(
                "simulation configuration",
                simulation_configuration["corsika_configuration"],
                commandline_parameters.get_corsika_configuration_args(),
            )
            self.initialize_argument_group(
                "shower parameters",
                simulation_configuration["corsika_configuration"],
                commandline_parameters.PARAMETER_DEFINITIONS["SHOWER_ARGS"],
            )
            self.initialize_argument_group(
                "corsika configuration",
                simulation_configuration["corsika_configuration"],
                commandline_parameters.PARAMETER_DEFINITIONS["CORSIKA_ARGS"],
            )
        if "sim_telarray_configuration" in simulation_configuration:
            self.initialize_argument_group(
                "sim_telarray configuration",
                simulation_configuration["sim_telarray_configuration"],
                commandline_parameters.PARAMETER_DEFINITIONS["SIMTEL_ARGS"],
            )

    def add_parameter_from_definition(self, container, name, definition, group_name=None):
        """Add one argument from a parameter-definition dictionary."""
        argparse_kwargs, doc_metadata = _split_argument_metadata(definition)
        action = container.add_argument(f"--{name}", **argparse_kwargs)
        action.simtools_doc = doc_metadata["doc"]
        action.simtools_doc_group = doc_metadata["doc_group"] or group_name
        action.simtools_doc_groups = doc_metadata["doc_groups"]
        action.simtools_doc_hidden = doc_metadata["doc_hidden"]
        action.simtools_scopes = doc_metadata["scopes"]
        return action

    def initialize_application_argument_group(self, selected_parameters, available_parameters=None):
        """Initialize application-specific arguments."""
        self.initialize_argument_group("application", selected_parameters, available_parameters)

    def initialize_argument_group(self, group_name, selected_parameters, available_parameters=None):
        """Initialize a group of arguments from a parameter-definition dictionary."""
        if available_parameters is None:
            available_parameters = commandline_parameters.PARAMETER_DEFINITIONS["APPLICATION_ARGS"]

        configuration_group = self.add_argument_group(group_name)

        if "all" in selected_parameters:
            selected_parameters = available_parameters.keys()

        for param in selected_parameters:
            try:
                self.add_parameter_from_definition(
                    configuration_group,
                    param,
                    available_parameters[param],
                    group_name=group_name,
                )
            except KeyError:
                pass

    def _add_model_option_layout(self, job_group, model_options, required=True):
        """
        Add layout option to the job group.

        Parameters
        ----------
        job_group: argparse.ArgumentParser
            Job group
        model_options: list
            List of model options.

        Returns
        -------
        argparse.ArgumentParser
        """
        _layout_group = job_group.add_mutually_exclusive_group(required=required)
        self.add_parameter_from_definition(
            _layout_group,
            "array_layout_name",
            commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                "array_layout_name"
            ],
            group_name="simulation model",
        )
        self.add_parameter_from_definition(
            _layout_group,
            "array_element_list",
            commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                "array_element_list"
            ],
            group_name="simulation model",
        )
        if "layout_file" in model_options:
            self.add_parameter_from_definition(
                _layout_group,
                "array_layout_file",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                    "array_layout_file"
                ],
                group_name="simulation model",
            )
        if "layout_parameter_file" in model_options:
            self.add_parameter_from_definition(
                job_group,
                "array_layout_parameter_file",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                    "array_layout_parameter_file"
                ],
                group_name="simulation model",
            )
        if "plot_all_layouts" in model_options:
            self.add_parameter_from_definition(
                _layout_group,
                "plot_all_layouts",
                commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"][
                    "plot_all_layouts"
                ],
                group_name="simulation model",
            )
        return job_group

    def _add_model_option_site(self, job_group):
        """
        Add site option to the job group.

        Parameters
        ----------
        job_group: argparse.ArgumentParser
            Job group

        Returns
        -------
        argparse.ArgumentParser
        """
        self.add_parameter_from_definition(
            job_group,
            "site",
            commandline_parameters.PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"]["site"],
        )
        return job_group


def _split_argument_metadata(definition):
    """Split a parameter definition into argparse kwargs and documentation metadata."""
    doc_metadata = {
        "doc": definition.get("doc", definition.get("help")),
        "doc_group": definition.get("doc_group"),
        "doc_groups": definition.get("doc_groups", {}),
        "doc_hidden": definition.get("doc_hidden", definition.get("help") is argparse.SUPPRESS),
        "scopes": definition.get("scopes"),
    }
    argparse_kwargs = {k: v for k, v in definition.items() if k not in doc_metadata}
    return argparse_kwargs, doc_metadata


def resolve_doc_group_for_scope(action, scope=None, fallback=None):
    """Resolve a documentation group from action metadata for a specific application scope."""
    doc_groups = getattr(action, "simtools_doc_groups", {}) or {}
    if scope and scope in doc_groups:
        return doc_groups[scope]
    return getattr(action, "simtools_doc_group", None) or fallback
