"""Definitions for command-line applications."""

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from simtools.application.control import ApplicationContext, _initialize_runtime
from simtools.configuration import configurator
from simtools.configuration.arguments import (
    DATABASE_ARGUMENTS,
    STANDARD_ARGUMENTS,
    ArgumentDefinition,
)

PostParseHook = Callable[[dict, Mapping[str, set], object], None]


@dataclass(frozen=True)
class ApplicationDefinition:
    """Command-line and startup definition for an application."""

    module_name: str
    description: str
    arguments: tuple[ArgumentDefinition, ...] = ()
    database: bool = False
    initialize_output: bool = False
    setup_io_handler: bool = True
    resolve_sim_software_executables: bool = True
    post_parse: PostParseHook | None = None
    defer_required_validation: bool = False

    @classmethod
    def for_module(cls, module_name, **kwargs):
        """Define an application using the loaded module's name and documentation."""
        module = sys.modules.get(module_name)
        if module is None:
            raise ValueError(f"Application module is not loaded: {module_name}")
        if module_name == "__main__":
            module_name = f"simtools.applications.{Path(module.__file__).stem}"
        return cls(module_name=module_name, description=module.__doc__, **kwargs)

    def __post_init__(self):
        """Normalize and validate the immutable definition."""
        object.__setattr__(self, "arguments", tuple(self.arguments))
        self._validate_arguments(self.all_arguments)

    @property
    def label(self):
        """Return the application module label."""
        return self.module_name.rsplit(".", maxsplit=1)[-1]

    @property
    def all_arguments(self):
        """Return standard and application-selected arguments in registration order."""
        database = DATABASE_ARGUMENTS if self.database else ()
        return (*self.arguments, *database, *STANDARD_ARGUMENTS)

    @staticmethod
    def _validate_arguments(arguments):
        names = [argument.name for argument in arguments]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate command-line argument(s): {', '.join(duplicates)}")

    def build_parser(self):
        """Build the complete parser without reading configuration or starting the application."""
        return self._configurator(self.all_arguments).parser

    def _configurator(self, arguments):
        """Return a configurator populated with the given arguments."""
        config_builder = configurator.Configurator(
            label=self.label,
            description=self._description_line(),
        )
        config_builder.parser.add_argument_definitions(arguments)
        return config_builder

    def _description_line(self):
        """Return the first non-empty line of the application description."""
        for line in (self.description or "").splitlines():
            if line.strip():
                return line.strip()
        raise ValueError("Missing or empty application description")

    def _parse(self):
        """Read configuration using this definition's preconfigured parser."""
        runtime_arguments = (
            tuple(argument.without_requiredness() for argument in self.all_arguments)
            if self.defer_required_validation
            else self.all_arguments
        )
        config_builder = self._configurator(runtime_arguments)
        args_dict, db_config = config_builder.configure(initialize_output=self.initialize_output)
        if self.post_parse is not None:
            self.post_parse(args_dict, config_builder.config_sources, config_builder.parser)
        if self.defer_required_validation:
            self._validate_required_values(args_dict, config_builder.parser)
        if not self.database:
            db_config = {}
        return args_dict, db_config

    def _validate_required_values(self, args, parser):
        """Validate declarations deferred until after the post-parse hook."""
        missing = [
            argument.name
            for argument in self.all_arguments
            if argument.kwargs.get("required") and args.get(argument.name) is None
        ]
        exclusive_groups = {}
        for argument in self.all_arguments:
            if argument.exclusive_group_required:
                exclusive_groups.setdefault(argument.exclusive_group, []).append(argument.name)
        missing.extend(
            "/".join(names)
            for names in exclusive_groups.values()
            if not any(args.get(name) not in (None, False) for name in names)
        )
        if missing:
            parser.error(
                "the following arguments are required: "
                + ", ".join(f"--{name}" for name in missing)
            )

    def start(self) -> ApplicationContext:
        """Read configuration and run the standard application startup sequence."""
        args_dict, db_config = self._parse()
        return _initialize_runtime(
            args_dict,
            db_config,
            setup_io_handler=self.setup_io_handler,
            resolve_sim_software_executables=self.resolve_sim_software_executables,
        )
