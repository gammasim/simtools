"""Definitions for command-line applications."""

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from simtools.application.model_reader import create_model_reader_from_configuration
from simtools.configuration import arguments as configurator_arguments
from simtools.configuration import configurator
from simtools.configuration.arguments import STANDARD_ARGUMENTS, ArgumentDefinition
from simtools.context import RunContext

PostParseHook = Callable[[dict, Mapping[str, str], object], None]


@dataclass(frozen=True)
class ApplicationDefinition:
    """Command-line and startup definition for an application."""

    module_name: str
    description: str
    arguments: tuple[ArgumentDefinition, ...] = ()
    model_repository: bool = False
    needs: tuple[str, ...] | None = None
    post_parse: PostParseHook | None = None
    defer_required_validation: bool = False
    usage: str | None = None
    excluded_standard_arguments: tuple[str, ...] = ()

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
        object.__setattr__(
            self,
            "excluded_standard_arguments",
            tuple(self.excluded_standard_arguments),
        )
        needs = ("model",) if self.needs is None and self.model_repository else (self.needs or ())
        object.__setattr__(self, "needs", tuple(needs))
        self._validate_arguments(self.all_arguments)

    @property
    def label(self):
        """Return the application module label."""
        return self.module_name.rsplit(".", maxsplit=1)[-1]

    @property
    def all_arguments(self):
        """Return standard and application-selected arguments in registration order."""
        model_repository = (
            configurator_arguments.MODEL_REPOSITORY_ARGUMENTS if self.model_repository else ()
        )
        standard = tuple(
            argument
            for argument in STANDARD_ARGUMENTS
            if argument.name not in self.excluded_standard_arguments
        )
        return (*self.arguments, *model_repository, *standard)

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
            usage=self.usage,
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
        if self._help_requested():
            parser = self.build_parser()
            parser.print_help()
            parser.exit()

        show_options_requested = self._show_options_requested()
        defer_required_validation = self.defer_required_validation or show_options_requested
        runtime_arguments = (
            tuple(argument.without_requiredness() for argument in self.all_arguments)
            if defer_required_validation
            else self.all_arguments
        )
        config_builder = self._configurator(runtime_arguments)
        args_dict = config_builder.configure(initialize_output="output" in self.needs)
        if args_dict.get("show_options"):
            from simtools.configuration.show_options import (  # pylint: disable=import-outside-toplevel
                handle_show_options,
            )

            try:
                context = RunContext.from_arguments(
                    args_dict,
                    resolve_sim_software_executables="corsika" in self.needs,
                )
                if self._show_options_need_model_reader(args_dict):
                    context = context.replace(
                        model_reader=create_model_reader_from_configuration(args_dict)
                    )
            except (FileNotFoundError, PermissionError, ValueError) as exc:
                config_builder.parser.error(str(exc))
            handle_show_options(args_dict, config_builder.parser, context=context)
        if self.post_parse is not None:
            self.post_parse(args_dict, config_builder.config_sources, config_builder.parser)
        if defer_required_validation:
            self._validate_required_values(args_dict, config_builder.parser)
        args_dict["_metadata_configuration_sources"] = dict(config_builder.config_sources)
        return args_dict

    @staticmethod
    def _help_requested():
        """Return whether the command line explicitly requests help."""
        arguments = sys.argv[1:]
        return not arguments or any(argument in {"--help", "-h"} for argument in arguments)

    @staticmethod
    def _show_options_requested():
        """Return whether the command line requests option discovery."""
        return any(
            argument.split("=", maxsplit=1)[0] == "--show_options" for argument in sys.argv[1:]
        )

    @staticmethod
    def _show_options_need_model_reader(args):
        """Return whether this option discovers values from the model repository."""
        return args.get("show_options") in {"array_layout_name", "model_version"}

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

    def start(self):
        """Read configuration and run the standard application startup sequence."""
        args_dict = self._parse()

        from simtools.application.control import (  # pylint: disable=import-outside-toplevel
            initialize_runtime,
        )

        return initialize_runtime(
            args_dict,
            needs=self.needs,
        )
