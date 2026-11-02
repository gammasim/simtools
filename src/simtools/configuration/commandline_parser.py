"""Command-line parser for explicitly defined applications."""

import argparse


class CommandLineParser(argparse.ArgumentParser):
    """Argument parser that registers explicit argument definitions."""

    def __init__(self, *args, **kwargs):
        """Initialize an empty parser and its declaration metadata."""
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)
        self.preserve_by_version = set()
        self.argument_definitions = {}

    def declared_defaults(self):
        """Return declared argument defaults and argparse action defaults."""
        defaults = {}
        action_defaults = {"store_true": False, "store_false": True}
        for name, definition in self.argument_definitions.items():
            defaults[name] = definition.kwargs.get(
                "default", action_defaults.get(definition.kwargs.get("action"))
            )
        for action in self._actions:  # pylint: disable=protected-access
            if action.default is not argparse.SUPPRESS:
                defaults.setdefault(action.dest, action.default)
        return defaults

    def parse_explicit_args(self, args):
        """Return only the options given on the command line, converted by their types.

        Defaults are not filled in and required arguments are not checked, so that values
        from other sources can be merged first.

        Parameters
        ----------
        args : list of str
            Command-line arguments.

        Returns
        -------
        dict
            Explicitly given options.
        """
        actions = [action for action in self._actions if action.dest != argparse.SUPPRESS]
        saved_actions = [(action, action.default, action.required) for action in actions]
        groups = self._mutually_exclusive_groups
        saved_groups = [(group, group.required) for group in groups]
        try:
            for action in actions:
                action.default = argparse.SUPPRESS
                action.required = False
            for group in groups:
                group.required = False
            return vars(self.parse_args(args))
        finally:
            for action, default, required in saved_actions:
                action.default = default
                action.required = required
            for group, required in saved_groups:
                group.required = required

    def add_argument_definitions(self, arguments):
        """Register a sequence of explicit argument definitions.

        Parameters
        ----------
        arguments : iterable of ArgumentDefinition
            Arguments to register in declaration order.
        """
        display_groups = {}
        exclusive_groups = {}
        for argument in arguments:
            self.argument_definitions[argument.name] = argument
            container = self._display_container(argument, display_groups)
            if argument.preserve_by_version:
                self.preserve_by_version.add(argument.name)
            container = self._exclusive_container(argument, container, exclusive_groups)
            container.add_argument(
                f"--{argument.name}",
                *(f"--{alias}" for alias in argument.aliases),
                **argument.kwargs,
            )

    def _display_container(self, argument, display_groups):
        """Return the parser or shared display group for an argument."""
        if argument.group is None:
            return self
        if argument.group not in display_groups:
            display_groups[argument.group] = self.add_argument_group(argument.group)
        return display_groups[argument.group]

    @staticmethod
    def _exclusive_container(argument, container, exclusive_groups):
        """Return the shared mutually exclusive group for an argument."""
        if argument.exclusive_group is None:
            return container

        group_key = (argument.group, argument.exclusive_group)
        required = argument.exclusive_group_required
        if group_key not in exclusive_groups:
            exclusive_container = container.add_mutually_exclusive_group(required=required)
            exclusive_groups[group_key] = (exclusive_container, required)
            return exclusive_container

        exclusive_container, existing_required = exclusive_groups[group_key]
        if existing_required != required:
            raise ValueError(
                f"Conflicting required state for exclusive group {argument.exclusive_group!r}"
            )
        return exclusive_container
