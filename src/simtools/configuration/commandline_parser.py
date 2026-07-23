"""Command-line parser for explicitly defined applications."""

import argparse


class CommandLineParser(argparse.ArgumentParser):
    """Argument parser that registers explicit argument definitions."""

    def __init__(self, *args, **kwargs):
        """Initialize an empty parser and its declaration metadata."""
        super().__init__(*args, **kwargs)
        self.preserve_by_version = set()

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
            container = self._display_container(argument, display_groups)
            if argument.preserve_by_version:
                self.preserve_by_version.add(argument.name)
            container = self._exclusive_container(argument, container, exclusive_groups)
            container.add_argument(f"--{argument.name}", **argument.kwargs)

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
