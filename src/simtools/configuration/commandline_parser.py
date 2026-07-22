"""Command-line parser for explicitly defined applications."""

import argparse


class CommandLineParser(argparse.ArgumentParser):
    """Argument parser that registers explicit argument definitions."""

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
            container = self
            if argument.group is not None:
                container = display_groups.setdefault(
                    argument.group, self.add_argument_group(argument.group)
                )
            if argument.exclusive_group is not None:
                group_key = (argument.group, argument.exclusive_group)
                required = argument.exclusive_group_required and (
                    argument.required_unless not in self._option_string_actions
                )
                if group_key in exclusive_groups:
                    exclusive_container, existing_required = exclusive_groups[group_key]
                    if existing_required != required:
                        raise ValueError(
                            f"Conflicting required state for exclusive group "
                            f"{argument.exclusive_group!r}"
                        )
                    container = exclusive_container
                else:
                    container = container.add_mutually_exclusive_group(required=required)
                    exclusive_groups[group_key] = (container, required)
            container.add_argument(f"--{argument.name}", **argument.kwargs)
