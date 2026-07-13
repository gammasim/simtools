"""
Render filtered application CLI help from simtools parsers.

Example
-------
Render the CLI reference for an application while hiding the default repeated groups:

.. code-block:: rst

    .. simtools-cli-help::
       :module: simtools.applications.plot_array_layout

Render the CLI reference while keeping the ``paths`` group visible:

.. code-block:: rst

    .. simtools-cli-help::
       :module: simtools.applications.plot_array_layout
       :show-groups: paths
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass

from docutils import nodes
from docutils.parsers.rst import Directive, directives

from simtools.application_control import build_application_parser

try:
    from sphinx.util.docutils import SphinxDirective
except ModuleNotFoundError:  # pragma: no cover - fallback for non-Sphinx imports
    SphinxDirective = Directive

DEFAULT_HIDDEN_GROUPS = {
    "configuration",
    "execution",
    "paths",
    "run time",
    "user",
}


class CliHelpError(ValueError):
    """Raised when CLI help cannot be rendered for an application module."""


class _ParserCapturedError(RuntimeError):
    """Internal exception used to short-circuit application main functions."""

    def __init__(self, parser):
        super().__init__("parser captured")
        self.parser = parser


@dataclass(frozen=True)
class CliHelpOptions:
    """Options controlling CLI-help rendering."""

    module_name: str
    prog: str | None
    hidden_groups: set[str]


@dataclass(frozen=True)
class CliInspection:
    """Captured CLI construction context for one application module."""

    parser: argparse.ArgumentParser


def _split_csv_option(value: str | None) -> set[str]:
    """Parse a comma-separated directive option into a normalized set."""
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _resolve_module_name(module_name: str) -> str:
    """Resolve short application-module names to their package path."""
    if "." in module_name:
        return module_name
    return f"simtools.applications.{module_name}"


def _capture_parser_from_main(module, prog: str | None):
    """Execute module.main() with a patched build_application that captures the parser."""
    original_build_application = getattr(module, "build_application", None)
    if original_build_application is None:
        raise CliHelpError(f"Application module has no build_application import: {module.__name__}")

    def _fake_build_application(*_args, **kwargs):
        initialization_kwargs = kwargs.get("initialization_kwargs")
        if initialization_kwargs is None and kwargs.get("parse_function") is not None:
            initialization_kwargs = getattr(module, "_INITIALIZATION_KWARGS", {})

        parser = build_application_parser(
            application_path=getattr(module, "__file__", None),
            description=getattr(module, "__doc__", None),
            add_arguments_function=getattr(module, "_add_arguments", None),
            application_argument_definitions=kwargs.get(
                "application_argument_definitions",
                getattr(module, "_APPLICATION_ARG_DEFINITIONS", None),
            ),
            initialization_kwargs=initialization_kwargs,
            usage=kwargs.get("usage"),
            epilog=kwargs.get("epilog"),
        )
        if prog is not None:
            parser.prog = prog
        raise _ParserCapturedError(parser)

    module.build_application = _fake_build_application
    try:
        module.main()
    except _ParserCapturedError as error:
        return CliInspection(parser=error.parser)
    finally:
        module.build_application = original_build_application

    raise CliHelpError(f"Failed to capture parser from application main(): {module.__name__}")


def load_application_parser(module_name: str, prog: str | None = None):
    """Load an application parser from the target module."""
    module = importlib.import_module(_resolve_module_name(module_name))
    return _capture_parser_from_main(module, prog=prog)


def _iter_visible_group_actions(parser):
    """Yield `(group_title, action)` pairs for doc-visible parser actions."""
    for group in parser._action_groups:  # pylint: disable=protected-access
        for action in group._group_actions:  # pylint: disable=protected-access
            if action.help is argparse.SUPPRESS:
                continue
            yield group.title, action


def _group_actions_for_docs(parser, hidden_groups):
    """Group visible parser actions by documentation section title."""
    grouped_actions = {}
    for group_title, action in _iter_visible_group_actions(parser):
        if group_title in hidden_groups:
            continue
        grouped_actions.setdefault(group_title, []).append(action)
    return grouped_actions


def _format_term(action):
    """Format the display term for one CLI parameter."""
    long_option = next((opt for opt in action.option_strings if opt.startswith("--")), None)
    return long_option or action.dest


def _format_default(action):
    """Format a default value when it is useful in the docs."""
    if action.default in (None, False, argparse.SUPPRESS):
        return None
    if isinstance(action.default, list) and len(action.default) == 0:
        return None
    return str(action.default)


def _build_description(action):
    """Build paragraph nodes for one CLI parameter description."""
    content = [nodes.paragraph(text=action.help)]
    notes = []
    if getattr(action, "required", False):
        notes.append("Required.")
    default = _format_default(action)
    if default is not None:
        notes.append(f"Default: {default}.")
    if action.choices:
        choices = ", ".join(str(choice) for choice in action.choices)
        notes.append(f"Choices: {choices}.")
    if notes:
        content.append(nodes.paragraph(text=" ".join(notes)))
    return content


def render_native_cli_docs(inspection: CliInspection, hidden_groups: set[str]):
    """Render native RST nodes for one application CLI."""
    section = nodes.section(ids=[nodes.make_id("command-line-arguments")])
    section += nodes.title(text="Command line arguments")

    grouped_actions = _group_actions_for_docs(inspection.parser, hidden_groups)
    for group_title, actions in grouped_actions.items():
        subgroup = nodes.section(ids=[nodes.make_id(group_title)])
        subgroup += nodes.title(text=group_title)
        definition_list = nodes.definition_list()
        for action in actions:
            item = nodes.definition_list_item()
            item += nodes.term(text=_format_term(action))
            definition = nodes.definition()
            for node in _build_description(action):
                definition += node
            item += definition
            definition_list += item
        subgroup += definition_list
        section += subgroup
    return [section]


class SimtoolsCliHelpDirective(SphinxDirective):
    """Directive that renders filtered CLI help for simtools applications."""

    has_content = False
    option_spec = {
        "module": directives.unchanged_required,
        "prog": directives.unchanged,
        "hide-groups": directives.unchanged,
        "show-groups": directives.unchanged,
    }

    def run(self):
        """Render the directive content."""
        options = self._parse_options()
        try:
            inspection = load_application_parser(options.module_name, prog=options.prog)
        except (CliHelpError, ImportError, ValueError) as error:
            raise self.error(str(error)) from error

        return render_native_cli_docs(inspection, options.hidden_groups)

    def _parse_options(self) -> CliHelpOptions:
        """Parse directive options into a structured configuration."""
        hidden_groups = DEFAULT_HIDDEN_GROUPS | _split_csv_option(self.options.get("hide-groups"))
        hidden_groups -= _split_csv_option(self.options.get("show-groups"))
        return CliHelpOptions(
            module_name=self.options["module"],
            prog=self.options.get("prog"),
            hidden_groups=hidden_groups,
        )


def setup(app):
    """Register the Sphinx directive."""
    app.add_directive("simtools-cli-help", SimtoolsCliHelpDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
