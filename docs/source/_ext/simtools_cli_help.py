"""
Render filtered application CLI help from the real simtools parsers.

Example
-------
Render the CLI reference for an application while hiding the default repeated groups:

.. code-block:: rst

    .. simtools-cli-help::
       :module: plot_array_layout

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
        return error.parser
    finally:
        module.build_application = original_build_application

    raise CliHelpError(f"Failed to capture parser from application main(): {module.__name__}")


def load_application_parser(module_name: str, prog: str | None = None):
    """Load an application parser from the target module."""
    module = importlib.import_module(_resolve_module_name(module_name))
    return _capture_parser_from_main(module, prog=prog)


def render_filtered_help(parser, hidden_groups: set[str]) -> str:
    """Render parser help while suppressing selected argparse groups."""
    formatter = parser._get_formatter()  # pylint: disable=protected-access
    visible_actions = []
    for group in parser._action_groups:  # pylint: disable=protected-access
        if group.title in hidden_groups:
            continue
        visible_actions.extend(
            # argparse stores grouped actions on this internal attribute;
            # no public equivalent exists.
            # pylint: disable=protected-access
            [action for action in group._group_actions if action.help is not argparse.SUPPRESS]
        )

    formatter.add_usage(
        parser.usage,
        visible_actions,
        parser._mutually_exclusive_groups,  # pylint: disable=protected-access
    )
    formatter.add_text(parser.description)

    for group in parser._action_groups:  # pylint: disable=protected-access
        if group.title in hidden_groups:
            continue

        group_actions = [
            # argparse stores grouped actions on this internal attribute;
            # no public equivalent exists.
            # pylint: disable=protected-access
            action
            for action in group._group_actions
            if action.help is not argparse.SUPPRESS
        ]
        if not group_actions:
            continue

        formatter.start_section(group.title)
        formatter.add_text(group.description)
        formatter.add_arguments(group_actions)
        formatter.end_section()

    formatter.add_text(parser.epilog)
    return formatter.format_help()


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
            parser = load_application_parser(options.module_name, prog=options.prog)
        except (CliHelpError, ImportError, ValueError) as error:
            raise self.error(str(error)) from error

        rendered_help = render_filtered_help(parser, options.hidden_groups)
        section = nodes.section(ids=[nodes.make_id("command-line-arguments")])
        section += nodes.title(text="Command line arguments")
        help_block = nodes.literal_block(text=rendered_help)
        help_block["language"] = "text"
        section += help_block
        return [section]

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
