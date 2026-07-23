"""
Render filtered application CLI help from simtools parsers.

Example
-------
Render the CLI reference for an application while hiding the default repeated groups:

.. code-block:: rst

    .. simtools-cli-help::
       :application: plot_array_layout

Render the CLI reference while keeping the ``paths`` group visible:

.. code-block:: rst

    .. simtools-cli-help::
       :application: plot_array_layout
       :show-groups: paths

Render the CLI reference under a heading supplied by the surrounding page:

.. code-block:: rst

    .. simtools-cli-help::
       :application: plot_array_layout
       :no-heading:
"""

import argparse
import importlib
from dataclasses import dataclass

from docutils import nodes
from docutils.parsers.rst import Directive, directives

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


@dataclass(frozen=True)
class CliHelpOptions:
    """Options controlling CLI-help rendering."""

    application_name: str
    prog: str | None
    hidden_groups: set[str]
    include_heading: bool


def _split_csv_option(value: str | None) -> set[str]:
    """Parse a comma-separated directive option into a normalized set."""
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def load_application_parser(application_name: str, prog: str | None = None):
    """Load the explicit parser definition from an application module."""
    if "." in application_name:
        raise ValueError("Application names must not contain a module path")
    module = importlib.import_module(f"simtools.applications.{application_name}")
    application = getattr(module, "APPLICATION", None)
    if application is None:
        raise ValueError(f"Application has no APPLICATION definition: {application_name}")
    parser = application.build_parser()
    if prog is not None:
        parser.prog = prog
    return parser


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


def render_native_cli_docs(parser, hidden_groups: set[str], include_heading: bool = True):
    """Render native RST nodes for one application CLI."""
    rendered_nodes = []
    section = None
    if include_heading:
        section = nodes.section(ids=[nodes.make_id("command-line-arguments")])
        section += nodes.title(text="Command line arguments")
        rendered_nodes.append(section)

    grouped_actions = _group_actions_for_docs(parser, hidden_groups)
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
        if section is None:
            rendered_nodes.append(subgroup)
        else:
            section += subgroup
    return rendered_nodes


class SimtoolsCliHelpDirective(SphinxDirective):
    """Directive that renders filtered CLI help for simtools applications."""

    has_content = False
    option_spec = {
        "application": directives.unchanged_required,
        "prog": directives.unchanged,
        "hide-groups": directives.unchanged,
        "show-groups": directives.unchanged,
        "no-heading": directives.flag,
    }

    def run(self):
        """Render the directive content."""
        options = self._parse_options()
        try:
            parser = load_application_parser(options.application_name, prog=options.prog)
        except (ValueError, ImportError) as error:
            raise self.error(str(error)) from error

        return render_native_cli_docs(
            parser,
            options.hidden_groups,
            include_heading=options.include_heading,
        )

    def _parse_options(self) -> CliHelpOptions:
        """Parse directive options into a structured configuration."""
        hidden_groups = DEFAULT_HIDDEN_GROUPS | _split_csv_option(self.options.get("hide-groups"))
        hidden_groups -= _split_csv_option(self.options.get("show-groups"))
        return CliHelpOptions(
            application_name=self.options["application"],
            prog=self.options.get("prog"),
            hidden_groups=hidden_groups,
            include_heading="no-heading" not in self.options,
        )


def setup(app):
    """Register the Sphinx directive."""
    app.add_directive("simtools-cli-help", SimtoolsCliHelpDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
