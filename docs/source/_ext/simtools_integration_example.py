"""
Render integration-test configs as documentation examples.

Example
-------
Render both the equivalent command and the tested YAML configuration:

.. code-block:: rst

    .. simtools-integration-example::
       :file: docs_produce_production_summary_run.yml

Render only the command block for an example:

.. code-block:: rst

    .. simtools-integration-example::
       :file: docs_produce_array_element_report_run.yml
       :show-command:
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive, directives

from simtools import constants

try:
    from sphinx.util.docutils import SphinxDirective
except ModuleNotFoundError:  # pragma: no cover - fallback for non-Sphinx imports
    SphinxDirective = Directive


@dataclass(frozen=True)
class IntegrationExample:
    """Structured representation of an integration-config example."""

    file_name: str
    application: str
    configuration: dict
    title: str | None = None
    summary: str | None = None


def get_repo_root() -> Path:
    """Return the repository root from the extension location."""
    return Path(__file__).resolve().parents[3]


def get_integration_config_dir(repo_root: Path | None = None) -> Path:
    """Return the directory containing integration-config YAML files."""
    return (repo_root or get_repo_root()) / "tests" / "integration_tests" / "config"


def _resolve_example_file(file_name: str, config_dir: Path) -> Path:
    file_path = (config_dir / file_name).resolve()
    if not file_path.is_file() or not file_path.is_relative_to(config_dir.resolve()):
        raise ValueError(f"Integration config not found: {file_name}")
    return file_path


def get_integration_config_source_url(file_name: str) -> str:
    """Return the GitHub source URL for an integration-config file."""
    return f"{constants.GITHUB_BLOB_BASE}tests/integration_tests/config/{file_name}"


def load_integration_example(file_name: str, config_dir: Path | None = None) -> IntegrationExample:
    """Load a single application example from an integration config."""
    resolved_config_dir = (config_dir or get_integration_config_dir()).resolve()
    file_path = _resolve_example_file(file_name, resolved_config_dir)

    with file_path.open(encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle) or {}

    applications = data.get("applications")
    if not isinstance(applications, list) or not applications:
        raise ValueError(f"Integration config has no applications entry: {file_name}")

    application_entry = applications[0]
    if not isinstance(application_entry, dict):
        raise ValueError(f"Invalid application entry in integration config: {file_name}")

    application = application_entry.get("application")
    configuration = application_entry.get("configuration")
    docs_metadata = application_entry.get("docs") or {}

    if not isinstance(application, str) or not application:
        raise ValueError(f"Missing application name in integration config: {file_name}")
    if not isinstance(configuration, dict):
        raise ValueError(f"Missing configuration block in integration config: {file_name}")
    if not isinstance(docs_metadata, dict):
        raise ValueError(f"Invalid docs metadata in integration config: {file_name}")

    return IntegrationExample(
        file_name=file_name,
        application=application,
        configuration=configuration,
        title=docs_metadata.get("title"),
        summary=docs_metadata.get("summary"),
    )


def render_configuration_yaml(example: IntegrationExample) -> str:
    """Render the inner configuration block as YAML."""
    return yaml.safe_dump(example.configuration, sort_keys=False).rstrip()


def render_command(example: IntegrationExample) -> str:
    """Render the executable command for the integration configuration."""
    return f"{example.application} --config tests/integration_tests/config/{example.file_name}"


class SimtoolsIntegrationExampleDirective(SphinxDirective):
    """Directive that renders a tested integration config as a docs example."""

    has_content = False
    option_spec = {
        "file": directives.unchanged_required,
        "show-command": directives.flag,
        "show-config": directives.flag,
    }

    def run(self):
        """Render the directive content."""
        try:
            example = load_integration_example(self.options["file"])
        except ValueError as error:
            raise self.error(str(error)) from error

        rendered_nodes = []
        content_parent = self._render_title(example, rendered_nodes)
        self._append_summary(example, rendered_nodes, content_parent)
        self._append_application_info(example, rendered_nodes, content_parent)
        self._append_visibility_blocks(example, rendered_nodes, content_parent)
        return rendered_nodes

    def _render_title(self, example, rendered_nodes):
        """Render the example title as a nested section heading when present."""
        if not example.title:
            return None

        section = nodes.section(ids=[nodes.make_id(example.title)])
        section += nodes.title(text=example.title)
        rendered_nodes.append(section)
        return section

    @staticmethod
    def _append_node(rendered_nodes, content_parent, node):
        """Append a node either to the current titled section or to the top level."""
        if content_parent is not None:
            content_parent += node
        else:
            rendered_nodes.append(node)

    def _append_summary(self, example, rendered_nodes, content_parent):
        """Append the optional summary paragraph."""
        if example.summary:
            self._append_node(rendered_nodes, content_parent, nodes.paragraph(text=example.summary))

    def _append_application_info(self, example, rendered_nodes, content_parent):
        """Append application and integration-test metadata."""
        application_paragraph = nodes.paragraph()
        application_paragraph += nodes.strong(text="Application: ")
        application_paragraph += nodes.literal(text=example.application)
        self._append_node(rendered_nodes, content_parent, application_paragraph)

        integration_test_paragraph = nodes.paragraph()
        integration_test_paragraph += nodes.strong(text="Integration test: ")
        integration_test_paragraph += nodes.reference(
            text=example.file_name,
            refuri=get_integration_config_source_url(example.file_name),
        )
        self._append_node(rendered_nodes, content_parent, integration_test_paragraph)

    def _append_visibility_blocks(self, example, rendered_nodes, content_parent):
        """Append the requested command/configuration blocks."""
        show_command, show_config = self._get_visibility_options()
        if show_command:
            self._append_block(
                rendered_nodes,
                content_parent,
                "Command",
                render_command(example),
                "console",
            )
        if show_config:
            self._append_block(
                rendered_nodes,
                content_parent,
                "Configuration",
                render_configuration_yaml(example),
                "yaml",
            )

    def _append_block(self, rendered_nodes, content_parent, title, text, language):
        """Append a titled literal block."""
        self._append_node(rendered_nodes, content_parent, nodes.rubric(text=title))
        literal_block = nodes.literal_block(text=text)
        literal_block["language"] = language
        self._append_node(rendered_nodes, content_parent, literal_block)

    def _get_visibility_options(self):
        """Resolve which content blocks to render.

        If neither option is provided, render both blocks. If one or both options
        are provided, render only the explicitly requested blocks.
        """
        show_command = "show-command" in self.options
        show_config = "show-config" in self.options

        if not show_command and not show_config:
            return True, True

        return show_command, show_config


def setup(app):
    """Register the Sphinx directive."""
    app.add_directive("simtools-integration-example", SimtoolsIntegrationExampleDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
