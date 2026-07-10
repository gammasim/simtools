import sys
from pathlib import Path

from docutils.parsers.rst import directives

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "source" / "_ext"))

from simtools_integration_example import (
    SimtoolsIntegrationExampleDirective,
    get_integration_config_source_url,
    load_integration_example,
)


def test_visibility_defaults_to_showing_command_and_config():
    directive = object.__new__(SimtoolsIntegrationExampleDirective)
    directive.options = {}

    assert directive._get_visibility_options() == (True, True)


def test_visibility_can_show_only_command():
    directive = object.__new__(SimtoolsIntegrationExampleDirective)
    directive.options = {"show-command": directives.flag("")}

    assert directive._get_visibility_options() == (True, False)


def test_visibility_can_show_only_config():
    directive = object.__new__(SimtoolsIntegrationExampleDirective)
    directive.options = {"show-config": directives.flag("")}

    assert directive._get_visibility_options() == (False, True)


def test_existing_docs_example_requests_both_blocks():
    doc_file = Path("src/simtools/applications/simulate_prod.py")
    content = doc_file.read_text(encoding="utf-8")

    assert ":show-command:" in content


def test_load_example_keeps_source_file_name():
    example = load_integration_example("simulate_prod_proton_20_deg_north_check_output.yml")

    assert example.file_name == "simulate_prod_proton_20_deg_north_check_output.yml"


def test_integration_config_source_url_points_to_github_blob():
    assert get_integration_config_source_url(
        "simulate_prod_proton_20_deg_north_check_output.yml"
    ) == (
        "https://github.com/gammasim/simtools/blob/main/tests/integration_tests/config/"
        "simulate_prod_proton_20_deg_north_check_output.yml"
    )
