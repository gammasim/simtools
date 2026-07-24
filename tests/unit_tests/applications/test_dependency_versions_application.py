"""Tests for the dependency-versions application."""

import json
import subprocess
import sys
from pathlib import Path

from simtools.applications import dependency_versions


def test_application_definition_is_configured():
    """Test the dependency-version exporter uses the standard application definition."""
    assert dependency_versions.APPLICATION.setup_io_handler is False
    assert dependency_versions.APPLICATION.resolve_sim_software_executables is False


def test_main_writes_exported_configuration(mocker, capsys):
    """Test the application writes the library export to standard output."""
    mocker.patch(
        "simtools.applications.dependency_versions.APPLICATION",
        start=mocker.Mock(
            return_value=mocker.Mock(args={"pyproject": None, "format": "summary", "extras": []})
        ),
    )
    mocker.patch(
        "simtools.applications.dependency_versions._export_dependency_configuration",
        return_value="summary\n",
    )

    dependency_versions.main()

    assert capsys.readouterr().out == "summary\n"


def test_standalone_application_exports_build_summary(simtools_root_path):
    """Test build workflows can execute the application before installing simtools."""
    application = Path(dependency_versions.__file__)
    result = subprocess.run(
        [sys.executable, application, "--format", "summary"],
        cwd=simtools_root_path,
        capture_output=True,
        check=True,
        text=True,
    )

    assert json.loads(result.stdout)["python_version"] == "3.14"
