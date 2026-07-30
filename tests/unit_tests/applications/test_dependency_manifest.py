"""Tests for the dependency-manifest application."""

from simtools.applications import dependency_manifest


def test_application_definition_is_configured():
    """Test the dependency-manifest application avoids runtime setup."""
    assert dependency_manifest.APPLICATION.setup_io_handler is False
    assert dependency_manifest.APPLICATION.resolve_sim_software_executables is False


def test_main_writes_runtime_manifest(mocker):
    """Test the application dispatches the regular manifest writer."""
    output_file = mocker.Mock()
    mocker.patch(
        "simtools.applications.dependency_manifest.APPLICATION",
        start=mocker.Mock(
            return_value=mocker.Mock(
                args={
                    "development": False,
                    "output_file": output_file,
                    "project_file": mocker.Mock(),
                    "build_option_files": [],
                }
            )
        ),
    )
    write_manifest = mocker.patch(
        "simtools.applications.dependency_manifest.write_dependency_manifest"
    )

    dependency_manifest.main()

    write_manifest.assert_called_once_with(output_file)


def test_main_writes_development_manifest(mocker):
    """Test the application dispatches the development manifest writer."""
    output_file = mocker.Mock()
    project_file = mocker.Mock()
    build_option_files = [mocker.Mock()]
    mocker.patch(
        "simtools.applications.dependency_manifest.APPLICATION",
        start=mocker.Mock(
            return_value=mocker.Mock(
                args={
                    "development": True,
                    "output_file": output_file,
                    "project_file": project_file,
                    "build_option_files": build_option_files,
                }
            )
        ),
    )
    write_manifest = mocker.patch(
        "simtools.applications.dependency_manifest.write_development_dependency_manifest"
    )

    dependency_manifest.main()

    write_manifest.assert_called_once_with(output_file, project_file, build_option_files)
