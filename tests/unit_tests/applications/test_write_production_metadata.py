"""Tests for the production metadata application."""

from simtools.applications import write_production_metadata


def test_main_delegates_to_production_metadata_workflow(mocker):
    app_context = mocker.MagicMock()
    app_context.args = {"production_path": "production"}
    mock_application = mocker.patch("simtools.applications.write_production_metadata.APPLICATION")
    mock_application.start.return_value = app_context
    mock_write = mocker.patch(
        "simtools.applications.write_production_metadata.write_production_metadata"
    )

    write_production_metadata.main()

    mock_write.assert_called_once_with(app_context.args)
