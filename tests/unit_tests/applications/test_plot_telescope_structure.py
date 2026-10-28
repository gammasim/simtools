"""Tests for the telescope-structure plotting application."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from simtools.applications import plot_telescope_structure as app


def _run_application(mocker, tmp_test_directory, structure_file):
    """Run the application with a small telescope-model stub."""
    tmp_test_directory = Path(tmp_test_directory)
    if structure_file is not None:
        structure_file = Path(structure_file)
    model_directory = tmp_test_directory / "transient-model"
    mirror_parameter = {"file": True, "value": "mirror_list-1.0.0.ecsv"}
    structure_parameter = {"file": True, "value": "telescope_structure-1.0.0.ecsv"}
    model_reader = SimpleNamespace(export_model_files=Mock())
    telescope_model = SimpleNamespace(
        name="MSTx-FlashCam",
        parameters={
            "mirror_list": mirror_parameter,
            "telescope_structure": structure_parameter,
        },
        config_file_directory=model_directory,
        model_reader=model_reader,
        export_model_files=Mock(),
    )
    args = {
        "site": "North",
        "telescope": "MSTx-FlashCam",
        "model_version": "8.0.0",
        "telescope_structure_file": structure_file,
        "shadowing": False,
    }
    context = SimpleNamespace(
        args=args,
        model_reader=model_reader,
        io_handler=SimpleNamespace(get_output_directory=lambda: tmp_test_directory),
    )
    mocker.patch.object(type(app.APPLICATION), "start", return_value=context)
    mocker.patch.object(app, "TelescopeModel", return_value=telescope_model)
    temporary_directory = mocker.MagicMock()
    temporary_directory.__enter__.return_value = str(model_directory)
    temporary_directory.__exit__.return_value = False
    mocker.patch.object(app, "TemporaryDirectory", return_value=temporary_directory)
    mocker.patch.object(app.structure, "telescope_geometry", return_value=object())
    mocker.patch.object(app.structure, "plot_telescope_structure")
    app.main()
    return telescope_model, model_reader, structure_parameter


def test_main_exports_only_model_files_used_for_plot(mocker, tmp_test_directory):
    telescope_model, model_reader, _ = _run_application(mocker, tmp_test_directory, None)

    telescope_model.export_model_files.assert_not_called()
    model_reader.export_model_files.assert_called_once_with(
        parameters={
            "mirror_list": telescope_model.parameters["mirror_list"],
            "telescope_structure": telescope_model.parameters["telescope_structure"],
        },
        dest=telescope_model.config_file_directory,
    )


def test_main_does_not_export_model_assets_for_custom_structure(mocker, tmp_test_directory):
    structure_file = tmp_test_directory / "custom.ecsv"
    telescope_model, model_reader, _ = _run_application(mocker, tmp_test_directory, structure_file)

    telescope_model.export_model_files.assert_not_called()
    model_reader.export_model_files.assert_called_once_with(
        parameters={
            "mirror_list": telescope_model.parameters["mirror_list"],
        },
        dest=telescope_model.config_file_directory,
    )
    assert telescope_model.parameters["telescope_structure"]["value"] == str(
        Path(structure_file).resolve()
    )
