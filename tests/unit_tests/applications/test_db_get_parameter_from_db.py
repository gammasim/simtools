"""Tests for source-neutral model parameter retrieval."""

import json
from pathlib import Path
from types import SimpleNamespace

from simtools.applications import db_get_parameter_from_db


def test_main_reads_parameter_from_configured_model_reader(tmp_test_directory, mocker):
    """The application uses the configured reader for JSON parameter output."""
    output_file = Path(tmp_test_directory) / "parameter.json"
    parameter_data = {
        "parameter": "array_element_position_ground",
        "parameter_version": "2.0.0",
        "value": [1.0, 2.0, 3.0],
    }
    model_reader = mocker.Mock()
    model_reader.get_model_parameter.return_value = {
        "array_element_position_ground": parameter_data
    }
    app_context = SimpleNamespace(
        args={
            "parameter": "array_element_position_ground",
            "site": "North",
            "telescope": "MSTN-09",
            "parameter_version": "2.0.0",
            "model_version": None,
            "output_file": "parameter.json",
            "export_model_file": False,
            "export_model_file_as_table": False,
        },
        io_handler=SimpleNamespace(get_output_file=lambda _: output_file),
        model_reader=model_reader,
    )
    mocker.patch(
        "simtools.application.definition.ApplicationDefinition.start",
        return_value=app_context,
    )

    db_get_parameter_from_db.main()

    assert json.loads(output_file.read_text(encoding="utf-8")) == parameter_data
    model_reader.get_model_parameter.assert_called_once_with(
        parameter="array_element_position_ground",
        site="North",
        array_element_name="MSTN-09",
        parameter_version="2.0.0",
        model_version=None,
    )


def test_main_exports_through_configured_model_reader(tmp_test_directory, mocker):
    """The application routes file exports through filesystem or Git readers."""
    exported_file = Path(tmp_test_directory) / "mirror.dat"
    model_reader = mocker.Mock()
    model_reader.export_parameter_data.return_value = [exported_file]
    app_context = SimpleNamespace(
        args={
            "parameter": "mirror_reflectivity",
            "site": "North",
            "telescope": "LSTN-01",
            "parameter_version": None,
            "model_version": "6.0.2",
            "output_file": "mirror.dat",
            "export_model_file": True,
            "export_model_file_as_table": False,
        },
        io_handler=SimpleNamespace(get_output_directory=lambda: Path(tmp_test_directory)),
        model_reader=model_reader,
        logger=SimpleNamespace(info=mocker.Mock()),
    )
    mocker.patch(
        "simtools.application.definition.ApplicationDefinition.start",
        return_value=app_context,
    )

    db_get_parameter_from_db.main()

    model_reader.export_parameter_data.assert_called_once_with(
        parameter="mirror_reflectivity",
        site="North",
        array_element_name="LSTN-01",
        parameter_version=None,
        model_version="6.0.2",
        output_file="mirror.dat",
        export_model_file=True,
        export_model_file_as_table=False,
        dest=Path(tmp_test_directory),
    )
