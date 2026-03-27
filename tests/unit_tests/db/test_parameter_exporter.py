"""Tests for db parameter export orchestration."""

# pylint: disable=redefined-outer-name

import pytest

from simtools.db import parameter_exporter


@pytest.fixture
def db_handler_mock(mocker):
    """Create a basic mocked DB handler with output file helper."""
    db = mocker.Mock()
    db.io_handler.get_output_file.return_value = mocker.MagicMock()
    return db


def test_export_parameter_data_writes_ecsv_for_dict_parameter(mocker, db_handler_mock):
    """Export dict-typed parameter values as ECSV using output_file."""
    db_handler_mock.get_model_parameter.return_value = {
        "fadc_pulse_shape": {
            "type": "dict",
            "value": {"columns": ["time", "amplitude"], "rows": [[1.0, 2.0]]},
        }
    }
    table = mocker.Mock()
    mock_export_single = mocker.patch.object(
        parameter_exporter, "export_single_model_file", return_value=table
    )
    db_handler_mock.io_handler.get_output_file.return_value.with_suffix.return_value = (
        "fadc_pulse_shape.ecsv"
    )

    output_files = parameter_exporter.export_parameter_data(
        db=db_handler_mock,
        parameter="fadc_pulse_shape",
        site="North",
        array_element_name="LSTN-01",
        parameter_version="2.0.0",
        model_version=None,
        output_file="fadc_pulse_shape.json",
        export_model_file=True,
        export_model_file_as_table=False,
    )

    mock_export_single.assert_called_once_with(
        db=db_handler_mock,
        parameter="fadc_pulse_shape",
        site="North",
        array_element_name="LSTN-01",
        parameter_version="2.0.0",
        model_version=None,
        export_file_as_table=True,
    )
    table.write.assert_called_once_with(
        "fadc_pulse_shape.ecsv", format="ascii.ecsv", overwrite=True
    )
    assert output_files == ["fadc_pulse_shape.ecsv"]


def test_export_parameter_data_requires_output_file_for_dict_parameter(db_handler_mock):
    """Require output_file for dict-typed export."""
    db_handler_mock.get_model_parameter.return_value = {
        "fadc_pulse_shape": {
            "type": "dict",
            "value": {"columns": ["time", "amplitude"], "rows": [[1.0, 2.0]]},
        }
    }

    with pytest.raises(ValueError, match="--output_file"):
        parameter_exporter.export_parameter_data(
            db=db_handler_mock,
            parameter="fadc_pulse_shape",
            site="North",
            array_element_name="LSTN-01",
            parameter_version="2.0.0",
            model_version=None,
            output_file=None,
            export_model_file=True,
            export_model_file_as_table=False,
        )


def test_export_parameter_data_requires_export_model_file_for_table_export(db_handler_mock):
    """Reject export_model_file_as_table without export_model_file."""
    with pytest.raises(ValueError, match="Use --export_model_file together"):
        parameter_exporter.export_parameter_data(
            db=db_handler_mock,
            parameter="mirror_reflectivity",
            site="North",
            array_element_name="LSTN-01",
            parameter_version=None,
            model_version="6.0.2",
            output_file=None,
            export_model_file=False,
            export_model_file_as_table=True,
        )


def test_export_parameter_data_rejects_output_file_for_file_parameter(db_handler_mock):
    """Reject output_file for file-backed parameter export."""
    db_handler_mock.get_model_parameter.return_value = {
        "mirror_reflectivity": {"type": "file", "value": "ref_LST1_2022_04_01.dat"}
    }

    with pytest.raises(ValueError, match="Do not use --output_file"):
        parameter_exporter.export_parameter_data(
            db=db_handler_mock,
            parameter="mirror_reflectivity",
            site="North",
            array_element_name="LSTN-01",
            parameter_version=None,
            model_version="6.0.2",
            output_file="mirror_reflectivity.dat",
            export_model_file=True,
            export_model_file_as_table=False,
        )


def test_export_parameter_data_returns_file_and_table_outputs(mocker, db_handler_mock):
    """Return both original file and ECSV file for file-backed table export."""
    db_handler_mock.get_model_parameter.return_value = {
        "mirror_reflectivity": {"type": "file", "value": "ref_LST1_2022_04_01.dat"}
    }
    table = mocker.Mock()
    mock_export_single = mocker.patch.object(
        parameter_exporter, "export_single_model_file", return_value=table
    )
    file_path = mocker.MagicMock()
    file_path.suffix = ".dat"
    file_path.with_suffix.return_value = "ref_LST1_2022_04_01.ecsv"
    db_handler_mock.io_handler.get_output_file.return_value = file_path

    output_files = parameter_exporter.export_parameter_data(
        db=db_handler_mock,
        parameter="mirror_reflectivity",
        site="North",
        array_element_name="LSTN-01",
        parameter_version=None,
        model_version="6.0.2",
        output_file=None,
        export_model_file=True,
        export_model_file_as_table=True,
    )

    mock_export_single.assert_called_once_with(
        db=db_handler_mock,
        parameter="mirror_reflectivity",
        site="North",
        array_element_name="LSTN-01",
        parameter_version=None,
        model_version="6.0.2",
        export_file_as_table=True,
    )
    table.write.assert_called_once_with(
        "ref_LST1_2022_04_01.ecsv", format="ascii.ecsv", overwrite=True
    )
    assert output_files == [file_path, "ref_LST1_2022_04_01.ecsv"]
