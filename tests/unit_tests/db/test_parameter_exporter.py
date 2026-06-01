"""Tests for db parameter export orchestration."""

# pylint: disable=redefined-outer-name

import pytest

from simtools.db import parameter_exporter

pytestmark = pytest.mark.db_unit_test


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
            "value": {
                "columns": ["time", "amplitude"],
                "column_units": ["ns", "dimensionless"],
                "rows": [[1.0, 2.0]],
            },
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
        export_model_file=False,
        export_model_file_as_table=True,
    )

    mock_export_single.assert_called_once_with(
        db=db_handler_mock,
        parameter="fadc_pulse_shape",
        site="North",
        array_element_name="LSTN-01",
        parameter_version="2.0.0",
        model_version=None,
        export_file_as_table=True,
        parameters=db_handler_mock.get_model_parameter.return_value,
        par_info=db_handler_mock.get_model_parameter.return_value["fadc_pulse_shape"],
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
            "value": {
                "columns": ["time", "amplitude"],
                "column_units": ["ns", "dimensionless"],
                "rows": [[1.0, 2.0]],
            },
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
            export_model_file=False,
            export_model_file_as_table=True,
        )


def test_export_parameter_data_exports_original_file_only(mocker, db_handler_mock):
    """Export only the original model file when export_model_file is set."""
    db_handler_mock.get_model_parameter.return_value = {
        "mirror_reflectivity": {"type": "file", "value": "ref_LST1_2022_04_01.dat"}
    }
    mock_export_single = mocker.patch.object(
        parameter_exporter, "export_single_model_file", return_value=None
    )
    source_file = mocker.MagicMock()
    target_file = mocker.MagicMock()
    db_handler_mock.io_handler.get_output_file.side_effect = [source_file, target_file]

    output_files = parameter_exporter.export_parameter_data(
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

    mock_export_single.assert_called_once_with(
        db=db_handler_mock,
        parameter="mirror_reflectivity",
        site="North",
        array_element_name="LSTN-01",
        parameter_version=None,
        model_version="6.0.2",
        export_file_as_table=False,
        parameters=db_handler_mock.get_model_parameter.return_value,
        par_info=db_handler_mock.get_model_parameter.return_value["mirror_reflectivity"],
    )
    source_file.rename.assert_called_once_with(target_file)
    assert output_files == [target_file]


def test_export_parameter_data_allows_output_file_for_file_parameter(mocker, db_handler_mock):
    """Allow overriding output file name for file-backed table export."""
    db_handler_mock.get_model_parameter.return_value = {
        "mirror_reflectivity": {"type": "file", "value": "ref_LST1_2022_04_01.dat"}
    }
    table = mocker.Mock()
    mocker.patch.object(parameter_exporter, "export_single_model_file", return_value=table)
    source_file = mocker.MagicMock()
    source_file.exists.return_value = True
    target_file = mocker.MagicMock()
    ecsv_output = mocker.MagicMock()
    target_file.with_suffix.return_value = ecsv_output

    def _mock_get_output_file(file_name):
        if file_name == "ref_LST1_2022_04_01.dat":
            return source_file
        return target_file

    db_handler_mock.io_handler.get_output_file.side_effect = _mock_get_output_file

    output_files = parameter_exporter.export_parameter_data(
        db=db_handler_mock,
        parameter="mirror_reflectivity",
        site="North",
        array_element_name="LSTN-01",
        parameter_version=None,
        model_version="6.0.2",
        output_file="mirror_reflectivity.dat",
        export_model_file=False,
        export_model_file_as_table=True,
    )

    table.write.assert_called_once_with(ecsv_output, format="ascii.ecsv", overwrite=True)
    source_file.unlink.assert_called_once()
    assert output_files == [ecsv_output]


def test_export_parameter_data_returns_only_table_output(mocker, db_handler_mock):
    """Return only ECSV output for file-backed table export."""
    db_handler_mock.get_model_parameter.return_value = {
        "mirror_reflectivity": {"type": "file", "value": "ref_LST1_2022_04_01.dat"}
    }
    table = mocker.Mock()
    mock_export_single = mocker.patch.object(
        parameter_exporter, "export_single_model_file", return_value=table
    )
    file_path = mocker.MagicMock()
    file_path.exists.return_value = False
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
        export_model_file=False,
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
        parameters=db_handler_mock.get_model_parameter.return_value,
        par_info=db_handler_mock.get_model_parameter.return_value["mirror_reflectivity"],
    )
    table.write.assert_called_once_with(
        "ref_LST1_2022_04_01.ecsv", format="ascii.ecsv", overwrite=True
    )
    assert output_files == ["ref_LST1_2022_04_01.ecsv"]


def test_export_parameter_data_returns_file_and_table_outputs(mocker, db_handler_mock):
    """Return both original and ECSV outputs when both export flags are set."""
    db_handler_mock.get_model_parameter.return_value = {
        "mirror_reflectivity": {"type": "file", "value": "ref_LST1_2022_04_01.dat"}
    }
    table = mocker.Mock()
    mocker.patch.object(parameter_exporter, "export_single_model_file", return_value=table)

    source_file = mocker.MagicMock()
    model_output_file = mocker.MagicMock()
    table_output_file = mocker.MagicMock()
    model_output_file.with_suffix.return_value = table_output_file
    db_handler_mock.io_handler.get_output_file.side_effect = [source_file, model_output_file]

    output_files = parameter_exporter.export_parameter_data(
        db=db_handler_mock,
        parameter="mirror_reflectivity",
        site="North",
        array_element_name="LSTN-01",
        parameter_version=None,
        model_version="6.0.2",
        output_file="mirror_reflectivity.dat",
        export_model_file=True,
        export_model_file_as_table=True,
    )

    source_file.rename.assert_called_once_with(model_output_file)
    table.write.assert_called_once_with(table_output_file, format="ascii.ecsv", overwrite=True)
    assert output_files == [model_output_file, table_output_file]


def test_export_model_files_requires_destination(db_handler_mock):
    """Require destination path for exporting files from DB."""
    with pytest.raises(ValueError, match="Destination path is required"):
        parameter_exporter.export_model_files(
            db=db_handler_mock,
            parameters={"mirror_reflectivity": {"file": True, "value": "test.dat"}},
            dest=None,
        )


def test_normalize_file_names_returns_empty_list_for_no_inputs():
    """Return empty file list when neither file_names nor parameters are provided."""
    assert parameter_exporter._normalize_file_names() == []


def test_export_parameter_data_file_parameter_ecsv_suffix_skips_table_conversion(
    mocker, db_handler_mock
):
    """Write ECSV output even when source payload filename already has .ecsv suffix."""
    db_handler_mock.get_model_parameter.return_value = {
        "mirror_reflectivity": {"type": "file", "value": "mirror_reflectivity.ecsv"}
    }
    table = mocker.Mock()
    mocker.patch.object(parameter_exporter, "export_single_model_file", return_value=table)

    output_file = mocker.MagicMock()
    output_file.with_suffix.return_value = output_file
    output_file.exists.return_value = False
    db_handler_mock.io_handler.get_output_file.return_value = output_file

    output_files = parameter_exporter.export_parameter_data(
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

    assert output_files == [output_file]
    table.write.assert_called_once_with(output_file, format="ascii.ecsv", overwrite=True)


def test_export_parameter_data_no_export_returns_empty_list(db_handler_mock):
    """No export flags returns empty list."""
    result = parameter_exporter.export_parameter_data(
        db=db_handler_mock,
        parameter="test_param",
        site="North",
        array_element_name="LSTN-01",
        export_model_file=False,
        export_model_file_as_table=False,
    )
    assert result == []
