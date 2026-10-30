#!/usr/bin/python3

import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.table import Column, Table
from astropy.utils.diff import report_diff_values

from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH, SCHEMA_PATH
from simtools.data_model import schema, validate_data
from simtools.io import ascii_handler

logger = logging.getLogger()

mirror_2f_schema_file = SCHEMA_PATH / "input/MST_mirror_2f_measurements.schema.yml"


def _write_mirror_file(tmp_test_directory):
    mirror_table = Table(
        {
            "mirror_panel_id": ["1"],
            "mirror_curvature_radius": [100.0] * u.cm,
            "psf": [1.0] * u.cm,
            "psf_opt": [0.8] * u.cm,
        }
    )
    mirror_file_path = Path(tmp_test_directory) / "mirror_measurement.ecsv"
    mirror_table.write(mirror_file_path, format="ascii.ecsv", overwrite=True)
    return mirror_file_path


def _write_num_gains_file(tmp_test_directory, file_name="num_gains-1.0.0.json"):
    parameter_file = Path(tmp_test_directory) / file_name
    ascii_handler.write_data_to_file(
        {
            "parameter": "num_gains",
            "parameter_version": "1.0.0",
            "instrument": "LSTN-01",
            "site": "North",
            "value": 2,
            "unit": None,
        },
        parameter_file,
    )
    return parameter_file


@pytest.fixture
def reference_columns():
    """Return a test reference data column definition."""
    return [
        {
            "name": "wavelength",
            "description": "wavelength",
            "required": True,
            "unit": "nm",
            "type": "double",
            "required_range": {"unit": "nm", "min": 300, "max": 700},
            "input_processing": ["remove_duplicates", "sort"],
        },
        {
            "name": "qe",
            "description": "average quantum or photon detection efficiency",
            "required": True,
            "unit": "dimensionless",
            "type": "double",
            "allowed_range": {"unit": "unitless", "min": 0.0, "max": 1.0},
        },
        {
            "name": "position_x",
            "description": "x position",
            "required": False,
            "unit": "m",
            "type": "double",
            "allowed_range": {"unit": "m", "min": 0.0, "max": 1.0},
            "input_processing": ["allow_nan"],
        },
        {
            "name": "position_y",
            "description": "y position",
            "required": False,
            "unit": "m",
            "type": "double",
            "allowed_range": {"unit": "m", "min": 0.0, "max": 1.0},
            "input_processing": ["allow_nan"],
        },
        {
            "name": "abc",
            "description": "not required",
            "required": False,
            "unit": "kg",
            "type": "double",
            "allowed_range": {"unit": "kg", "min": 0.0, "max": 100.0},
        },
        {
            "name": "no_units",
            "description": "not required",
            "required": False,
            "type": "double",
        },
    ]


@pytest.fixture
def reference_columns_name():
    """Test reference data column definition with columns named col0, col1, col3."""
    return [
        {
            "name": "col0",
        },
        {
            "name": "col1",
        },
        {
            "name": "col2",
        },
    ]


def test_validate_and_transform(caplog, mocker, tmp_test_directory):
    data_validator = validate_data.DataValidator()
    # no input file defined
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TypeError):
            data_validator.validate_and_transform()
    assert "No data or data table to validate" in caplog.text

    data_validator.data_file_name = _write_mirror_file(tmp_test_directory)
    data_validator.schema_file_name = mirror_2f_schema_file
    with caplog.at_level(logging.INFO):
        _table = data_validator.validate_and_transform()
        assert isinstance(_table, Table)
    assert "Validating tabled data from:" in caplog.text

    data_validator.data_file_name = _write_num_gains_file(tmp_test_directory)
    data_validator.schema_file_name = MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"
    mock_prepare_model_parameter = mocker.patch(
        "simtools.data_model.validate_data.DataValidator._prepare_model_parameter"
    )
    with caplog.at_level(logging.INFO):
        _dict = data_validator.validate_and_transform(True)
        assert isinstance(_dict, dict)
    assert "Validating data from:" in caplog.text
    mock_prepare_model_parameter.assert_called_once()


def test_validate_data_file(caplog, tmp_test_directory):
    data_validator = validate_data.DataValidator()
    # no input file defined, should pass
    data_validator.validate_data_file()

    data_validator.data_file_name = _write_mirror_file(tmp_test_directory)
    with caplog.at_level(logging.INFO):
        data_validator.validate_data_file()
    assert "Validating tabled data from:" in caplog.text

    data_validator.data_file_name = _write_num_gains_file(tmp_test_directory)
    with caplog.at_level(logging.INFO):
        data_validator.validate_data_file()
    assert "Validating data from:" in caplog.text


def test_validate_parameter_and_file_name(caplog, tmp_test_directory):
    num_gain_file = _write_num_gains_file(tmp_test_directory)
    parameter = "num_gains"
    parameter_version = "1.0.0"

    data_validator = validate_data.DataValidator()
    data_validator.data_file_name = num_gain_file
    data_validator.schema_file_name = MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"
    data_validator.validate_and_transform()

    data_validator.data_dict["parameter"] = "incorrect_name"
    with pytest.raises(
        ValueError,
        match=r"Mismatch: parameter 'incorrect_name' vs. file 'num_gains-1.0.0'.",
    ):
        data_validator.validate_parameter_and_file_name()

    data_validator.data_file_name = num_gain_file
    data_validator.data_dict = {
        "parameter": parameter,
        "parameter_version": parameter_version,
    }
    data_validator.validate_parameter_and_file_name()

    data_validator.data_file_name = Path(tmp_test_directory) / "incorrect_name-1.0.0.json"
    data_validator.data_dict = {
        "parameter": parameter,
        "parameter_version": parameter_version,
    }
    with pytest.raises(
        ValueError, match=r"Mismatch: parameter 'num_gains' vs. file 'incorrect_name-1.0.0'"
    ):
        data_validator.validate_parameter_and_file_name()

    data_validator.data_file_name = Path(tmp_test_directory) / "num_gains-2.0.0.json"
    data_validator.data_dict = {
        "parameter": parameter,
        "parameter_version": parameter_version,
    }
    with pytest.raises(ValueError, match=r"Mismatch: version '1.0.0' vs. file 'num_gains-2.0.0'"):
        data_validator.validate_parameter_and_file_name()

    data_validator.data_file_name = Path(tmp_test_directory) / "num_gains.json"
    data_validator.data_dict = {"parameter": parameter, "parameter_version": None}
    with caplog.at_level(logging.WARNING):
        data_validator.validate_parameter_and_file_name()
    assert "File 'num_gains' has no parameter version defined." in caplog.text


def test_validate_invalid_version():
    data_validator = validate_data.DataValidator(
        schema_file=MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"
    )
    data_validator.data_dict = {
        "parameter": "num_gains",
        "parameter_version": "invalid",
        "value": 2,
        "unit": "",
    }
    with pytest.raises(ValueError, match="Invalid version string 'invalid'"):
        data_validator.validate_and_transform()


def test_validate_data_columns(tmp_test_directory, caplog):
    data_validator = validate_data.DataValidator()
    with pytest.raises(TypeError):
        data_validator._validate_data_table()

    data_validator_1 = validate_data.DataValidator(
        schema_file=None,
        data_file=_write_mirror_file(tmp_test_directory),
    )
    data_validator_1.validate_data_file()
    with pytest.raises(TypeError):
        data_validator_1._validate_data_table()

    data_validator_3 = validate_data.DataValidator(
        schema_file=mirror_2f_schema_file,
        data_file=_write_mirror_file(tmp_test_directory),
    )
    data_validator_3.validate_data_file()
    data_validator_3._validate_data_table()
    # test change of units
    _value_in_org = data_validator_3.data_table["psf"].value[0]
    data_validator_3._validate_data_columns()
    for col in data_validator_3._data_description:
        col["unit"] = "m" if col["unit"] == "cm" else col["unit"]
    data_validator_3._validate_data_columns()
    _value_in_m = data_validator_3.data_table["psf"].value[0]
    assert data_validator_3.data_table["psf"].unit == "m"
    assert _value_in_org == pytest.approx(_value_in_m * 100.0)

    _incomplete_schema = {"data": []}
    with open(tmp_test_directory / "incomplete_data_schema.schema.yml", "w") as _file:
        yaml.dump(_incomplete_schema, _file)

    data_validator_3.schema_file_name = tmp_test_directory / "incomplete_data_schema.schema.yml"
    with caplog.at_level(logging.ERROR):
        with pytest.raises(IndexError):
            data_validator_3._validate_data_table()
    assert "Error reading validation schema from" in caplog.text


def test_sort_data(reference_columns, caplog):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    table_1 = Table()
    table_1["wavelength"] = Column([300.0, 350.0, 315.0], unit="nm", dtype="float32")
    table_1["qe"] = Column([0.1, 0.5, 0.2], dtype="float32")

    table_sorted = Table()
    table_sorted["wavelength"] = Column([300.0, 315.0, 350.0], unit="nm", dtype="float32")
    table_sorted["qe"] = Column([0.1, 0.2, 0.5], dtype="float32")

    table_reverse_sorted = Table()
    table_reverse_sorted["wavelength"] = Column([350.0, 315.0, 300.0], unit="nm", dtype="float32")
    table_reverse_sorted["qe"] = Column([0.5, 0.2, 0.1], dtype="float32")

    data_validator.data_table = table_1
    data_validator._sort_data()

    identical_sorted = report_diff_values(
        data_validator.data_table, table_sorted, fileobj=sys.stdout
    )

    data_validator.data_table = None
    with caplog.at_level(logging.ERROR):
        with pytest.raises(AttributeError):
            data_validator._sort_data()
    assert "No data table defined for sorting" in caplog.text

    reverse_sorted_data_columns = reference_columns
    reverse_sorted_data_columns[0]["input_processing"] = ["remove_duplicates", "reversesort"]
    data_validator_reverse = validate_data.DataValidator()
    data_validator_reverse._data_description = reverse_sorted_data_columns

    data_validator_reverse.data_table = table_1
    data_validator_reverse._sort_data()

    identical_reverse_sorted = report_diff_values(
        data_validator_reverse.data_table, table_reverse_sorted, fileobj=sys.stdout
    )

    assert identical_sorted
    assert identical_reverse_sorted

    data_validator_reverse.data_table = None
    with caplog.at_level(logging.ERROR):
        with pytest.raises(AttributeError):
            data_validator_reverse._sort_data()
    assert "No data table defined for sorting" in caplog.text


def test_check_data_for_duplicates(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    table_unique = Table()
    table_unique["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_unique["qe"] = Column([0.1, 0.5], dtype="float32")

    table_1 = Table()
    table_1["wavelength"] = Column([300.0, 350.0, 350.0], unit="nm", dtype="float32")
    table_1["qe"] = Column([0.1, 0.5, 0.5], dtype="float32")

    data_validator.data_table = table_1
    data_validator._check_data_for_duplicates()

    identical_1 = report_diff_values(data_validator.data_table, table_unique, fileobj=sys.stdout)

    assert identical_1

    table_2 = Table()
    table_2["wavelength"] = Column([300.0, 315.0, 350.0], unit="nm", dtype="float32")
    table_2["qe"] = Column([0.1, 0.5, 0.5], dtype="float32")

    data_validator.data_table = table_2
    data_validator._check_data_for_duplicates()

    not_identical = report_diff_values(data_validator.data_table, table_2, fileobj=sys.stdout)

    assert not_identical

    table_3 = Table()
    table_3["wavelength"] = Column([300.0, 350.0, 350.0], unit="nm", dtype="float32")
    table_3["qe"] = Column([0.1, 0.5, 0.8], dtype="float32")

    data_validator.data_table = table_3
    with pytest.raises(ValueError, match=r"^Failed removal of duplication for column"):
        data_validator._check_data_for_duplicates()


def test_check_range(reference_columns, caplog):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    col_1 = Column(name="qe", data=[0.1, 0.5], dtype="float32")
    data_validator._check_range(col_1.name, col_1.min(), col_1.max(), "allowed_range")
    col_w = Column(name="wavelength", data=[250.0, 750.0], dtype="float32")
    data_validator._check_range(col_w.name, col_w.min(), col_w.max(), "required_range")

    col_2 = Column(name="key_error", data=[0.1, 0.5], dtype="float32")
    with pytest.raises(IndexError):
        data_validator._check_range(col_2.name, col_2.min(), col_2.max(), "allowed_range")

    col_3 = Column(name="qe", data=[0.1, 5.00], dtype="float32")
    with pytest.raises(ValueError, match=r"Value for column 'qe' out of range"):
        data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "allowed_range")
    col_3 = Column(name="qe", data=[-0.1, 0.5], dtype="float32")
    with pytest.raises(ValueError, match=r"^Value for column 'qe' out of range"):
        data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "allowed_range")

    # a range that is not defined in the schema is not checked
    data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "failed_range")


def test_check_and_convert_units_with_errors(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    table_2 = Table()
    table_2["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_2["wrong_column"] = Column([0.1, 0.5], dtype="float32")

    with pytest.raises(IndexError):
        data_validator._check_and_convert_units(
            table_2["wrong_column"], unit=None, col_name="wrong_column"
        )

    table_3 = Table()
    table_3["wavelength"] = Column([300.0, 350.0], unit="kg", dtype="float32")

    for col_name in table_3.colnames:
        with pytest.raises(u.core.UnitConversionError):
            data_validator._check_and_convert_units(table_3[col_name], unit=None, col_name=col_name)


def test_check_and_convert_units_integer_arrays(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    data_validator._data_description[0]["type"] = ["int", "int"]
    assert data_validator._check_and_convert_units(
        [300, 350], unit=["nm", "nm"], col_name="wavelength"
    ) == ([300, 350], "nm")
    data_validator._data_description[0]["type"] = ["int", "int"]
    assert data_validator._check_and_convert_units(
        [300, 350], unit=["nm", None], col_name="wavelength"
    ) == ([300, 350], "nm")


def test_get_data_description(reference_columns, reference_columns_name):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    assert isinstance(data_validator._get_data_description("wavelength"), dict)

    _entry = data_validator._get_data_description("wavelength")
    assert _entry["name"] == "wavelength"

    with pytest.raises(IndexError):
        data_validator._get_data_description("wrong_column")

    assert data_validator._get_data_description("wavelength", status_test=True)
    assert not data_validator._get_data_description("wrong_column", status_test=True)

    data_validator._data_description = reference_columns_name

    assert isinstance(data_validator._get_data_description("col1"), dict)

    with pytest.raises(IndexError):
        data_validator._get_data_description("col3")

    assert data_validator._get_data_description("col1", status_test=True)

    assert data_validator._get_data_description("col1") == {"name": "col1"}

    with pytest.raises(IndexError):
        data_validator._get_data_description(100)


def test_check_data_type(reference_columns, caplog):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    with caplog.at_level(logging.DEBUG):
        assert data_validator._check_data_type(np.dtype("double"), "wavelength") is None

    with caplog.at_level(logging.ERROR):
        with pytest.raises(TypeError):
            data_validator._check_data_type(np.dtype("float32"), "wavelength")
        assert (
            "Invalid data type in column 'wavelength'. Expected type 'double', found 'float32'"
            in caplog.text
        )

    # sub types only
    data_validator.check_exact_data_type = False

    # floats
    assert data_validator._check_data_type(np.dtype("float32"), "wavelength") is None
    assert data_validator._check_data_type(np.dtype("int64"), "wavelength") is None

    # ints
    data_validator._data_description[0]["type"] = "int"
    assert data_validator._check_data_type(np.dtype("int64"), "wavelength") is None

    # string types
    data_validator._data_description[0]["type"] = "string"
    assert data_validator._check_data_type(np.dtype("U10"), "wavelength") is None
    data_validator._data_description[0]["type"] = "str"
    assert data_validator._check_data_type(np.dtype("U10"), "wavelength") is None
    data_validator._data_description[0]["type"] = "file"
    assert data_validator._check_data_type(np.dtype("U10"), "wavelength") is None
    assert data_validator._check_data_type(np.array(None).dtype, "wavelength") is None

    # bool types
    data_validator._data_description[0]["type"] = "bool"
    assert data_validator._check_data_type(np.dtype("bool"), "wavelength") is None
    data_validator._data_description[0]["type"] = "boolean"
    assert data_validator._check_data_type(np.dtype("bool"), "wavelength") is None


def test_check_for_not_a_number(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    assert not (
        data_validator._check_for_not_a_number(
            Column([300.0, 350.0, 315.0], dtype="float32", name="wavelength"), "wavelength"
        )
    )

    assert data_validator._check_for_not_a_number("string", "wavelength")

    error_message = "NaN or Inf values found in data"
    # wavelength does not allow for nan
    with pytest.raises(ValueError, match=rf"{error_message}"):
        data_validator._check_for_not_a_number([np.nan, 350.0, 315.0], "wavelength")
    with pytest.raises(ValueError, match=rf"{error_message}"):
        data_validator._check_for_not_a_number([np.nan, 350.0, np.inf], "wavelength")
    with pytest.raises(ValueError, match=rf"{error_message}"):
        data_validator._check_for_not_a_number([300.0, 350.0, np.inf], "wavelength")

    # position_x allows for nan
    assert not (data_validator._check_for_not_a_number([300.0, 350.0, 315.0], "position_x"))
    assert data_validator._check_for_not_a_number([np.nan, 350.0, 315.0], "position_x")
    assert data_validator._check_for_not_a_number([333.0, np.inf, 315.0], "position_x")

    assert not data_validator._check_for_not_a_number(333.0, "wavelength")
    with pytest.raises(ValueError, match=rf"{error_message}"):
        data_validator._check_for_not_a_number(np.inf, "wavelength")
    with pytest.raises(ValueError, match=rf"{error_message}"):
        data_validator._check_for_not_a_number(np.nan, "wavelength")


def test_read_validation_schema(tmp_test_directory):
    data_validator = validate_data.DataValidator()

    # no file given
    with pytest.raises(TypeError):
        data_validator._read_validation_schema(schema_file=None)

    # file given
    _schema = data_validator._read_validation_schema(schema_file=mirror_2f_schema_file)
    assert isinstance(_schema, list)

    # file does not exist
    with pytest.raises(FileNotFoundError):
        data_validator._read_validation_schema(schema_file="this_file_does_not_exist.yml")

    # read a 'wrong' schema file with no 'data' key included
    with open(tmp_test_directory / "wrong_schema.yml", "w") as _file:
        yaml.dump({"wrong_key": []}, _file)
    wrong_schema = "wrong_schema.yml"
    with pytest.raises(KeyError, match=r"Error reading validation schema from .*wrong_schema.yml"):
        data_validator._read_validation_schema(schema_file=tmp_test_directory / wrong_schema)

    with pytest.raises(ValueError, match=r"^Schema version not_a_version not found"):
        data_validator._read_validation_schema(
            schema_file=tmp_test_directory / wrong_schema, schema_version="not_a_version"
        )

    corsika_starting_grammage = "corsika_starting_grammage.schema.yml"
    # test model parameter schema with multiple documents - no version given
    dict_1 = data_validator._read_validation_schema(
        MODEL_PARAMETER_SCHEMA_PATH / corsika_starting_grammage
    )
    assert isinstance(dict_1[0], dict)

    # test model parameter schema with multiple documents - version given
    dict_2 = data_validator._read_validation_schema(
        MODEL_PARAMETER_SCHEMA_PATH / corsika_starting_grammage, schema_version="0.2.0"
    )
    assert dict_2[0]["type"] == "dict"
    # test model parameter schema with multiple documents - version given
    dict_3 = data_validator._read_validation_schema(
        MODEL_PARAMETER_SCHEMA_PATH / corsika_starting_grammage, schema_version="0.1.0"
    )
    assert dict_3[0]["type"] == "float64"


# incomplete test
def test_validate_heterogeneous_list_from_multi_entry_schema(mocker):
    mocker.patch(
        "simtools.data_model.schema.get_model_parameter_schema_file",
        return_value="/mock/schema.yml",
    )
    mocker.patch(
        "simtools.data_model.validate_data.DataValidator._read_validation_schema",
        return_value=[
            {"name": "flasher_shape", "type": "string", "unit": None},
            {"name": "width", "type": "float64", "unit": "ns"},
            {"name": "decay", "type": "float64", "unit": "ns"},
        ],
    )
    mocker.patch(
        "simtools.data_model.validate_data.DataValidator._validate_value_and_unit",
        side_effect=lambda value, unit, index: (value, unit),
    )

    par_dict = {
        "parameter": "flasher_pulse_shape",
        "type": ["string", "float64", "float64"],
        "value": ["Gauss", 0.0, 0.0],
        "unit": [None, "ns", "ns"],
    }

    validated_data = validate_data.DataValidator.validate_model_parameter(par_dict)
    assert validated_data["value"] == ["Gauss", 0.0, 0.0]


def test_validate_heterogeneous_list_length_mismatch(mocker):
    mocker.patch(
        "simtools.data_model.schema.get_model_parameter_schema_file",
        return_value="/mock/schema.yml",
    )
    mocker.patch(
        "simtools.data_model.validate_data.DataValidator._read_validation_schema",
        return_value=[
            {
                "name": "heterogeneous_param",
                "type": ["int64", "float64"],
                "unit": [None, "km"],
            }
        ],
    )

    # Invalid: only 1 value for 2 expected types
    par_dict = {
        "parameter": "heterogeneous_param",
        "value": [10],
        "unit": [None],
    }

    with pytest.raises(TypeError, match=r"Error validating heterogeneous list"):
        validate_data.DataValidator.validate_model_parameter(par_dict)


def test_validate_homogeneous_list_type_mismatch(mocker):
    mocker.patch(
        "simtools.data_model.schema.get_model_parameter_schema_file",
        return_value="/mock/schema.yml",
    )
    mocker.patch(
        "simtools.data_model.validate_data.DataValidator._read_validation_schema",
        return_value=[{"name": "homogeneous_param", "type": "float64", "unit": "km"}],
    )

    # Invalid: mixed types (float and string)
    par_dict = {
        "parameter": "homogeneous_param",
        "value": [1.0, "invalid", 3.0],
        "unit": ["km", "km", "km"],
    }

    with pytest.raises(TypeError, match=r"Error validating dictionary using"):
        validate_data.DataValidator.validate_model_parameter(par_dict)


def test_validate_data_files_directory(tmp_test_directory, caplog):
    # Create test files
    test_file_1 = tmp_test_directory / "num_gains-1.0.0.json"
    test_file_2 = tmp_test_directory / "reference_point_altitude-1.0.0.json"

    test_data_1 = {
        "parameter": "num_gains",
        "parameter_version": "1.0.0",
        "value": [2],
        "unit": [""],
    }
    test_data_2 = {
        "parameter": "reference_point_altitude",
        "parameter_version": "1.0.0",
        "value": 1000.0,
        "unit": "m",
    }

    ascii_handler.write_data_to_file(test_data_1, test_file_1)
    ascii_handler.write_data_to_file(test_data_2, test_file_2)

    with caplog.at_level(logging.INFO):
        validate_data.DataValidator.validate_data_files(file_directory=tmp_test_directory)

    assert "Validated data file" in caplog.text
    assert "num_gains-1.0.0.json" in caplog.text
    assert "reference_point_altitude-1.0.0.json" in caplog.text


def test_validate_data_files_single_file(caplog, tmp_test_directory):
    test_file = _write_num_gains_file(tmp_test_directory)

    with caplog.at_level(logging.INFO):
        validate_data.DataValidator.validate_data_files(file_name=test_file)

    assert "Validated data file" in caplog.text
    assert "num_gains-1.0.0.json" in caplog.text


def test_validate_data_files_no_input():
    result = validate_data.DataValidator.validate_data_files()
    assert result is None


def test__check_site_and_array_element_consistency():
    data_validator = validate_data.DataValidator()

    data_validator._check_site_and_array_element_consistency("LSTN-01", "North")
    data_validator._check_site_and_array_element_consistency("MSTN-01", "North")
    data_validator._check_site_and_array_element_consistency("SSTS-01", "South")

    with pytest.raises(ValueError, match=r"Site .* and instrument site .* are inconsistent"):
        data_validator._check_site_and_array_element_consistency("LSTN-01", "South")

    with pytest.raises(ValueError, match=r"Site .* and instrument site .* are inconsistent"):
        data_validator._check_site_and_array_element_consistency("SSTS-01", "North")

    data_validator._check_site_and_array_element_consistency(None, "North")
    data_validator._check_site_and_array_element_consistency("LSTN-01", None)
    data_validator._check_site_and_array_element_consistency("OBS-South", "South")

    data_validator._check_site_and_array_element_consistency(["LSTN-01", "LSTN-02"], "North")

    with pytest.raises(ValueError, match=r"Site .* and instrument site .* are inconsistent"):
        data_validator._check_site_and_array_element_consistency(["LSTN-01", "MSTN-01"], "South")


@pytest.mark.parametrize("required_range", [False, True])
def test_empty_table_columns_validate_units_without_value_ranges(required_range):
    validator = validate_data.DataValidator()
    column_description = {
        "name": "distance",
        "required": True,
        "type": "double",
        "unit": "m",
        "allowed_range": {"min": 0, "max": 10},
    }
    if required_range:
        column_description["required_range"] = {"min": 0, "max": 10}
    validator._data_description = [column_description]
    validator.data_table = Table({"distance": Column([], dtype="float64", unit="cm")})
    if required_range:
        with pytest.raises(
            ValueError, match="Empty column 'distance' cannot cover its required range"
        ):
            validator._validate_data_columns()
    else:
        validator._validate_data_columns()
        assert validator.data_table["distance"].unit == u.m
        assert len(validator.data_table) == 0


def test_validate_model_parameter_keeps_single_item_json_array():
    """A one-item JSON array of a dict parameter stays a list and validates again."""
    record = {
        "parameter": "camera_pixel_types",
        "parameter_version": "1.0.0",
        "model_parameter_schema_version": schema.get_model_parameter_schema("camera_pixel_types")[
            "schema_version"
        ],
        "type": "dict",
        "unit": None,
        "file": False,
        "value": [
            {
                "type_id": 1,
                "pmt_type": 0,
                "cathode_shape": 0,
                "cathode_diameter_cm": 2.5,
                "funnel_shape": 3,
                "funnel_diameter_cm": 4.9,
                "funnel_depth_cm": 5.1,
            }
        ],
    }

    validated = validate_data.DataValidator.validate_model_parameter(copy.deepcopy(record))

    assert validated["value"] == record["value"]
    assert (
        validate_data.DataValidator.validate_model_parameter(validated)["value"] == record["value"]
    )


def test_validate_model_parameter_keeps_empty_json_array():
    """An empty JSON array of a dict parameter (allowed by its schema) stays empty."""
    record = {
        "parameter": "array_layouts",
        "parameter_version": "1.0.0",
        "model_parameter_schema_version": schema.get_model_parameter_schema("array_layouts")[
            "schema_version"
        ],
        "type": "dict",
        "unit": None,
        "file": False,
        "value": [],
    }

    validated = validate_data.DataValidator.validate_model_parameter(record)

    assert validated["value"] == []
    assert validated["unit"] is None
