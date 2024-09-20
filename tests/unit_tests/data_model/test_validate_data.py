#!/usr/bin/python3

import logging
import re
import shutil
import sys
from importlib.resources import files

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.table import Column, Table
from astropy.utils.diff import report_diff_values

from simtools.data_model import validate_data

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


mirror_file = "tests/resources/MLTdata-preproduction.ecsv"
mirror_2f_schema_file = "tests/resources/MST_mirror_2f_measurements.schema.yml"


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


def test_validate_and_transform(caplog):
    data_validator = validate_data.DataValidator()
    # no input file defined
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TypeError):
            data_validator.validate_and_transform()
    assert "No data or data table to validate" in caplog.text

    data_validator.data_file_name = mirror_file
    data_validator.schema_file_name = mirror_2f_schema_file
    with caplog.at_level(logging.INFO):
        _table = data_validator.validate_and_transform()
        assert isinstance(_table, Table)
    assert "Validating tabled data from:" in caplog.text

    data_validator.data_file_name = "tests/resources/model_parameters/num_gains.json"
    data_validator.schema_file_name = "tests/resources/num_gains.schema.yml"
    with caplog.at_level(logging.INFO):
        _dict = data_validator.validate_and_transform()
        assert isinstance(_dict, dict)
    assert "Validating data from:" in caplog.text


def test_validate_data_file(caplog):
    data_validator = validate_data.DataValidator()
    # no input file defined, should pass
    data_validator.validate_data_file()

    data_validator.data_file_name = mirror_file
    with caplog.at_level(logging.INFO):
        data_validator.validate_data_file()
    assert "Validating tabled data from:" in caplog.text

    data_validator.data_file_name = "tests/resources/reference_point_altitude.json"
    with caplog.at_level(logging.INFO):
        data_validator.validate_data_file()
    assert "Validating data from:" in caplog.text


def test_validate_parameter_and_file_name():

    data_validator = validate_data.DataValidator()
    data_validator.data_file_name = "tests/resources/model_parameters/num_gains.json"
    data_validator.schema_file_name = "tests/resources/num_gains.schema.yml"
    data_validator.validate_and_transform()

    data_validator.data_dict["parameter"] = "incorrect_name"
    with pytest.raises(
        ValueError,
        match="Parameter name in data dict incorrect_name and file name num_gains do not match.",
    ):
        data_validator.validate_parameter_and_file_name()


def test_validate_data_columns(tmp_test_directory, caplog):
    data_validator = validate_data.DataValidator()
    with pytest.raises(TypeError):
        data_validator._validate_data_table()

    data_validator_1 = validate_data.DataValidator(
        schema_file=None,
        data_file=mirror_file,
    )
    data_validator_1.validate_data_file()
    with pytest.raises(TypeError):
        data_validator_1._validate_data_table()

    data_validator_3 = validate_data.DataValidator(
        schema_file=mirror_2f_schema_file,
        data_file=mirror_file,
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
        assert "No data table defined for reverse sorting" in caplog.text


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


def test_interval_check_allow_range():
    data_validator = validate_data.DataValidator()

    assert data_validator._interval_check((0.1, 0.9), (0.0, 1.0), "allowed_range")
    assert data_validator._interval_check((0.0, 1.0), (0.0, 1.0), "allowed_range")

    assert not data_validator._interval_check((-1.0, 0.9), (0.0, 1.0), "allowed_range")
    assert not data_validator._interval_check((0.0, 1.1), (0.0, 1.0), "allowed_range")
    assert not data_validator._interval_check((-1.0, 1.1), (0.0, 1.0), "allowed_range")


def test_interval_check_required_range():
    data_validator = validate_data.DataValidator()

    assert data_validator._interval_check((250.0, 700.0), (300.0, 600), "required_range")
    assert data_validator._interval_check((300.0, 600.0), (300.0, 600), "required_range")

    assert not data_validator._interval_check((350.0, 700.0), (300.0, 600), "required_range")
    assert not data_validator._interval_check((300.0, 500.0), (300.0, 600), "required_range")
    assert not data_validator._interval_check((350.0, 500.0), (300.0, 600), "required_range")


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

    with pytest.raises(KeyError):
        data_validator._check_range(col_w.name, col_w.min(), col_w.max(), "failed_range")

    col_3 = Column(name="qe", data=[0.1, 5.00], dtype="float32")
    with pytest.raises(ValueError, match=r"Value for column 'qe' out of range"):
        data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "allowed_range")
    col_3 = Column(name="qe", data=[-0.1, 0.5], dtype="float32")
    with pytest.raises(ValueError, match=r"^Value for column 'qe' out of range"):
        data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "allowed_range")

    with pytest.raises(KeyError, match=r"Allowed range types are"):
        data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "invalid_range")


def test_is_dimensionless():

    data_validator = validate_data.DataValidator()

    assert data_validator._is_dimensionless(None)
    assert data_validator._is_dimensionless("")
    assert data_validator._is_dimensionless("dimensionless")
    assert not data_validator._is_dimensionless("kpc")


def test_check_and_convert_units(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    table_1 = Table()
    table_1["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], unit="dimensionless", dtype="float32")
    table_1["position_x"] = [0.1, 0.5] * u.km
    table_1["position_y"] = Column([5.0, 7], unit="km", dtype="float32")

    for col_name in table_1.colnames:
        table_1[col_name], _ = data_validator._check_and_convert_units(
            table_1[col_name], unit=None, col_name=col_name
        )

    # check unit conversion for "position_x" (column type Quantity)
    assert table_1["position_x"].unit == u.m
    assert 100.0 == pytest.approx(table_1["position_x"].value[0])
    # check unit conversion for "position_y" (column type Column)
    assert table_1["position_y"].unit == u.m
    assert 7000.0 == pytest.approx(table_1["position_y"].value[1])


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


def test_check_and_convert_units_simple_numbers(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    # convert numbers and quantities
    assert data_validator._check_and_convert_units(300.0, unit="nm", col_name="wavelength") == (
        300.0,
        u.nm,
    )
    assert data_validator._check_and_convert_units(300.0, unit="mm", col_name="wavelength") == (
        300000000.0,
        u.nm,
    )
    data_validator._data_description[0]["type"] = "int"
    assert data_validator._check_and_convert_units(300, unit="nm", col_name="wavelength") == (
        300,
        u.nm,
    )


def test_check_and_convert_units_dimensionless(reference_columns, caplog):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    assert data_validator._check_and_convert_units(0.1, unit="dimensionless", col_name="qe") == (
        0.1,
        u.dimensionless_unscaled,
    )

    # reference column requires a unit, give no unit
    with caplog.at_level(logging.ERROR):
        with pytest.raises(u.core.UnitConversionError):
            data_validator._check_and_convert_units(
                300.0, unit="dimensionless", col_name="wavelength"
            )
        assert "Invalid unit in data column " in caplog.text


def test_check_and_convert_units_integer_arrays(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    data_validator._data_description[0]["type"] = ["int", "int"]
    assert data_validator._check_and_convert_units(
        [300, 350], unit=["nm", "nm"], col_name="wavelength"
    ) == ([300, 350], u.nm)
    data_validator._data_description[0]["type"] = ["int", "int"]
    assert data_validator._check_and_convert_units(
        [300, 350], unit=["nm", None], col_name="wavelength"
    ) == ([300, 350], u.nm)


def test_check_required_columns(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    table_1 = Table()
    table_1["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], dtype="float32")

    data_validator.data_table = table_1
    data_validator._check_required_columns()

    table_2 = Table()
    table_2["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")

    data_validator.data_table = table_2

    with pytest.raises(KeyError, match=r"'Missing required column qe'"):
        data_validator._check_required_columns()


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


def test_get_reference_unit(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    assert data_validator._get_reference_unit("wavelength") == "nm"
    assert data_validator._get_reference_unit("qe") == u.dimensionless_unscaled
    assert data_validator._get_reference_unit("no_units") == u.dimensionless_unscaled


def test_get_unique_column_requirements(reference_columns):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    assert data_validator._get_unique_column_requirement() == ["wavelength"]


def test_check_data_type(reference_columns, caplog):
    data_validator = validate_data.DataValidator()
    data_validator._data_description = reference_columns

    with caplog.at_level(logging.DEBUG):
        assert data_validator._check_data_type(np.dtype("double"), "wavelength") is None

    with caplog.at_level(logging.ERROR):
        with pytest.raises(TypeError):
            assert data_validator._check_data_type(np.dtype("float32"), "wavelength")
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
    data_validator._read_validation_schema(schema_file=mirror_2f_schema_file)

    # file does not exist
    with pytest.raises(FileNotFoundError):
        data_validator._read_validation_schema(schema_file="this_file_does_not_exist.yml")

    # file given and parameter name given
    data_validator._read_validation_schema(
        schema_file=mirror_2f_schema_file,
        parameter="mirror_2f_measurement",
    )

    # copy the schema file to a temporary directory; this is to test
    # that the schema file is read from the temporary directory with the
    # correct path / name
    shutil.copy(
        mirror_2f_schema_file,
        tmp_test_directory / "mirror_2f_measurement.schema.yml",
    )
    data_validator._read_validation_schema(
        schema_file=str(tmp_test_directory), parameter="mirror_2f_measurement"
    )

    _incomplete_schema = {"description": "test schema"}
    # write yaml file in temp directory
    with open(tmp_test_directory / "incomplete_schema.schema.yml", "w") as _file:
        yaml.dump(_incomplete_schema, _file)

    with pytest.raises(KeyError):
        data_validator._read_validation_schema(
            schema_file=str(tmp_test_directory), parameter="incomplete_schema"
        )


# incomplete test
def test_validate_data_dict():

    schema_dir = files("simtools").joinpath("schemas/model_parameters/")

    # parameter with unit
    data_validator = validate_data.DataValidator(
        schema_file=str(schema_dir) + "/reference_point_altitude.schema.yml"
    )
    data_validator.data_dict = {
        "name": "reference_point_altitude",
        "value": 1000.0,
        "unit": "km",
    }
    data_validator._validate_data_dict()

    # parameter without unit
    data_validator_2 = validate_data.DataValidator(
        schema_file=str(schema_dir) + "/num_gains.schema.yml"
    )
    data_validator_2.data_dict = {"name": "num_gains", "value": [2], "unit": [""]}
    data_validator_2._validate_data_dict()

    data_validator_2.data_dict = {"name": "num_gains", "value": np.array([2]), "unit": [""]}
    data_validator_2._validate_data_dict()

    data_validator.data_dict = {
        "no_name": "test_data",
        "value": [1.0, 2.0, 3.0],
        "unit": ["", "", ""],
    }
    with pytest.raises(KeyError):
        data_validator._validate_data_dict()

    data_validator_2.data_dict = {"name": "num_gains", "value": [2], "unit": [None]}
    data_validator_2._validate_data_dict()


def test_prepare_model_parameter():
    data_validator = validate_data.DataValidator()
    data_validator.data_dict = {
        "name": "reference_point_altitude",
        "value": 1000.0,
        "unit": "km",
    }
    data_validator._prepare_model_parameter()
    assert pytest.approx(data_validator.data_dict["value"]) == 1000.0
    assert data_validator.data_dict["unit"] == "km"

    data_validator.data_dict["value"] = "1000. 2000. 3000."
    data_validator._prepare_model_parameter()
    assert pytest.approx(data_validator.data_dict["value"][0]) == 1000.0
    assert pytest.approx(data_validator.data_dict["value"][2]) == 3000.0
    assert data_validator.data_dict["unit"] == "km"

    data_validator.data_dict["value"] = "1000. 2000. 3000."
    data_validator.data_dict["unit"] = "km, kg, s"
    data_validator._prepare_model_parameter()
    assert data_validator.data_dict["unit"][0] == "km"
    assert data_validator.data_dict["unit"][1] == "kg"
    assert data_validator.data_dict["unit"][2] == "s"

    data_validator.data_dict["value"] = "1000. 2000. 3000."
    data_validator.data_dict["unit"] = ", , "
    data_validator._prepare_model_parameter()
    assert all(item == "" for item in data_validator.data_dict["unit"])

    data_validator.data_dict["value"] = "1000. 2000. 3000."
    data_validator.data_dict["unit"] = "ct mV, m/s, N /m**2"
    data_validator._prepare_model_parameter()
    assert data_validator.data_dict["unit"][0] == "ct mV"
    assert data_validator.data_dict["unit"][1] == "m/s"
    assert data_validator.data_dict["unit"][2] == "N /m**2"

    data_validator.data_dict["value"] = "1000 2000 3000"
    data_validator.data_dict["unit"] = "ct"
    data_validator.data_dict["type"] = "int64"
    data_validator._prepare_model_parameter()
    assert isinstance(data_validator.data_dict["value"][0], int)


def test_check_version_string(caplog):
    data_validator = validate_data.DataValidator()

    valid_versions = [
        "1.0.0",
        "0.1.0",
        "2.3.4",
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-0.3.7",
        "1.0.0-x.7.z.92",
        "1.0.0-alpha+001",
        "1.0.0+20130313144700",
        "1.0.0-beta+exp.sha.5114f85",
    ]

    for version in valid_versions:
        with caplog.at_level("DEBUG"):
            data_validator._check_version_string(version)
        assert f"Valid version string '{version}'" in caplog.text
        caplog.clear()

    invalid_versions = [
        "1.0",
        "1.0.0.0",
        "1.0.a",
        "1.0.0-",
        "1.0.0+",
        "a.b.c",
    ]

    for version in invalid_versions:
        with pytest.raises(ValueError, match=f"Invalid version string '{re.escape(version)}'"):
            data_validator._check_version_string(version)

    assert data_validator._check_version_string(None) is None
