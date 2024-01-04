#!/usr/bin/python3

import logging
import shutil
import sys

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.table import Column, Table
from astropy.utils.diff import report_diff_values

from simtools.data_model import validate_data

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_validate_and_transform(caplog):
    data_validator = validate_data.DataValidator()
    # no input file defined
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TypeError):
            data_validator.validate_and_transform()
    assert "No data or data table to validate" in caplog.text

    data_validator.data_file_name = "tests/resources/MLTdata-preproduction.ecsv"
    data_validator.schema_file_name = "tests/resources/MST_mirror_2f_measurements.schema.yml"
    with caplog.at_level(logging.INFO):
        data_validator.validate_and_transform()
    assert "Validating tabled data from:" in caplog.text


def test_validate_data_file(caplog):
    data_validator = validate_data.DataValidator()
    # no input file defined, should pass
    data_validator.validate_data_file()

    data_validator.data_file_name = "tests/resources/MLTdata-preproduction.ecsv"
    with caplog.at_level(logging.INFO):
        data_validator.validate_data_file()
    assert "Validating tabled data from:" in caplog.text

    data_validator.data_file_name = "tests/resources/reference_position_mercator.yml"
    with caplog.at_level(logging.INFO):
        data_validator.validate_data_file()
    assert "Validating data from:" in caplog.text


def test_validate_data_columns(tmp_test_directory, caplog):
    data_validator = validate_data.DataValidator()
    with pytest.raises(TypeError):
        data_validator._validate_data_table()

    data_validator_1 = validate_data.DataValidator(
        schema_file=None,
        data_file="tests/resources/MLTdata-preproduction.ecsv",
    )
    data_validator_1.validate_data_file()
    with pytest.raises(TypeError):
        data_validator_1._validate_data_table()

    data_validator_3 = validate_data.DataValidator(
        schema_file="tests/resources/MST_mirror_2f_measurements.schema.yml",
        data_file="tests/resources/MLTdata-preproduction.ecsv",
    )
    data_validator_3.validate_data_file()
    data_validator_3._validate_data_table()

    _incomplete_schema = {"data": []}
    with open(tmp_test_directory / "incomplete_data_schema.schema.yml", "w") as _file:
        yaml.dump(_incomplete_schema, _file)

    data_validator_3.schema_file_name = tmp_test_directory / "incomplete_data_schema.schema.yml"
    with caplog.at_level(logging.ERROR):
        with pytest.raises(IndexError):
            data_validator_3._validate_data_table()
    assert "Error reading validation schema from" in caplog.text


def test_sort_data():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

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

    reverse_sorted_data_columns = get_reference_columns()
    reverse_sorted_data_columns[0]["input_processing"] = ["remove_duplicates", "reversesort"]
    data_validator_reverse = validate_data.DataValidator()
    data_validator_reverse._reference_data_columns = reverse_sorted_data_columns

    data_validator_reverse.data_table = table_1
    data_validator_reverse._sort_data()

    identical_reverse_sorted = report_diff_values(
        data_validator_reverse.data_table, table_reverse_sorted, fileobj=sys.stdout
    )

    assert identical_sorted
    assert identical_reverse_sorted

    data_validator_reverse.data_table = None
    with pytest.raises(AttributeError):
        data_validator_reverse._sort_data()
    data_validator.data_table = None
    with pytest.raises(AttributeError):
        data_validator._sort_data()


def test_check_data_for_duplicates():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

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
    with pytest.raises(ValueError):
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


def test_check_range(caplog):
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

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
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "allowed_range")
    assert "Value for column 'qe' out of range" in caplog.text
    col_3 = Column(name="qe", data=[-0.1, 0.5], dtype="float32")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            data_validator._check_range(col_3.name, col_3.min(), col_3.max(), "allowed_range")
    assert "Value for column 'qe' out of range" in caplog.text


def test_check_and_convert_units():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    table_1 = Table()
    table_1["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], unit=None, dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], unit="dimensionless", dtype="float32")
    table_1["position_x"] = [0.1, 0.5] * u.km
    table_1["position_y"] = Column([5.0, 7], unit="km", dtype="float32")

    for col_name in table_1.colnames:
        table_1[col_name] = data_validator._check_and_convert_units(table_1[col_name], col_name)

    # check unit conversion for "position_x" (column type Quantity)
    assert table_1["position_x"].unit == u.m
    assert 100.0 == pytest.approx(table_1["position_x"].value[0])
    # check unit conversion for "position_y" (column type Column)
    assert table_1["position_y"].unit == u.m
    assert 7000.0 == pytest.approx(table_1["position_y"].value[1])

    table_2 = Table()
    table_2["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_2["wrong_column"] = Column([0.1, 0.5], dtype="float32")

    with pytest.raises(IndexError):
        for col_name in table_2.colnames:
            data_validator._check_and_convert_units(table_2[col_name], col_name)

    table_3 = Table()
    table_3["wavelength"] = Column([300.0, 350.0], unit="kg", dtype="float32")

    with pytest.raises(u.core.UnitConversionError):
        for col_name in table_3.colnames:
            data_validator._check_and_convert_units(table_3[col_name], col_name)


def test_check_required_columns():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

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


def test_get_reference_data_column():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    assert isinstance(data_validator._get_reference_data_column("wavelength"), dict)

    _entry = data_validator._get_reference_data_column("wavelength")
    assert _entry["name"] == "wavelength"

    with pytest.raises(IndexError):
        data_validator._get_reference_data_column("wrong_column")

    assert data_validator._get_reference_data_column("wavelength", status_test=True)
    assert not data_validator._get_reference_data_column("wrong_column", status_test=True)

    data_validator._reference_data_columns = get_reference_columns_name_colx()

    assert isinstance(data_validator._get_reference_data_column("col1"), dict)

    with pytest.raises(IndexError):
        data_validator._get_reference_data_column("col3")

    assert data_validator._get_reference_data_column("col1", status_test=True)

    assert data_validator._get_reference_data_column("col1") == {"name": "col1"}


def test_get_reference_unit():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    assert data_validator._get_reference_unit("wavelength") == "nm"
    assert data_validator._get_reference_unit("qe") == u.dimensionless_unscaled
    assert data_validator._get_reference_unit("no_units") == u.dimensionless_unscaled


def test_get_unique_column_requirements():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    assert data_validator._get_unique_column_requirement() == ["wavelength"]


def test_check_data_type(caplog):
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    with caplog.at_level(logging.DEBUG):
        data_validator._check_data_type(
            Column([300.0, 350.0, 315.0], dtype="double", name="wavelength"), "wavelength"
        )
    assert "Data column 'wavelength' has correct data type" in caplog.text

    print("AAA", caplog.text)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(TypeError):
            assert data_validator._check_data_type(
                Column([300.0, 350.0, 315.0], dtype="float32", name="wavelength"), "wavelength"
            )
    assert (
        "Invalid data type in column 'wavelength'. Expected type 'double', found 'float32'"
        in caplog.text
    )


def test_check_for_not_a_number():
    data_validator = validate_data.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    assert not (
        data_validator._check_for_not_a_number(
            Column([300.0, 350.0, 315.0], dtype="float32", name="wavelength"), "wavelength"
        )
    )

    # wavelength does not allow for nan
    with pytest.raises(ValueError):
        data_validator._check_for_not_a_number(
            Column([np.nan, 350.0, 315.0], dtype="float32", name="wavelength"), "wavelength"
        )
    with pytest.raises(ValueError):
        data_validator._check_for_not_a_number(
            Column([np.nan, 350.0, np.inf], dtype="float32", name="wavelength"), "wavelength"
        )
    with pytest.raises(ValueError):
        data_validator._check_for_not_a_number(
            Column([300.0, 350.0, np.inf], dtype="float32", name="wavelength"), "wavelength"
        )

    # position_x allows for nan
    assert not (
        data_validator._check_for_not_a_number(
            Column([300.0, 350.0, 315.0], dtype="float32", name="position_x"), "position_x"
        )
    )
    assert data_validator._check_for_not_a_number(
        Column([np.nan, 350.0, 315.0], dtype="float32", name="position_x"), "position_x"
    )
    assert data_validator._check_for_not_a_number(
        Column([333.0, np.inf, 315.0], dtype="float32", name="position_x"), "position_x"
    )


def test_read_validation_schema(tmp_test_directory):
    data_validator = validate_data.DataValidator()

    # no file given
    with pytest.raises(TypeError):
        data_validator._read_validation_schema(schema_file=None)

    # file given
    data_validator._read_validation_schema(
        schema_file="tests/resources/MST_mirror_2f_measurements.schema.yml"
    )

    # file does not exist
    with pytest.raises(FileNotFoundError):
        data_validator._read_validation_schema(schema_file="this_file_does_not_exist.yml")

    # file given and parameter name given
    data_validator._read_validation_schema(
        schema_file="tests/resources/MST_mirror_2f_measurements.schema.yml",
        parameter="mirror_2f_measurement",
    )

    # copy the schema file to a temporary directory; this is to test
    # that the schema file is read from the temporary directory with the
    # correct path / name
    shutil.copy(
        "tests/resources/MST_mirror_2f_measurements.schema.yml",
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
    data_validator = validate_data.DataValidator(
        schema_file="tests/resources/MST_mirror_2f_measurements.schema.yml"
    )
    data_validator.data = {"no_name": "test_data", "value": [1, 2, 3], "units": ["", "", ""]}
    with pytest.raises(KeyError):
        data_validator._validate_data_dict()


def get_reference_columns_name_colx():
    """
    return a test reference data column definition
    with columns named col0, col1, col3

    """
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


def get_reference_columns():
    """
    return a test reference data column definition

    """
    return [
        {
            "name": "wavelength",
            "description": "wavelength",
            "required": True,
            "units": "nm",
            "type": "double",
            "required_range": {"unit": "nm", "min": 300, "max": 700},
            "input_processing": ["remove_duplicates", "sort"],
        },
        {
            "name": "qe",
            "description": "average quantum or photon detection efficiency",
            "required": True,
            "units": "dimensionless",
            "type": "double",
            "allowed_range": {"unit": "unitless", "min": 0.0, "max": 1.0},
        },
        {
            "name": "position_x",
            "description": "x position",
            "required": False,
            "units": "m",
            "type": "double",
            "allowed_range": {"unit": "m", "min": 0.0, "max": 1.0},
            "input_processing": ["allow_nan"],
        },
        {
            "name": "position_y",
            "description": "y position",
            "required": False,
            "units": "m",
            "type": "double",
            "allowed_range": {"unit": "m", "min": 0.0, "max": 1.0},
            "input_processing": ["allow_nan"],
        },
        {
            "name": "abc",
            "description": "not required",
            "required": False,
            "units": "kg",
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
