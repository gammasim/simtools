#!/usr/bin/python3

import logging
import sys

import pytest
from astropy import units as u
from astropy.table import Column, Table
from astropy.utils.diff import report_diff_values

import simtools.data_model.validate_data as ds

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_sort_data():

    data_validator = ds.DataValidator()
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
    reverse_sorted_data_columns["wavelength"]["attribute"] = ["remove_duplicates", "reversesort"]
    data_validator_reverse = ds.DataValidator()
    data_validator_reverse._reference_data_columns = reverse_sorted_data_columns

    data_validator_reverse.data_table = table_1
    data_validator_reverse._sort_data()

    identical_reverse_sorted = report_diff_values(
        data_validator_reverse.data_table, table_reverse_sorted, fileobj=sys.stdout
    )

    assert identical_sorted
    assert identical_reverse_sorted


def test_check_data_for_duplicates():

    data_validator = ds.DataValidator()
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

    data_validator = ds.DataValidator()

    assert data_validator._interval_check((0.1, 0.9), (0.0, 1.0), "allowed_range") == True
    assert data_validator._interval_check((0.0, 1.0), (0.0, 1.0), "allowed_range") == True

    assert data_validator._interval_check((-1.0, 0.9), (0.0, 1.0), "allowed_range") == False
    assert data_validator._interval_check((0.0, 1.1), (0.0, 1.0), "allowed_range") == False
    assert data_validator._interval_check((-1.0, 1.1), (0.0, 1.0), "allowed_range") == False


def test_interval_check_required_range():

    data_validator = ds.DataValidator()

    assert data_validator._interval_check((250.0, 700.0), (300.0, 600), "required_range") == True
    assert data_validator._interval_check((300.0, 600.0), (300.0, 600), "required_range") == True

    assert data_validator._interval_check((350.0, 700.0), (300.0, 600), "required_range") == False
    assert data_validator._interval_check((300.0, 500.0), (300.0, 600), "required_range") == False
    assert data_validator._interval_check((350.0, 500.0), (300.0, 600), "required_range") == False


def test_check_range():

    data_validator = ds.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    col_1 = Column(name="qe", data=[0.1, 0.5], dtype="float32")
    data_validator._check_range(col_1.name, col_1.min(), col_1.max(), "allowed_range")
    col_w = Column(name="wavelength", data=[250.0, 750.0], dtype="float32")
    data_validator._check_range(col_w.name, col_w.min(), col_w.max(), "required_range")

    col_2 = Column(name="key_error", data=[0.1, 0.5], dtype="float32")
    with pytest.raises(KeyError):
        data_validator._check_range(col_2.name, col_2.min(), col_2.max(), "allowed_range")

    with pytest.raises(KeyError):
        data_validator._check_range(col_w.name, col_w.min(), col_w.max(), "failed_range")


def test_column_units():

    data_validator = ds.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    table_1 = Table()
    table_1["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], unit=None, dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], unit="dimensionless", dtype="float32")

    for col in table_1.itercols():
        data_validator._check_and_convert_units(col)

    table_2 = Table()
    table_2["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_2["wrong_column"] = Column([0.1, 0.5], dtype="float32")

    with pytest.raises(KeyError, match=r"'wrong_column'"):
        for col in table_2.itercols():
            data_validator._check_and_convert_units(col)

    table_3 = Table()
    table_3["wavelength"] = Column([300.0, 350.0], unit="kg", dtype="float32")

    with pytest.raises(u.core.UnitConversionError):
        for col in table_3.itercols():
            data_validator._check_and_convert_units(col)


def test_check_required_columns():

    data_validator = ds.DataValidator()
    data_validator._reference_data_columns = get_reference_columns()

    table_1 = Table()
    table_1["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")
    table_1["qe"] = Column([0.1, 0.5], dtype="float32")

    data_validator.data_table = table_1
    data_validator._check_required_columns()

    table_2 = Table()
    table_2["wavelength"] = Column([300.0, 350.0], unit="nm", dtype="float32")

    data_validator.data_table = table_2

    with pytest.raises(KeyError, match=r"'qe'"):
        data_validator._check_required_columns()


def get_reference_columns():
    """
    return a test reference data column definition

    """
    return {
        "wavelength": {
            "description": "wavelength",
            "required_column": True,
            "unit": "nm",
            "type": "float32",
            "required_range": {"unit": "nm", "min": 300, "max": 700},
            "attribute": ["remove_duplicates", "sort"],
        },
        "qe": {
            "description": "average quantum or photon detection efficiency",
            "required_column": True,
            "unit": "dimensionless",
            "type": "float32",
            "allowed_range": {"unit": "unitless", "min": 0.0, "max": 1.0},
        },
        "abc": {
            "description": "not required",
            "required_column": False,
            "unit": "kg",
            "type": "float32",
            "allowed_range": {"unit": "kg", "min": 0.0, "max": 100.0},
        },
    }


def get_generic_workflow_config():

    return {
        "CTASIMPIPE": {
            "ACTIVITY": {"NAME": "workflow_name"},
            "DATAMODEL": {
                "USERINPUTSCHEMA": "schema",
                "TOPLEVELMODEL": "model",
                "SCHEMADIRECTORY": "directory",
            },
            "PRODUCT": {"DIRECTORY": None},
        }
    }
