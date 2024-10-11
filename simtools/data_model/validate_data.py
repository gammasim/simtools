"""Validation of data using schema."""

import logging
import os
import re
from pathlib import Path

import jsonschema
import numpy as np
from astropy import units as u
from astropy.table import Column, Table, unique
from astropy.utils.diff import report_diff_values

import simtools.utils.general as gen
from simtools.data_model import format_checkers

__all__ = ["DataValidator"]


class DataValidator:
    """
    Validate data for type and units following a describing schema; converts or transform data.

    Data can be of table or dict format (internally, all data is converted to astropy tables).

    Parameters
    ----------
    schema_file: Path
        Schema file describing input data and transformations.
    data_file: Path
        Input data file.
    data_table: astropy.table
        Input data table.
    data_dict: dict
        Input data dict.
    check_exact_data_type: bool
        Check for exact data type (default: True).

    """

    def __init__(
        self,
        schema_file=None,
        data_file=None,
        data_table=None,
        data_dict=None,
        check_exact_data_type=True,
    ):
        """Initialize validation class and read required reference data columns."""
        self._logger = logging.getLogger(__name__)

        self.data_file_name = data_file
        self.schema_file_name = schema_file
        self._data_description = None
        self.data_dict = data_dict
        self.data_table = data_table
        self.check_exact_data_type = check_exact_data_type

    def validate_and_transform(self, is_model_parameter=False):
        """
        Validate data and data file.

        Parameters
        ----------
        is_model_parameter: bool
            This is a model parameter (add some data preparation)

        Returns
        -------
        data: dict or astropy.table
            Data dict or table

        Raises
        ------
        TypeError
            if no data or data table is available

        """
        if self.data_file_name:
            self.validate_data_file()
        if isinstance(self.data_dict, dict):
            if is_model_parameter:
                self._prepare_model_parameter()
            self._validate_data_dict()
            return self.data_dict
        if isinstance(self.data_table, Table):
            self._validate_data_table()
            return self.data_table
        self._logger.error("No data or data table to validate")
        raise TypeError

    def validate_data_file(self):
        """
        Open data file and read data from file.

        Doing this successfully is understood as file validation.
        """
        try:
            if Path(self.data_file_name).suffix in (".yml", ".yaml", ".json"):
                self.data_dict = gen.collect_data_from_file_or_dict(self.data_file_name, None)
                self._logger.info(f"Validating data from: {self.data_file_name}")
            else:
                self.data_table = Table.read(self.data_file_name, guess=True, delimiter=r"\s")
                self._logger.info(f"Validating tabled data from: {self.data_file_name}")
        except (AttributeError, TypeError):
            pass

    def validate_parameter_and_file_name(self):
        """Validate that file name and key 'parameter_name' in data dict are the same."""
        if self.data_dict.get("parameter") != Path(self.data_file_name).stem:
            raise ValueError(
                f"Parameter name in data dict {self.data_dict.get('parameter')} and "
                f"file name {Path(self.data_file_name).stem} do not match."
            )

    def _validate_data_dict(self):
        """
        Validate values in a dictionary.

        Handles different types of naming in data dicts (using 'name' or 'parameter'
        keys for name fields).

        Raises
        ------
        KeyError
            if data dict does not contain a 'name' or 'parameter' key.

        """
        if not (_name := self.data_dict.get("name") or self.data_dict.get("parameter")):
            raise KeyError("Data dict does not contain a 'name' or 'parameter' key.")
        self._data_description = self._read_validation_schema(self.schema_file_name, _name)

        value_as_list, unit_as_list = self._get_value_and_units_as_lists()

        for index, (value, unit) in enumerate(zip(value_as_list, unit_as_list)):
            value_as_list[index], unit_as_list[index] = self._validate_value_and_unit(
                value, unit, index
            )

        if len(value_as_list) == 1:
            self.data_dict["value"], self.data_dict["unit"] = value_as_list[0], unit_as_list[0]

        self._check_version_string(self.data_dict.get("version"))

    def _validate_value_and_unit(self, value, unit, index):
        """
        Validate value, unit, and perform type checking and conversions.

        Take into account different data types and allow to use json_schema for testing.
        """
        if self._get_data_description(index).get("type", None) == "dict":
            self._validate_data_dict_using_json_schema(
                self.data_dict["value"], self._get_data_description(index).get("json_schema")
            )
        else:
            self._check_data_type(np.array(value).dtype, index)

        if self.data_dict.get("type") not in ("string", "dict"):
            self._check_for_not_a_number(value, index)
            value, unit = self._check_and_convert_units(value, unit, index)
            for range_type in ("allowed_range", "required_range"):
                self._check_range(index, np.nanmin(value), np.nanmax(value), range_type)
        return value, unit

    def _get_value_and_units_as_lists(self):
        """
        Convert value and unit to lists if required.

        Returns
        -------
        list
            value as list
        list
            unit as list
        """
        value_as_list = (
            self.data_dict.get("value")
            if isinstance(self.data_dict["value"], list | np.ndarray)
            else [self.data_dict["value"]]
        )
        unit_as_list = (
            self.data_dict.get("unit")
            if isinstance(self.data_dict["unit"], list | np.ndarray)
            else [self.data_dict["unit"]]
        )
        if isinstance(value_as_list, np.ndarray):
            value_as_list = value_as_list.tolist()
        if isinstance(unit_as_list, np.ndarray):
            unit_as_list = unit_as_list.tolist()
        unit_as_list = [None if unit == "null" else unit for unit in unit_as_list]

        return value_as_list, unit_as_list

    def _validate_data_dict_using_json_schema(self, data, json_schema):
        """
        Validate a dictionary using a json schema.

        Parameters
        ----------
        data: dict
            Data dictionary
        json_schema: dict
            JSON schema
        """
        if json_schema is None:
            self._logger.debug("Skipping validation of dict type")
            return
        self._logger.debug("Validation of dict type using JSON schema")
        try:
            jsonschema.validate(data, json_schema, format_checker=format_checkers.format_checker)
        except jsonschema.exceptions.ValidationError as exc:
            self._logger.error(f"Validation error: {exc}")
            raise exc

    def _validate_data_table(self):
        """Validate tabulated data."""
        try:
            self._data_description = self._read_validation_schema(self.schema_file_name)[0].get(
                "table_columns", None
            )
        except IndexError:
            self._logger.error(f"Error reading validation schema from {self.schema_file_name}")
            raise

        if self._data_description is not None:
            self._validate_data_columns()
            self._check_data_for_duplicates()
            self._sort_data()

    def _validate_data_columns(self):
        """
        Validate that data columns.

        This includes:

        - required data columns are available
        - columns are in the correct units (if necessary apply a unit conversion)
        - ranges (minimum, maximum) are correct.

        This is not applied to columns of type 'string'.

        """
        self._check_required_columns()

        for col_name in self.data_table.colnames:
            col = self.data_table[col_name]
            if not self._get_data_description(col_name, status_test=True):
                continue
            if not np.issubdtype(col.dtype, np.number):
                continue
            self._check_for_not_a_number(col.data, col_name)
            self._check_data_type(col.dtype, col_name)
            self.data_table[col_name] = col.to(u.Unit(self._get_reference_unit(col_name)))
            self._check_range(col_name, np.nanmin(col.data), np.nanmax(col.data), "allowed_range")
            self._check_range(col_name, np.nanmin(col.data), np.nanmax(col.data), "required_range")

    def _check_required_columns(self):
        """
        Check that all required data columns are available in the input data table.

        Raises
        ------
        KeyError
            if a required data column is missing

        """
        for entry in self._data_description:
            if entry.get("required", False):
                if entry["name"] in self.data_table.columns:
                    self._logger.debug(f"Found required data column {entry['name']}")
                else:
                    raise KeyError(f"Missing required column {entry['name']}")

    def _sort_data(self):
        """
        Sort data according to one data column (if required by any column attribute).

        Data is either sorted or reverse sorted.

        Raises
        ------
        AttributeError
            if no table is defined for sorting

        """
        _columns_by_which_to_sort = []
        _columns_by_which_to_reverse_sort = []
        for entry in self._data_description:
            if "input_processing" in entry:
                if "sort" in entry["input_processing"]:
                    _columns_by_which_to_sort.append(entry["name"])
                elif "reversesort" in entry["input_processing"]:
                    _columns_by_which_to_reverse_sort.append(entry["name"])

        if len(_columns_by_which_to_sort) > 0:
            self._logger.debug(f"Sorting data columns: {_columns_by_which_to_sort}")
            try:
                self.data_table.sort(_columns_by_which_to_sort)
            except AttributeError:
                self._logger.error("No data table defined for sorting")
                raise
        elif len(_columns_by_which_to_reverse_sort) > 0:
            self._logger.debug(f"Reverse sorting data columns: {_columns_by_which_to_reverse_sort}")
            try:
                self.data_table.sort(_columns_by_which_to_reverse_sort, reverse=True)
            except AttributeError:
                self._logger.error("No data table defined for reverse sorting")
                raise

    def _check_data_for_duplicates(self):
        """
        Remove duplicates from data columns as defined in the data columns description.

        Raises
        ------
            if row values are different for those rows with duplications in the data columns to be
            checked for unique values.

        """
        _column_with_unique_requirement = self._get_unique_column_requirement()
        if len(_column_with_unique_requirement) == 0:
            self._logger.debug("No data columns with unique value requirement")
            return
        _data_table_unique_for_key_column = unique(
            self.data_table, keys=_column_with_unique_requirement
        )
        _data_table_unique_for_all_columns = unique(self.data_table, keys=None)
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            if report_diff_values(
                _data_table_unique_for_key_column,
                _data_table_unique_for_all_columns,
                fileobj=devnull,
            ):
                self.data_table = unique(self.data_table)
            else:
                raise ValueError(
                    "Failed removal of duplication for column "
                    f"{_column_with_unique_requirement}, values are not unique"
                )

    def _get_unique_column_requirement(self):
        """
        Return data column name with unique value requirement.

        Returns
        -------
        list
            list of data column with unique value requirement

        """
        _unique_required_column = []

        for entry in self._data_description:
            if "input_processing" in entry and "remove_duplicates" in entry["input_processing"]:
                self._logger.debug(f"Removing duplicates for column {entry['name']}")
                _unique_required_column.append(entry["name"])

        self._logger.debug(f"Unique required columns: {_unique_required_column}")
        return _unique_required_column

    def _get_reference_unit(self, column_name):
        """
        Return reference column unit. Includes correct treatment of dimensionless units.

        Parameters
        ----------
        column_name: str
            column name of reference data column

        Returns
        -------
        astro.unit
            unit for reference column

        Raises
        ------
        KeyError
            if column name is not found in reference data columns

        """
        reference_unit = self._get_data_description(column_name).get("unit", None)
        if reference_unit in ("dimensionless", None, ""):
            return u.dimensionless_unscaled

        return u.Unit(reference_unit)

    def _check_data_type(self, dtype, column_name):
        """
        Check column data type.

        Parameters
        ----------
        dtype: numpy.dtype
            data type
        column_name: str
            column name

        Raises
        ------
        TypeError
            if data type is not correct

        """
        reference_dtype = self._get_data_description(column_name).get("type", None)
        if not gen.validate_data_type(
            reference_dtype=reference_dtype,
            value=None,
            dtype=dtype,
            allow_subtypes=(not self.check_exact_data_type),
        ):
            self._logger.error(
                f"Invalid data type in column '{column_name}'. "
                f"Expected type '{reference_dtype}', found '{dtype}' "
                f"(exact type: {self.check_exact_data_type})"
            )
            raise TypeError

    def _check_for_not_a_number(self, data, col_name):
        """
        Check that column values are finite and not NaN.

        Parameters
        ----------
        data: value or numpy.ndarray
            data to be tested
        col_name: str
            column name

        Returns
        -------
        bool
            if at least one column value is NaN or Inf.

        Raises
        ------
        ValueError
            if at least one column value is NaN or Inf.

        """
        if isinstance(data, str):
            return True
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if np.isnan(data).any():
            self._logger.info(f"Column {col_name} contains NaN.")
        if np.isinf(data).any():
            self._logger.info(f"Column {col_name} contains infinite value.")

        entry = self._get_data_description(col_name)
        if "allow_nan" in entry.get("input_processing", {}):
            return np.isnan(data).any() or np.isinf(data).any()

        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("NaN or Inf values found in data")

        return False

    @staticmethod
    def _is_dimensionless(unit):
        """
        Check if unit is dimensionless, None, or empty.

        Parameters
        ----------
        unit: str
            unit of data column

        Returns
        -------
        bool
            True if unit is dimensionless, None, or empty
        """
        return unit in ("dimensionless", None, "")

    def _check_and_convert_units(self, data, unit, col_name):
        """
        Check that input data have an allowed unit.

        Convert to reference unit (e.g., Angstrom to nm).

        Note on dimensionless columns:

        - should be given in unit descriptor as unit: ''
        - be forgiving and assume that in cases no unit is given in the data files
          means that it should be dimensionless (e.g., for a efficiency)

        Parameters
        ----------
        data: astropy.column, Quantity, list, value
            data to be converted
        unit: str
            unit of data column (read from column or Quantity if possible)
        col_name: str
            column name

        Returns
        -------
        data: astropy.column, Quantity, list, value
            unit-converted data

        Raises
        ------
        u.core.UnitConversionError
            If unit conversions fails

        """
        self._logger.debug(f"Checking data column '{col_name}'")

        reference_unit = self._get_reference_unit(col_name)
        try:
            column_unit = data.unit
        except AttributeError:
            column_unit = unit

        if self._is_dimensionless(column_unit) and self._is_dimensionless(reference_unit):
            return data, u.dimensionless_unscaled

        self._logger.debug(
            f"Data column '{col_name}' with reference unit "
            f"'{reference_unit}' and data unit '{column_unit}'"
        )
        try:
            if isinstance(data, u.Quantity | Column):
                return data.to(reference_unit), reference_unit

            if isinstance(data, list | np.ndarray):
                return self._check_and_convert_units_for_list(data, column_unit, reference_unit)

            # ensure that the data type is preserved (e.g., integers)
            return (type(data)(u.Unit(column_unit).to(reference_unit) * data), reference_unit)
        except (u.core.UnitConversionError, ValueError) as exc:
            self._logger.error(
                f"Invalid unit in data column '{col_name}'. "
                f"Expected type '{reference_unit}', found '{column_unit}'"
            )
            raise u.core.UnitConversionError from exc

    def _check_and_convert_units_for_list(self, data, column_unit, reference_unit):
        """
        Check and convert units data in a list or or numpy array.

        Takes into account that data can be dimensionless (with unit 'None', 'dimensionless'
        or '').

        Parameters
        ----------
        data: list
            list of data
        column_unit: str
            unit of data column
        reference_unit: str
            reference unit

        Returns
        -------
        list
            converted data

        """
        return [
            (
                u.Unit(_to_unit).to(reference_unit) * d
                if _to_unit not in (None, "dimensionless", "")
                else d
            )
            for d, _to_unit in zip(data, column_unit)
        ], reference_unit

    def _check_range(self, col_name, col_min, col_max, range_type="allowed_range"):
        """
        Check that column data is within allowed range or required range.

        Assumes that column and ranges have the same units.

        Parameters
        ----------
        col_name: string
            column name
        col_min: float
            minimum value of data column
        col_max: float
            maximum value of data column
        range_type: string
            column range type (either 'allowed_range' or 'required_range')

        Raises
        ------
        ValueError
            if columns are not in the required range
        KeyError
            if requested columns cannot be found or
            if there is now defined of required or allowed
            range columns

        """
        self._logger.debug(f"Checking data in column '{col_name}' for '{range_type}' ")

        if range_type not in ("allowed_range", "required_range"):
            raise KeyError("Allowed range types are 'allowed_range', 'required_range'")

        _entry = self._get_data_description(col_name)
        if range_type not in _entry:
            return

        if not self._interval_check(
            (col_min, col_max),
            (_entry[range_type].get("min", -np.inf), _entry[range_type].get("max", np.inf)),
            range_type,
        ):
            raise ValueError(
                f"Value for column '{col_name}' out of range. "
                f"([{col_min}, {col_max}], {range_type}: "
                f"[{_entry[range_type].get('min', -np.inf)}, "
                f"{_entry[range_type].get('max', np.inf)}])"
            )

    @staticmethod
    def _interval_check(data, axis_range, range_type):
        """
        Range checking for a given set of data.

        Check that values are inside allowed range or interval. This(range_type='allowed_range')
        or span at least the given interval (range_type='required_range').

        Parameters
        ----------
        data: tuple
            min and max of data
        axis_range: tuple
            allowed or required min max
        range_type: string
            column range type (either 'allowed_range' or 'required_range')

        Returns
        -------
        boolean
            True if range test is passed

        """
        if range_type == "allowed_range":
            if data[0] >= axis_range[0] and data[1] <= axis_range[1]:
                return True
        if range_type == "required_range":
            if data[0] <= axis_range[0] and data[1] >= axis_range[1]:
                return True

        return False

    def _read_validation_schema(self, schema_file, parameter=None):
        """
        Read validation schema from file.

        Parameters
        ----------
        schema_file: Path
            Schema file describing input data.
            If this is a directory, a filename of
            '<par>.schema.yml' is assumed.
        parameter: str
            Parameter name of required schema
            (if None, return first schema in file)

        Returns
        -------
        dict
           validation schema

        Raises
        ------
        KeyError
            if 'data' can not be read from dict in schema file

        """
        try:
            if Path(schema_file).is_dir():
                return gen.collect_data_from_file_or_dict(
                    file_name=Path(schema_file) / (parameter + ".schema.yml"),
                    in_dict=None,
                )["data"]
            return gen.collect_data_from_file_or_dict(file_name=schema_file, in_dict=None)["data"]
        except KeyError:
            self._logger.error(f"Error reading validation schema from {schema_file}")
            raise

    def _get_data_description(self, column_name=None, status_test=False):
        """
        Return data description as provided by the schema file.

        For tables (type: 'data_table'), return the description of
        the column named 'column_name'. For other types, return
        all data descriptions.
        For columns named 'colX' return the Xth column in the reference data.

        Parameters
        ----------
        column_name: str
            Column name.
        status_test: bool
            Test if reference column exists.

        Returns
        -------
        dict
            Reference schema column (for status_test==False).
        bool
            True if reference column exists (for status_test==True).

        Raises
        ------
        IndexError
            If data column is not found.

        """
        self._logger.debug(
            f"Getting reference data column {column_name} from schema {self._data_description}"
        )
        try:
            return (
                self._data_description[column_name]
                if not status_test
                else (
                    self._data_description[column_name] is not None
                    and len(self._data_description) > 0
                )
            )
        except IndexError as exc:
            self._logger.error(
                f"Data column '{column_name}' not found in reference column definition"
            )
            raise exc
        except TypeError:
            pass  # column_name is not an integer

        _index = 0
        if bool(re.match(r"^col\d$", column_name)):
            _index = int(column_name[3:])
            _entry = self._data_description
        else:
            _entry = [item for item in self._data_description if item["name"] == column_name]
        if status_test:
            return len(_entry) > 0
        try:
            return _entry[_index]
        except IndexError:
            self._logger.error(
                f"Data column '{column_name}' not found in reference column definition"
            )
            raise

    def _prepare_model_parameter(self):
        """Apply data preparation for model parameters."""
        if isinstance(self.data_dict["value"], str):
            try:
                _is_float = self.data_dict.get("type").startswith("float") | self.data_dict.get(
                    "type"
                ).startswith("double")
            except AttributeError:
                _is_float = True
            self.data_dict["value"] = gen.convert_string_to_list(
                self.data_dict["value"], is_float=_is_float
            )
            self.data_dict["unit"] = (
                None
                if self.data_dict["unit"] is None
                else gen.convert_string_to_list(self.data_dict["unit"])
            )

    def _check_version_string(self, version):
        """
        Check that version string follows semantic versioning.

        Parameters
        ----------
        version: str
            version string

        Raises
        ------
        ValueError
            if version string does not follow semantic versioning

        """
        if version is None:
            return
        semver_regex = r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$"
        if not re.match(semver_regex, version):
            raise ValueError(f"Invalid version string '{version}'")
        self._logger.debug(f"Valid version string '{version}'")
