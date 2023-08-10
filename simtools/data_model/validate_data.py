import logging
import os

import numpy as np
import yaml
from astropy import units as u
from astropy.table import Table, unique
from astropy.utils.diff import report_diff_values

__all__ = ["DataValidator"]


class DataValidator:
    """
    Validate data for type and units following a describing schema; converts or
    transform data if required.

    Parameters
    ----------
    schema_file: Path
        Schema file describing input data and transformations.
    data_file: Path
        Input data file.

    """

    def __init__(self, schema_file=None, data_file=None):
        """
        Initalize validation class and read required reference data columns
        """

        self._logger = logging.getLogger(__name__)

        self._reference_data_columns = self._read_validation_schema(schema_file)
        self._data_file_name = data_file

        self.data_table = None

    def validate_and_transform(self):
        """
        Data and data file validation.  Apply transformations to data columns:
        - duplication removal
        - sorting according to axes

        Returns
        -------
        data_table: astropy.table
            Data table

        """

        self.validate_data_file()
        if self._reference_data_columns is not None:
            self._validate_data_columns()
            self._check_data_for_duplicates()
            self._sort_data()

        return self.data_table

    def validate_data_file(self):
        """
        Open data file and read data from file
        (doing this successfully is understood as
        file validation).

        """

        self._logger.info(f"Reading data from {self._data_file_name}")
        self.data_table = Table.read(self._data_file_name, guess=True, delimiter=r"\s")

    def _validate_data_columns(self):
        """
        Validate that required data columns are available, columns are in the correct units (if
        necessary apply a unit conversion), and check ranges (minimum, maximum). This is not
        applied to columns of type 'string'.

        """

        self._check_required_columns()

        for col in self.data_table.itercols():
            if not self._get_reference_data_column(col.name, status_test=True):
                continue
            if not np.issubdtype(col.dtype, np.number):
                continue
            self._check_for_not_a_number(col)
            self._check_and_convert_units(col)
            self._check_range(col.name, np.nanmin(col.data), np.nanmax(col.data), "allowed_range")
            self._check_range(col.name, np.nanmin(col.data), np.nanmax(col.data), "required_range")

    def _check_required_columns(self):
        """
        Check that all required data columns are available in the input data table.

        Raises
        ------
        KeyError
            if a required data column is missing

        """

        for entry in self._reference_data_columns:
            if entry.get("required_column", False):
                if entry["name"] in self.data_table.columns:
                    self._logger.debug(f"Found required data column {entry['name']}")
                else:
                    raise KeyError(f"Missing required column {entry['name']}")

    def _sort_data(self):
        """
        Sort data according to one data column (if required by any column attribute). Data is
         either sorted or reverse sorted

        Raises
        ------
        AttributeError
            if no table is defined for sorting

        """

        _columns_by_which_to_sort = []
        _columns_by_which_to_reverse_sort = []
        for entry in self._reference_data_columns:
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
                self._logger.error("No data table defined for sorting")
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
                self._logger.error(
                    "Failed removal of duplication for column "
                    f"{_column_with_unique_requirement}, values are not unqiue"
                )
                raise ValueError

    def _get_unique_column_requirement(self):
        """
        Return data column name with unique value requirement.

        Returns
        -------
        list
            list of data column with unique value requirement

        """

        _unique_required_column = []

        for entry in self._reference_data_columns:
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

        reference_unit = self._get_reference_data_column(column_name).get("units", None)
        if reference_unit == "dimensionless" or reference_unit is None:
            return u.dimensionless_unscaled

        return u.Unit(reference_unit)

    def _check_for_not_a_number(self, col):
        """
        Check that column values are finite and not NaN.

        Parameters
        ----------
        col: astropy.column
            data column to be converted

        Returns
        -------
        bool
            if at least one column value is NaN or Inf.

        Raises
        ------
        ValueError
            if at least one column value is NaN or Inf.

        """

        if np.isnan(col.data).any():
            self._logger.info(f"Column {col.name} contains NaN.")
        if np.isinf(col.data).any():
            self._logger.info(f"Column {col.name} contains infinite value.")

        entry = self._get_reference_data_column(col.name)
        if "allow_nan" in entry.get("input_processing", {}):
            return np.isnan(col.data).any() or np.isinf(col.data).any()

        if np.isnan(col.data).any() or np.isinf(col.data).any():
            self._logger.error("NaN or Inf values found in data columns")
            raise ValueError

        return False

    def _check_and_convert_units(self, col):
        """
        Check that all columns have an allowed unit. Convert to reference unit (e.g., Angstrom to
        nm).

        Note on dimensionless columns:

        - should be given in unit descriptor as unit: ''
        - be forgiving and assume that in cases no unit is given in the data files
          means that it should be dimensionless (e.g., for a efficiency)

        Parameters
        ----------
        col: astropy.column
            data column to be converted

        Returns
        -------
        astropy.column
            unit-converted data column

        Raises
        ------
        u.core.UnitConversionError
            If column unit conversions fails

        """

        self._logger.debug(f"Checking data column '{col.name}'")

        try:
            reference_unit = self._get_reference_unit(col.name)
            if col.unit is None or col.unit == "dimensionless":
                col.unit = u.dimensionless_unscaled
                return col

            self._logger.debug(
                f"Data column '{col.name}' with reference unit "
                f"'{reference_unit}' and data unit '{col.unit}'"
            )

            col.convert_unit_to(reference_unit)

        except u.core.UnitConversionError:
            self._logger.error(
                f"Invalid unit in data column '{col.name}'. "
                f"Expected type '{reference_unit}', found '{col.unit}'"
            )
            raise

        return col

    def _check_range(self, col_name, col_min, col_max, range_type="allowed_range"):
        """
        Check that column data is within allowed range or required range. Assumes that column and
        ranges have the same units.

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
        self._logger.debug(f"Checking data in column '{col_name}' for '{range_type}'")

        try:
            if range_type not in ("allowed_range", "required_range"):
                raise KeyError
        except KeyError:
            self._logger.error("Allowed range types are 'allowed_range', 'required_range'")
            raise

        _entry = self._get_reference_data_column(col_name)
        if range_type not in _entry:
            return None

        try:
            if not self._interval_check(
                (col_min, col_max),
                (_entry[range_type].get("min", np.NINF), _entry[range_type].get("max", np.Inf)),
                range_type,
            ):
                raise ValueError
        except KeyError:
            self._logger.error(f"Invalid range ('{range_type}') definition for column '{col_name}'")
        except ValueError:
            self._logger.error(
                f"Value for column '{col_name}' out of range. "
                f"([{col_min}, {col_max}], {range_type}: "
                f"[{_entry[range_type].get('min', np.NINF)}, "
                f"{_entry[range_type].get('max', np.Inf)}])"
            )
            raise

        return None

    @staticmethod
    def _interval_check(data, axis_range, range_type):
        """
        Check that values are inside allowed range (range_type='allowed_range') or span at least
         the given inveral (range_type='required_range').

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

    def _read_validation_schema(self, schema_file):
        """
        Read validation schema from file.
        Returns 'None' in case no schema file is given.

        Parameters
        ----------
        schema_file: Path
            Schema file describing input data

        Returns
        -------
        dict
           validation schema

        """

        _schema_dict = {}
        _data_dict = {}
        try:
            self._logger.info(f"Reading validation schema from {schema_file}")
            with open(schema_file, "r", encoding="utf-8") as stream:
                _schema_dict = yaml.safe_load(stream)
        except TypeError:
            return None
        except FileNotFoundError:
            self._logger.error(f"Schema file not found: {schema_file}")
            raise

        # Note - assume that schema files contain a single data / schema
        #        description (index [0]). This might change in future,
        #        and we keep for now the list definition.
        try:
            _data_dict = _schema_dict["schema"][0]["data"][0]
        except (KeyError, IndexError):
            self._logger.error(f"Error reading validation schema from {_schema_dict}")
            raise

        if _data_dict is None:
            self._logger.info("No validation schema found in {schema_file}")

        # TODO - returning table columns only; should also work for non-tabled description
        return _data_dict.get("table_columns", None)

    def _get_reference_data_column(self, column_name, status_test=False):
        """
        Return entry in reference data for a given column name.

        Parameters
        ----------
        column_name: str
            Column name.

        Returns
        -------
        dict
            Reference schema column.

        Raises
        ------
        IndexError
            If data column is not found.

        """

        _entry = [item for item in self._reference_data_columns if item["name"] == column_name]
        if status_test:
            return len(_entry) > 0
        try:
            return _entry[0]
        except IndexError:
            self._logger.error(
                f"Data column '{column_name}' not found in reference column definition"
            )
            raise
