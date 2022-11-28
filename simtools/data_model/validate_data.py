import logging
import os

from astropy import units as u
from astropy.table import Table, unique
from astropy.utils.diff import report_diff_values

__all__ = ["DataValidator"]


class DataValidator:
    """
    Simulation model data transformation for data validation. Validate input data for type and \
    units; converts or transform data if required.

    Parameters
    ----------
    workflow: WorkflowDescription
        workflow description.

    """

    def __init__(self, workflow=None):
        """
        Initalize validation class and read required reference data columns
        """

        self._logger = logging.getLogger(__name__)

        if workflow:
            self._reference_data_columns = workflow.reference_data_columns()
            self._data_file_name = workflow.input_data_file_name()

        self.data_table = None

    def validate(self):
        """
        Data and data file validation

        Returns
        -------
        data_table: astropy.table
            Data table

        """

        self.validate_data_file()
        self.validate_data_columns()

        return self.data_table

    def transform(self):
        """
        Apply transformations to data columns:
        - duplication removal
        - sorting according to axes

        Returns
        -------
        data_table: astropy.table
            Data table

        """

        self._check_data_for_duplicates()
        self._sort_data()

        return self.data_table

    def validate_data_file(self):
        """
        Open data file and read data from file
        """

        self._logger.info("Reading data from {}".format(self._data_file_name))
        self.data_table = Table.read(self._data_file_name, guess=True, delimiter=r"\s")

    def validate_data_columns(self):
        """
        Validate that required data columns are available, columns are in the correct units (if \
        necessary apply a unit conversion), and check ranges (minimum, maximum). This is not \
        applied to columns of type 'string'

        """

        self._check_required_columns()

        for col in self.data_table.itercols():
            if not self._column_status(col.name):
                continue
            self._check_and_convert_units(col)
            self._check_range(col.name, col.min(), col.max(), "allowed_range")
            self._check_range(col.name, col.min(), col.max(), "required_range")

    def _check_required_columns(self):
        """
        Check that all required data columns are available in the input data table

        Raises
        ------
        KeyError
            if a required data column is missing

        """

        for key, value in self._reference_data_columns.items():
            if "required_column" in value and value["required_column"] is True:
                if key in self.data_table.columns:
                    self._logger.debug("Found required data column '{}'".format(key))
                else:
                    raise KeyError("Missing required column '{}'".format(key))

    def _sort_data(self):
        """
        Sort data according to one data column (if required by any column attribute). Data is \
         either sorted or reverse sorted

        Raises
        ------
        AttributeError
            if no table is defined for sorting

        """

        _columns_by_which_to_sort = []
        _columns_by_which_to_reverse_sort = []
        for key, value in self._reference_data_columns.items():
            if "attribute" in value:
                if "sort" in value["attribute"]:
                    _columns_by_which_to_sort.append(key)
                elif "reversesort" in value["attribute"]:
                    _columns_by_which_to_reverse_sort.append(key)

        if len(_columns_by_which_to_sort) > 0:
            self._logger.debug("Sorting data columns: {}".format(_columns_by_which_to_sort))
            try:
                self.data_table.sort(_columns_by_which_to_sort)
            except AttributeError:
                self._logger.error("No data table defined for sorting")
                raise
        elif len(_columns_by_which_to_reverse_sort) > 0:
            self._logger.debug(
                "Reverse sorting data columns: {}".format(_columns_by_which_to_reverse_sort)
            )
            try:
                self.data_table.sort(_columns_by_which_to_reverse_sort, reverse=True)
            except AttributeError:
                self._logger.error("No data table defined for sorting")
                raise

    def _check_data_for_duplicates(self):
        """
        Remove duplicates from data columns as defined in the data columns description

        Raises
        ------
            if row values are different for those rows with duplications in the data columns to be \
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
        with open(os.devnull, "w") as devnull:
            if report_diff_values(
                _data_table_unique_for_key_column,
                _data_table_unique_for_all_columns,
                fileobj=devnull,
            ):
                self.data_table = unique(self.data_table)
            else:
                self._logger.error(
                    "Failed removal of duplication for column {}, values are not unqiue".format(
                        _column_with_unique_requirement
                    )
                )
                raise ValueError

    def _get_unique_column_requirement(self):
        """
        Return data column name with unique value requirement

        Returns
        -------
        list
            list of data column with unique value requirement

        """

        _unique_required_column = []

        for key, value in self._reference_data_columns.items():
            if "attribute" in value and "remove_duplicates" in value["attribute"]:
                self._logger.debug("Removing duplicates for column '{}'".format(key))
                _unique_required_column.append(key)

        self._logger.debug("Unique required columns: {}".format(_unique_required_column))
        return _unique_required_column

    def _get_reference_unit(self, column_name):
        """
        Return reference column unit. Includes correct treatment of dimensionless units

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

        try:
            reference_unit = self._reference_data_columns[column_name]["unit"]
        except KeyError:
            self._logger.error(
                "Data column '{}' not found in reference column definition".format(column_name)
            )
            raise

        if reference_unit == "dimensionless" or reference_unit is None:
            return u.dimensionless_unscaled

        return u.Unit(reference_unit)

    def _column_status(self, col_name):
        """
        Check that column is defined in reference schema (additional data columns are allowed in\
        the input data, but are ignored) and that column type is not string (string-type columns\
        ignored for range checks)

        """

        if col_name not in self._reference_data_columns:
            return False
        if (
            "type" in self._reference_data_columns[col_name]
            and self._reference_data_columns[col_name]["type"] == "string"
        ):
            return False

        return True

    def _check_and_convert_units(self, col):
        """
        Check that all columns have an allowed units. Convert to reference unit (e.g., Angstrom to\
         nm).

        Note on dimensionless columns:

        - should be given in unit descriptor as unit: ''
        - be forgiving and assume that in cases no unit is given in the data files\
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

        self._logger.debug("Checking data column '{}'".format(col.name))

        try:
            reference_unit = self._get_reference_unit(col.name)
            if col.unit is None or col.unit == "dimensionless":
                col.unit = u.dimensionless_unscaled
                return col

            self._logger.debug(
                "Data column '{}' with reference unit '{}' and data unit '{}'".format(
                    col.name, reference_unit, col.unit
                )
            )

            col.convert_unit_to(reference_unit)

        except u.core.UnitConversionError:
            self._logger.error(
                "Invalid unit in data column '{}'. Expected type '{}', found '{}'".format(
                    col.name, reference_unit, col.unit
                )
            )
            raise

        return col

    def _check_range(self, col_name, col_min, col_max, range_type="allowed_range"):
        """
        Check that column data is within allowed range or required range. Assumes that column and \
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
        self._logger.debug("Checking data in column '{}' for '{}'".format(col_name, range_type))

        try:
            if range_type not in ("allowed_range", "required_range"):
                raise KeyError
            if range_type not in self._reference_data_columns[col_name]:
                return None
        except KeyError as e:
            raise KeyError(f"Missing column '{col_name} in reference range'") from e

        try:
            if not self._interval_check(
                (col_min, col_max),
                (
                    self._reference_data_columns[col_name][range_type]["min"],
                    self._reference_data_columns[col_name][range_type]["max"],
                ),
                range_type,
            ):
                raise ValueError
        except KeyError:
            self._logger.error(
                "Invalid range ('{}') definition for column '{}'".format(range_type, col_name)
            )
        except ValueError:
            self._logger.error(
                "Value for column '{}' out of range. [[{}, {}], {}: [[{}, {}]".format(
                    col_name,
                    col_min,
                    col_max,
                    range_type,
                    self._reference_data_columns[col_name][range_type]["min"],
                    self._reference_data_columns[col_name][range_type]["max"],
                )
            )
            raise

        return None

    @staticmethod
    def _interval_check(data, axis_range, range_type):
        """
        Check that values are inside allowed range (range_type='allowed_range') or span at least \
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
