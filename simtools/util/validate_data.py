import logging

from astropy.table import Table
from astropy import units as u


class DataValidator:
    """
    Simulation model data transformation for data validation.

    Attributes
    ----------
    reference_data_columns: dict
        data columns description
    data_file_name: str
        name of input data file


    Methods
    -------
    validate_and_transform()
        Main function to validate and transform.
    validate_data_file()
        Open data file and check for file consistency
    validate_data_columns()
        Check each data column for correct units and data ranges.

    """

    def __init__(self, reference_data_columns, data_file_name):
        """
        Initalize validation class and read required
        reference data columns

        """

        self._logger = logging.getLogger(__name__)

        self._reference_data_columns = reference_data_columns
        self._data_file_name = data_file_name

        self.data_table = None

    def validate_and_transform(self):
        """
        Data validation and coordination transformation

        """

        self.validate_data_file()
        self.validate_data_columns()

        return self.data_table

    def validate_data_file(self):
        """
        Open data file and check for file consistency

        """

        # TODO
        # understand how errors should be treated when
        # opening and reading fails.
        # FileNotFoundError
        # astropy.io.ascii.core.InconsistentTableError
        self.data_table = Table.read(self._data_file_name, guess=True)
        self._logger.info("Reading data from {}".format(self._data_file_name))

    def validate_data_columns(self):
        """
        Validate that required data columns are available
        and in the correct units

        """

        self._check_required_columns()

        for col in self.data_table.itercols():
            self._check_and_convert_units(col)
            self._check_range(col.name, col.min(), col.max(), 'allowed_range')
            self._check_range(col.name, col.min(), col.max(), 'required_range')

    def _check_required_columns(self):
        """
        Check that all required data columns are available
        in the input data table

        """

        for key, value in self._reference_data_columns.items():
            if 'required_column' in value and value['required_column'] is True:
                try:
                    self.data_table.columns[key]
                    self._logger.debug("Found required data column '{}'".format(key))
                except KeyError:
                    self._logger.error("Missing required column '{}'".format(key))
                    raise

    def _get_reference_unit(self, key):
        """
        Return reference column unit.
        Includes correct treatment of dimensionless units

        """

        try:
            reference_unit = self._reference_data_columns[key]['unit']
        except KeyError:
            self._logger.error(
                "Data column '{}' not found in reference column definition".format(
                    key))
            raise

        if reference_unit == 'dimensionless' or reference_unit is None:
            return u.dimensionless_unscaled

        return u.Unit(reference_unit)

    def _check_and_convert_units(self, col):
        """
        Check that all columns have an allowed units.
        Convert to reference unit (e.g., Angstrom to nm).

        Note on dimensionless columns:
            - should be given in unit descriptor as unit: ''
        - be forgiving and assume that in cases no unit is given in the data files
          means that it should be dimensionless (e.g., for a efficiency)

        """

        self._logger.debug("Checking data column '{}'".format(col.name))

        try:
            reference_unit = self._get_reference_unit(col.name)
            if col.unit is None or col.unit == 'dimensionless':
                col.unit = u.dimensionless_unscaled

            self._logger.debug(
                "Data column '{}' with reference unit '{}' and data unit '{}'".format(
                    col.name, reference_unit, col.unit))

            col.convert_unit_to(reference_unit)

        except u.core.UnitConversionError:
            self._logger.error(
                "Invalid unit in data column '{}'. Expected type '{}', found '{}'".format(
                    col.name, reference_unit, col.unit))
            raise

        return col

    def _interval_check(self, data_min, data_max, axis_min, axis_max, range_type):
        """
        Check that values are inside allowed range (range_type='allowed_range')
        or span at least the given inveral (range_type='required_range').

        """

        if range_type == 'allowed_range' and \
           data_min >= axis_min and data_max <= axis_max:
            return True
        elif range_type == 'required_range' and \
           data_min <= axis_min and data_max >= axis_max:
            return True

        return False

    def _check_range(self,
                     col_name,
                     col_min, col_max,
                     range_type='allowed_range'):
        """
        Check that column data is within allowed range
        or required range.

        Assume that column and ranges have the same units
        """
        self._logger.debug(
            "Checking data in column '{}' for '{}'".format(
                col_name, range_type))

        try:
            if range_type != 'allowed_range' and range_type != 'required_range':
                raise KeyError
            if range_type not in self._reference_data_columns[col_name]:
                return None
        except KeyError:
            raise KeyError(
                f"Missing column '{col_name} in reference range'")

        try:
            if not self._interval_check(
                    col_min, col_max,
                    self._reference_data_columns[col_name][range_type]['min'],
                    self._reference_data_columns[col_name][range_type]['max'],
                    range_type):
                raise ValueError
        except KeyError:
            self._logger.error(
                "Invalid range ('{}') definition for column '{}'".format(
                    range_type, col_name))
        except ValueError:
            self._logger.error(
                "Value for column '{}' out of range. [[{}, {}], {}: [[{}, {}]".format(
                    col_name,
                    col_min, col_max,
                    range_type,
                    self._reference_data_columns[col_name][range_type]['min'],
                    self._reference_data_columns[col_name][range_type]['max']))
            raise

        return None
