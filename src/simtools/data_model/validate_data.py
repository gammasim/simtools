"""Validation of data using schema (types, ranges, allowed values and required columns)."""

import logging
import os
import re
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Column, Table, unique
from astropy.utils.diff import report_diff_values
from jsonschema import Draft202012Validator

from simtools.data_model import schema, schema_loader
from simtools.io import ascii_handler
from simtools.utils import names, value_conversion
from simtools.version import is_valid_semantic_version

_JSON_TYPE_BY_KIND = {
    "b": "boolean",
    "i": "integer",
    "u": "integer",
    "f": "number",
    "U": "string",
    "S": "string",
    "O": "object",
}
_STRING_TYPES = ("string", "str", "file")
_BOOLEAN_TYPES = ("boolean", "bool")


def _errors(instance, json_schema):
    """Return jsonschema error messages (empty list if valid)."""
    return [error.message for error in Draft202012Validator(json_schema).iter_errors(instance)]


def _type_schema(reference_type, exact):
    """Return the jsonschema for a reference data type (instance: json_type, dtype, value)."""
    if exact and reference_type not in (*_STRING_TYPES, "dict", "list"):
        name = "bool" if reference_type in _BOOLEAN_TYPES else np.dtype(reference_type).name
        return {"properties": {"dtype": {"const": name}}}
    if reference_type in _BOOLEAN_TYPES:  # 0 and 1 are accepted as booleans
        return {
            "anyOf": [
                {"properties": {"json_type": {"const": "boolean"}}},
                {"properties": {"value": {"enum": [0, 1]}}, "required": ["value"]},
            ]
        }
    if reference_type in _STRING_TYPES:
        json_types = ["string", "object", "null"]
    elif reference_type in ("dict", "list"):
        json_types = ["object" if reference_type == "dict" else "array"]
    elif np.issubdtype(reference_type, np.integer):
        json_types = ["integer"]
    else:  # floating point references also accept integers
        json_types = ["integer", "number"]
    return {"properties": {"json_type": {"enum": json_types}}}


def _type_matches(reference_type, value=None, dtype=None, exact=False):
    """Return True if a value (or dtype) matches the reference data type."""
    dtype = np.asarray(value).dtype if dtype is None else np.dtype(dtype)
    if isinstance(value, dict | list):
        json_type = "object" if isinstance(value, dict) else "array"
    elif value is None and dtype.kind == "O":
        json_type = "null"
    else:
        json_type = _JSON_TYPE_BY_KIND.get(dtype.kind)
    instance = {"json_type": json_type, "dtype": dtype.name, "value": value}
    return Draft202012Validator(_type_schema(reference_type, exact)).is_valid(instance)


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
        """Initialize validation class."""
        self.logger = logging.getLogger(__name__)
        self.data_file_name = data_file
        self.schema_file_name = schema_file
        self._data_description = None
        self.data_dict = data_dict
        self.data_table = data_table
        self.check_exact_data_type = check_exact_data_type

    def validate_and_transform(self, is_model_parameter=False, lists_as_strings=False):
        """
        Validate data (and data file) and return the validated dict or table.

        Parameters
        ----------
        is_model_parameter: bool
            This is a model parameter (add some data preparation).
        lists_as_strings: bool
            Convert lists to strings (as needed for model parameters).

        Raises
        ------
        TypeError
            if no data or data table is available.
        """
        if self.data_file_name:
            self.validate_data_file(is_model_parameter)
        if isinstance(self.data_dict, dict):
            return self._validate_data_dict(is_model_parameter, lists_as_strings)
        if isinstance(self.data_table, Table):
            return self._validate_data_table()
        self.logger.error("No data or data table to validate")
        raise TypeError

    def validate_data_file(self, is_model_parameter=None):
        """Read the data file (successful reading counts as file validation)."""
        try:
            if Path(self.data_file_name).suffix in (".yml", ".yaml", ".json"):
                self.data_dict = ascii_handler.collect_data_from_file(self.data_file_name)
                self.logger.info(f"Validating data from: {self.data_file_name}")
            else:
                self.data_table = Table.read(self.data_file_name, guess=True, delimiter=r"\s")
                self.logger.info(f"Validating tabled data from: {self.data_file_name}")
        except AttributeError, TypeError:
            pass
        if is_model_parameter:
            self.validate_parameter_and_file_name()

    def validate_parameter_and_file_name(self):
        """Check that the file name starts with the parameter and ends with its version."""
        file_stem = Path(self.data_file_name).stem
        param = self.data_dict.get("parameter")
        param_version = self.data_dict.get("parameter_version")
        if not file_stem.startswith(param):
            raise ValueError(f"Mismatch: parameter '{param}' vs. file '{file_stem}'.")
        if param_version and not file_stem.endswith(param_version):
            raise ValueError(f"Mismatch: version '{param_version}' vs. file '{file_stem}'.")
        if param_version is None:
            self.logger.warning(f"File '{file_stem}' has no parameter version defined.")

    @staticmethod
    def validate_model_parameter(par_dict, par_name=None):
        """
        Validate a simulation model parameter.

        Parameters
        ----------
        par_dict: dict
            Data dictionary.
        par_name: str
            Parameter name (default: 'parameter' entry of par_dict).

        Returns
        -------
        dict
            Validated data dictionary.
        """
        par_name = par_dict.get("parameter") if par_name is None else par_name
        data_validator = DataValidator(
            schema_file=schema.get_model_parameter_schema_file(f"{par_name}"),
            data_dict=par_dict,
            check_exact_data_type=False,
        )
        return data_validator.validate_and_transform(is_model_parameter=True)

    @staticmethod
    def validate_data_files(
        file_directory=None,
        file_name=None,
        is_model_parameter=True,
        check_exact_data_type=False,
        schema_file=None,
    ):
        """
        Validate data or model parameters in files in a directory or a single file.

        Parameters
        ----------
        file_directory: str or Path
            Directory with files to be validated (JSON files, searched recursively).
        file_name: str or Path
            Name of the file to be validated.
        is_model_parameter: bool
            This is a model parameter (add some data preparation).
        check_exact_data_type: bool
            Require exact data type for validation.
        schema_file: str or Path
            Schema file used for all files (default: from parameter name or file metadata).
        """
        if file_directory:
            file_list = sorted(Path(file_directory).rglob("*.json"))
        elif file_name:
            file_list = [Path(file_name)]
        else:
            return

        for data_file in file_list:
            parameter_name = re.sub(r"-\d+\.\d+\.\d+", "", data_file.stem)
            schema_path = schema_file or (
                schema.get_model_parameter_schema_file(parameter_name)
                if is_model_parameter
                else schema.get_schema_file_from_file_metadata(data_file)
                or schema.get_model_parameter_schema_file(parameter_name)
            )
            data_validator = DataValidator(
                schema_file=schema_path,
                data_file=data_file,
                check_exact_data_type=check_exact_data_type,
            )
            data_validator.validate_and_transform(is_model_parameter)
            data_validator.logger.info(f"Validated data file {data_file} with schema {schema_path}")

    def _validate_data_dict(self, is_model_parameter=False, lists_as_strings=False):
        """Validate values in a dictionary (names from 'name' or 'parameter' keys)."""
        if is_model_parameter:
            self._prepare_model_parameter()
        # use 0.1.0 as default; this corresponds to the first definition of the schema
        self._data_description = self._read_validation_schema(
            self.schema_file_name, self.data_dict.get("model_parameter_schema_version", "0.1.0")
        )
        self._validate_and_assign_values(is_model_parameter)

        if self.data_dict.get("instrument"):
            if self.data_dict["instrument"] == self.data_dict["site"]:  # site parameters
                names.validate_site_name(self.data_dict["site"])
            else:
                names.validate_array_element_name(self.data_dict["instrument"])
                self._check_site_and_array_element_consistency(
                    self.data_dict.get("instrument"), self.data_dict.get("site")
                )

        for version_type in ("version", "parameter_version", "model_version"):
            version_string = self.data_dict.get(version_type, "0.0.0")
            if not is_valid_semantic_version(version_string):
                raise ValueError(f"Invalid version string '{version_string}'")

        if lists_as_strings:
            self._convert_results_to_model_format()
        return self.data_dict

    def _validate_and_assign_values(self, is_model_parameter):
        """Validate and assign values/units for homogeneous and heterogeneous lists."""
        value_as_list, unit_as_list = self._get_value_and_units_as_lists()
        per_element = self._uses_per_element_type_validation(is_model_parameter)
        kind = "heterogeneous list" if per_element else "dictionary"
        expected_types = self._get_expected_types_for_values(value_as_list, is_model_parameter)
        try:
            self._validate_list_types(value_as_list, expected_types)
        except (TypeError, ValueError) as ex:
            raise TypeError(f"Error validating {kind} using {self.schema_file_name}") from ex

        for index, (value, unit) in enumerate(zip(value_as_list, unit_as_list)):
            try:
                value_as_list[index], unit_as_list[index] = self._validate_value_and_unit(
                    value, unit, index
                )
            except TypeError as ex:
                kind = f"heterogeneous element {index}" if per_element else "dictionary"
                raise TypeError(f"Error validating {kind} using {self.schema_file_name}") from ex

        self._assign_validated_values(value_as_list, unit_as_list)

    @staticmethod
    def _validate_list_types(value_list, expected_types):
        """Validate list elements against per-element expected data types."""
        if not expected_types:
            return
        if len(value_list) != len(expected_types):
            raise ValueError(
                f"Length mismatch: parameter has {len(value_list)} elements "
                f"but type specification has {len(expected_types)} elements."
            )
        for i, (value, expected_type) in enumerate(zip(value_list, expected_types)):
            if not _type_matches(expected_type, value=value):
                raise TypeError(
                    f"Element {i}: expected type '{expected_type}', got {type(value).__name__}"
                )

    def _get_expected_types_for_values(self, value_as_list, is_model_parameter):
        """Return expected per-element data types (empty if no list validation applies)."""
        if not (
            is_model_parameter
            and self._data_description
            and isinstance(value_as_list, list)
            and value_as_list
            and value_as_list[0] is not None
        ):
            return []
        if self._uses_per_element_type_validation(is_model_parameter):
            return self._get_schema_type_list()
        expected_type = self._data_description[0].get("type")
        if len(value_as_list) > 1 and expected_type and not isinstance(expected_type, list):
            return [expected_type] * len(value_as_list)
        return []

    def _uses_per_element_type_validation(self, is_model_parameter=False):
        """Return True if an explicit type list (positional types) applies to a model parameter."""
        if not is_model_parameter or not isinstance(self._data_description, list):
            return False
        if isinstance(self.data_dict.get("type"), list):
            return True
        return len(set(self._get_schema_type_list())) > 1

    def _get_schema_type_list(self):
        """Return the schema types as a flat list."""
        if not self._data_description:
            return []
        first_type = self._data_description[0].get("type")
        return (
            first_type
            if isinstance(first_type, list)
            else [e.get("type") for e in self._data_description]
        )

    def _assign_validated_values(self, value_as_list, unit_as_list):
        """Assign validated value/unit; a list value of a 'dict' parameter stays a list."""
        is_json_array = self._get_data_description(0).get("type") == "dict" and isinstance(
            self.data_dict["value"], list
        )
        if is_json_array:
            self.data_dict["value"] = value_as_list
            self.data_dict["unit"] = unit_as_list[0] if unit_as_list else self.data_dict["unit"]
        elif len(value_as_list) == 1:
            self.data_dict["value"], self.data_dict["unit"] = value_as_list[0], unit_as_list[0]
        else:
            self.data_dict["value"], self.data_dict["unit"] = value_as_list, unit_as_list

    def _validate_value_and_unit(self, value, unit, index):
        """Validate type, allowed values, NaN and ranges of one value and convert its unit."""
        entry = self._get_data_description(index)
        if entry.get("type") == "dict":
            schema.validate_dict_using_schema(
                data=self.data_dict["value"], json_schema=entry.get("json_schema")
            )
        else:
            self._check_data_type(np.array(value).dtype, index, value)

        if entry.get("type") not in ("string", "dict", "file"):
            self._check_for_not_a_number(value, index)
            value, unit = self._check_and_convert_units(value, unit, index)
            for range_type in ("allowed_range", "required_range"):
                self._check_range(index, np.nanmin(value), np.nanmax(value), range_type)
        return value, unit

    def _get_value_and_units_as_lists(self):
        """Return value and unit as lists (value scaled to the target unit, if required)."""
        target_unit = self.data_dict["unit"]
        value, unit = value_conversion.split_value_and_unit(self.data_dict["value"])
        if not isinstance(value, list | np.ndarray):
            value, unit = [value], [unit]
        if not isinstance(target_unit, list | np.ndarray):
            target_unit = [target_unit] * len(value)

        target_unit = value_conversion.normalize_dimensionless_unit(target_unit)
        conversion_factor = [
            1 if v is None else u.Unit(v).to(u.Unit(t)) for v, t in zip(unit, target_unit)
        ]
        try:
            return [
                v * c if not isinstance(v, bool | dict) else v
                for v, c in zip(value, conversion_factor)
            ], target_unit
        except TypeError:
            return [None], target_unit

    def _validate_data_table(self):
        """Validate tabulated data."""
        try:
            self._data_description = self._read_validation_schema(self.schema_file_name)[0].get(
                "table_columns", None
            )
        except IndexError:
            self.logger.error(f"Error reading validation schema from {self.schema_file_name}")
            raise

        if self._data_description is not None:
            self._validate_data_columns()
            self._check_data_for_duplicates()
            self._sort_data()
        return self.data_table

    def _validate_data_columns(self):
        """Validate required columns, allowed values, types, NaN, units and ranges."""
        self._check_required_columns()

        for col_name in self.data_table.colnames:
            col = self.data_table[col_name]
            if not self._get_data_description(col_name, status_test=True):
                continue
            if not np.issubdtype(col.dtype, np.number):
                continue
            self._check_for_not_a_number(col.data, col_name)
            self._check_data_type(col.dtype, col_name)
            converted_col, _ = self._check_and_convert_units(
                col, unit=getattr(col, "unit", None), col_name=col_name
            )
            self.data_table[col_name] = converted_col
            if len(converted_col) == 0:
                if self._get_data_description(col_name).get("required_range"):
                    raise ValueError(f"Empty column '{col_name}' cannot cover its required range.")
                continue
            for range_type in ("allowed_range", "required_range"):
                self._check_range(
                    col_name,
                    np.nanmin(converted_col.data),
                    np.nanmax(converted_col.data),
                    range_type,
                )

    def _check_required_columns(self):
        """Check that all required data columns are available (KeyError otherwise)."""
        required = [entry["name"] for entry in self._data_description if entry.get("required")]
        columns = dict.fromkeys(self.data_table.colnames)
        if not Draft202012Validator({"required": required}).is_valid(columns):
            missing = [name for name in required if name not in columns]
            raise KeyError(f"Missing required column {', '.join(missing)}")

    def _sort_data(self):
        """Sort the table by columns with 'sort' (or reverse sort with 'reversesort')."""
        ascending = [
            e["name"] for e in self._data_description if "sort" in e.get("input_processing", [])
        ]
        descending = [
            e["name"]
            for e in self._data_description
            if "reversesort" in e.get("input_processing", [])
        ]
        if not (ascending or descending):
            return
        try:
            self.data_table.sort(ascending or descending, reverse=not ascending)
        except AttributeError:
            self.logger.error("No data table defined for sorting")
            raise

    def _check_data_for_duplicates(self):
        """Remove duplicates of the 'remove_duplicates' columns (ValueError if they conflict)."""
        keys = [
            e["name"]
            for e in self._data_description
            if "remove_duplicates" in e.get("input_processing", [])
        ]
        if not keys:
            return
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            if report_diff_values(
                unique(self.data_table, keys=keys), unique(self.data_table), fileobj=devnull
            ):
                self.data_table = unique(self.data_table)
            else:
                raise ValueError(f"Failed removal of duplication for column {keys}, values differ")

    def _get_data_description(self, column_name=None, status_test=False):
        """Return the schema entry of a column (name, 'colN' or index); IndexError if missing."""
        description = self._data_description
        if isinstance(column_name, int):
            if column_name < len(description):
                entry = description[column_name]
            else:
                entry = description[0] if len(description) == 1 else None
        elif re.fullmatch(r"col\d", column_name) and int(column_name[3:]) < len(description):
            entry = description[int(column_name[3:])]
        else:
            entry = next((item for item in description if item["name"] == column_name), None)
        if status_test:
            return entry is not None
        if entry is None:
            self.logger.error(
                f"Data column '{column_name}' not found in reference column definition"
            )
            raise IndexError(f"Data column '{column_name}' not found")
        return entry

    def _check_data_type(self, dtype, column_name, value=None):
        """Check the data type of a column against the schema (TypeError if not matching)."""
        reference_type = self._get_data_description(column_name).get("type")
        if not _type_matches(
            reference_type, value=value, dtype=dtype, exact=self.check_exact_data_type
        ):
            self.logger.error(
                f"Invalid data type in column '{column_name}'. "
                f"Expected type '{reference_type}', found '{dtype}' "
                f"(exact type: {self.check_exact_data_type})"
            )
            raise TypeError

    def _check_for_not_a_number(self, data, col_name):
        """Check for NaN or Inf values (ValueError unless 'allow_nan'); True if present."""
        if isinstance(data, str):
            return True
        data = data if isinstance(data, np.ndarray) else np.array(data)
        has_nan, has_inf = bool(np.isnan(data).any()), bool(np.isinf(data).any())
        if has_nan:
            self.logger.info(f"Column {col_name} contains NaN.")
        if has_inf:
            self.logger.info(f"Column {col_name} contains infinite value.")
        if "allow_nan" in self._get_data_description(col_name).get("input_processing", {}):
            return has_nan or has_inf
        if has_nan or has_inf:
            raise ValueError("NaN or Inf values found in data")
        return False

    def _check_and_convert_units(self, data, unit, col_name):
        """
        Convert data to the reference unit of the column.

        Returns the converted data and the reference unit (None if dimensionless).
        Raises u.core.UnitConversionError if the conversion fails.
        """
        ref = self._get_data_description(col_name).get("unit", None)
        reference_unit = value_conversion.normalize_dimensionless_unit(
            (
                u.dimensionless_unscaled
                if value_conversion.is_dimensionless_unit(ref)
                else u.Unit(ref)
            ).to_string()
        )
        target = reference_unit or u.dimensionless_unscaled
        column_unit = value_conversion.normalize_dimensionless_unit(getattr(data, "unit", unit))
        if value_conversion.is_dimensionless_unit(column_unit) and (
            value_conversion.is_dimensionless_unit(reference_unit)
        ):
            return data, None
        try:
            if isinstance(data, u.Quantity | Column):
                return data.to(target), reference_unit
            if isinstance(data, list | np.ndarray):
                return [
                    (
                        d
                        if value_conversion.is_dimensionless_unit(to_unit)
                        else u.Unit(to_unit).to(target) * d
                    )
                    for d, to_unit in zip(data, column_unit)
                ], reference_unit
            # keep the data type (e.g., integers)
            conversion_unit = column_unit or u.dimensionless_unscaled
            return type(data)(u.Unit(conversion_unit).to(target) * data), reference_unit
        except (u.core.UnitConversionError, ValueError) as exc:
            self.logger.error(
                f"Invalid unit in data column '{col_name}'. "
                f"Expected type '{reference_unit}', found '{column_unit}'"
            )
            raise u.core.UnitConversionError(str(exc)) from exc

    def _check_range(self, col_name, col_min, col_max, range_type="allowed_range"):
        """
        Check the data minimum and maximum against the allowed or required range.

        The allowed range must contain the data; the required range must be spanned by it.
        Raises ValueError if the check fails.
        """
        bounds = self._get_data_description(col_name).get(range_type)
        if bounds is None:
            return
        low, high = (
            ("minimum", "maximum") if range_type == "allowed_range" else ("maximum", "minimum")
        )
        properties = {}
        if "min" in bounds:
            properties["min"] = {low: bounds["min"]}
        if "max" in bounds:
            properties["max"] = {high: bounds["max"]}
        if _errors({"min": col_min, "max": col_max}, {"properties": properties}):
            raise ValueError(
                f"Value for column '{col_name}' out of range. "
                f"([{col_min}, {col_max}], {range_type}: "
                f"[{bounds.get('min', -np.inf)}, {bounds.get('max', np.inf)}])"
            )

    def _read_validation_schema(self, schema_file, schema_version=None):
        """Read the validation schema (first document, or the given schema version)."""
        if schema_file is None:
            raise TypeError("No validation schema file provided")
        schema_data = schema_loader.load_schema(schema_file, schema_version or "latest")
        if schema_version and schema_data.get("schema_version") != schema_version:
            raise ValueError(
                f"Schema version {schema_version} not found in schema file {schema_file}."
            )
        try:
            return schema_data["data"]
        except KeyError as exc:
            raise KeyError(f"Error reading validation schema from {schema_file}") from exc

    def _prepare_model_parameter(self):
        """Convert string values to numbers and string units to lists (model parameters)."""
        self.data_dict["value"], _ = value_conversion.split_value_and_unit(
            self.data_dict["value"], "int" in self.data_dict.get("type", "float")
        )
        if isinstance(self.data_dict["unit"], str):
            normalized = value_conversion.normalize_dimensionless_unit(
                self.data_dict["unit"].strip()
            )
            self.data_dict["unit"] = (
                None
                if normalized is None
                else value_conversion.convert_string_to_list(
                    normalized, force_comma_separation=True
                )
            )

    def _convert_results_to_model_format(self):
        """Convert lists to strings (as needed for model parameters)."""
        for key in ("value", "unit"):
            if isinstance(self.data_dict[key], list):
                self.data_dict[key] = value_conversion.convert_list_to_string(self.data_dict[key])

    def _check_site_and_array_element_consistency(self, instrument, site):
        """Check that site and array element names agree (e.g., 'LSTN' is not at 'South')."""
        if not (instrument and site):
            return
        instruments = [instrument] if isinstance(instrument, str) else instrument
        if any(inst.startswith("OBS") for inst in instruments):
            return
        instrument_sites = set()
        for inst in instruments:
            found = names.get_site_from_array_element_name(inst)
            instrument_sites.update(found if isinstance(found, list) else [found])
        site_list = [site] if isinstance(site, str) else sorted(site)
        if sorted(instrument_sites) != site_list:
            raise ValueError(
                f"Site '{site_list}' and instrument site "
                f"'{sorted(instrument_sites)}' are inconsistent."
            )
