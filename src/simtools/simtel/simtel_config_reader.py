#!/usr/bin/python3
"""Read model parameters and configuration from sim_telarray configuration files."""

import logging
import re

import astropy.units as u
import numpy as np

from simtools.io import ascii_handler
from simtools.utils import names


def get_list_of_simtel_parameters(simtel_config_file):
    """
    Return list of sim_telarray parameters found in sim_telarray configuration file.

    Parameters
    ----------
    simtel_config_file: str
        File name for sim_telarray configuration

    Returns
    -------
    list
        List of parameters found in sim_telarray configuration file.

    """
    simtel_parameter_set = set()
    with open(simtel_config_file, encoding="utf-8") as file:
        for line in file:
            parts_of_lines = re.split(r",\s*|\s+", line.strip())
            simtel_parameter_set.add(parts_of_lines[1].lower())
    return sorted(simtel_parameter_set)


class SimtelConfigReader:
    """
    Reads model parameters from configuration files and converts to the simtools representation.

    The output format are simtool-db-style json dicts.
    Model parameters are read from sim_telarray configuration files.
    The sim_telarray configuration can be generated using e.g., the following sim_telarray command:

    ... code-block:: console

        sim_telarray/bin/sim_telarray \
            -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal\
            -C typelist=no-internal -C maximum_telescopes=30\
            -DNSB_AUTOSCALE -DNECTARCAM -DHYPER_LAYOUT\
            -DNUM_TELESCOPES=30 /dev/null 2>|/dev/null | grep '(@cfg)' | sed 's/^(@cfg)

    Parameters
    ----------
    schema_file: str
        Schema file describing the model parameter.
    simtel_config_file: str or Path
        Path of the file to read from.
    simtel_telescope_name: str
        Telescope name (sim_telarray convention)
    parameter_name: str
        Parameter name (default: read from schema file)
    camera_pixels: int
        Number of camera pixels
    """

    def __init__(
        self,
        schema_file=None,
        simtel_config_file=None,
        simtel_telescope_name=None,
        parameter_name=None,
        camera_pixels=None,
    ):
        """Initialize SimtelConfigReader."""
        self._logger = logging.getLogger(__name__)

        self.schema_file = schema_file
        self.schema_dict = (
            ascii_handler.collect_data_from_file(file_name=self.schema_file, yaml_document=0)
            if self.schema_file is not None
            else None
        )
        self.parameter_name = self.schema_dict.get("name") if self.schema_dict else parameter_name
        try:
            self.simtel_parameter_name = names.get_simulation_software_name_from_parameter_name(
                self.parameter_name
            ).upper()
        except (KeyError, AttributeError):
            self.simtel_parameter_name = self.parameter_name.upper() if parameter_name else None
        self.simtel_telescope_name = simtel_telescope_name
        self.camera_pixels = camera_pixels
        self.parameter_dict = (
            self.read_simtel_config_file(simtel_config_file, simtel_telescope_name)
            if simtel_config_file
            else {}
        )

    def _should_skip_limits_check(self, data_type):
        """Check if limits should be skipped."""
        return data_type == "limits" and self.parameter_dict.get("type") == "bool"

    def _get_schema_values(self, data_type):
        """Check schema values for limits, unit, and default."""
        try:
            if data_type == "limits":
                _from_schema = [
                    self.schema_dict["data"][0]["allowed_range"].get("min"),
                    self.schema_dict["data"][0]["allowed_range"].get("max"),
                ]
                return _from_schema[0] if _from_schema[1] is None else _from_schema
            if len(self.schema_dict["data"]) == 1:
                return self.schema_dict["data"][0].get(data_type)
            return [data.get(data_type) for data in self.schema_dict["data"]]
        except (KeyError, IndexError):
            return None

    @staticmethod
    def _values_match(_from_simtel, _from_schema):
        """
        Check if values match (are close for floats).

        Convert where necessary astropy.Quantity to float.

        """
        if isinstance(_from_simtel, u.Quantity):
            _from_simtel = _from_simtel.value
        if isinstance(_from_simtel, np.ndarray) and len(_from_simtel) > 0:
            _from_simtel = np.array(
                [v.value if isinstance(v, u.Quantity) else v for v in _from_simtel]
            )
        try:
            if not isinstance(_from_schema, list | np.ndarray) and _from_simtel == _from_schema:
                return True
        except ValueError:
            pass

        try:
            if np.all(np.isclose(_from_simtel, _from_schema)):
                return True
        except (TypeError, ValueError):
            pass

        return False

    def _log_mismatch_warning(self, data_type, _from_simtel, _from_schema):
        """Log mismatch warning."""
        self._logger.warning(f"Values for {data_type} do not match:")
        self._logger.warning(
            f"  from simtel: {self.simtel_parameter_name} {_from_simtel} ({type(_from_simtel)})"
        )
        self._logger.warning(
            f"  from schema: {self.parameter_name} {_from_schema} ({type(_from_schema)})"
        )

    def compare_simtel_config_with_schema(self):
        """
        Compare limits and defaults reported by sim_telarray with schema.

        This is mostly for debugging purposes and includes simple printing.
        Check for differences in 'default' and 'limits' entries.
        """
        for data_type in ["default", "limits"]:
            _from_simtel = self.parameter_dict.get(data_type)
            if self._should_skip_limits_check(data_type):
                continue

            _from_schema = self._get_schema_values(data_type)
            if isinstance(_from_schema, list):
                _from_schema = np.array(_from_schema, dtype=np.dtype(self.parameter_dict["type"]))

            if self._values_match(_from_simtel, _from_schema):
                self._logger.debug(f"Values for {data_type} match")
            else:
                self._log_mismatch_warning(data_type, _from_simtel, _from_schema)

    def read_simtel_config_file(self, simtel_config_file, simtel_telescope_name):
        """
        Read sim_telarray configuration file and return a dictionary with the parameter values.

        Parameters
        ----------
        simtel_config_file: str or Path
            Path of the file to read from.
        simtel_telescope_name: str
            Telescope name (sim_telarray convention)

        Returns
        -------
        dict
            Dictionary with the parameter values.

        """
        self._logger.debug(
            f"Reading simtel config file {simtel_config_file} for parameter {self.parameter_name}"
        )
        matching_lines = {}
        try:
            with open(simtel_config_file, encoding="utf-8") as file:
                for line in file:
                    # split line into parts (space, tabs, comma separated)
                    parts_of_lines = re.split(r",\s*|\s+", line.strip())
                    if self.simtel_parameter_name == parts_of_lines[1].upper():
                        matching_lines[parts_of_lines[0]] = parts_of_lines[2:]
        except FileNotFoundError as exc:
            self._logger.error(f"File {simtel_config_file} not found.")
            raise exc
        if len(matching_lines) == 0:
            self._logger.info(f"No entries found for parameter {self.simtel_parameter_name}")
            return None

        _para_dict = {}
        # first: extract line type (required for conversions and dimension)
        _para_dict["type"], _para_dict["dimension"] = self._get_type_and_dimension_from_simtel_cfg(
            matching_lines["type"]
        )
        # then: extract other fields
        # (order of keys matter; not all field are present for all parameters)
        for key in ["default", simtel_telescope_name, "limits"]:
            try:
                _para_dict[key], _ = self.extract_value_from_sim_telarray_column(
                    matching_lines[key],
                    dtype=_para_dict.get("type"),
                    n_dim=_para_dict.get("dimension"),
                    default=_para_dict.get("default"),
                    is_limit=(key == "limits"),
                )
            except KeyError:
                pass

        return _para_dict

    def _resolve_all_in_column(self, column):
        """
        Resolve 'all' entries in a column.

        This needs to resolve the following cases:
        no 'all' in any entry; ['all:', '5'], ['all: 5'], ['all:5', '3:1']
        This function is fine-tuned to the sim_telarray configuration output.

        Parameters
        ----------
        column: list
            List of strings to resolve.

        Returns
        -------
        list
            List of resolved strings.

        """
        # don't do anything if all string items in column do not start with 'all'
        if not any(isinstance(item, str) and item.startswith("all") for item in column):
            return column, {}

        self._logger.debug(f"Resolving 'all' entries in column: {column}")
        # remove 'all:' entries
        column = [item for item in column if item not in ("all:", "all")]
        # resolve 'all:5' type entries
        column = [
            item.split(":")[1].replace(" ", "") if item.startswith("all:") else item
            for item in column
        ]
        # find 'index:value' type entries
        except_from_all = {}
        for item in column:
            if ":" in item:
                index, value = item.split(":")
                except_from_all[index] = value
        # finally remove entries containing ':'
        column = [item for item in column if ":" not in item]

        return column, except_from_all

    def extract_value_from_sim_telarray_column(
        self, column, dtype=None, n_dim=1, default=None, is_limit=False
    ):
        """
        Extract value(s) from sim_telarray configuration file columns.

        This function is fine-tuned to the sim_telarray configuration output.

        Parameters
        ----------
        column: list
            List of strings to extract value from.
        dtype: str
            Data type to convert value to.
        n_dim: int
            Length of array to be returned.
        default: object
            Default value to extend array to required length.

        Returns
        -------
        object, int
            Values extracted from column. Of object is a list of array, return length of array.

        """
        # string represents a lists of values (space or comma separated)
        if len(column) == 1:
            column = column[0].split(",") if "," in column[0] else column[0].split(" ")

        column = [None if item.lower() == "none" else item for item in column]
        column, except_from_all = self._resolve_all_in_column(column)
        # extend array to required length (simtel uses sometimes 'all:' for all entries)
        if n_dim > 1 and len(column) < n_dim:
            try:
                # skip formatting: black reformats and violates E203
                column += default[len(column):]  # fmt: skip
            except TypeError:
                # extend array to required length using previous value
                column.extend([column[-1]] * (n_dim - len(column)))
        for index, value in except_from_all.items():
            column[int(index)] = value
        if dtype == "bool":
            column = np.array([bool(int(item)) for item in column])

        column, ndim = self._process_column(column, dtype)
        if not is_limit:
            column = self._add_units(column)
        return column, ndim

    def _add_units(self, column):
        """
        Add units as given in schema file to column.

        Take into account array types and dimensionless units.
        Ensure that integer values are returned as integers (astropy converts
        values to floats when multiplying them with units).

        """
        try:
            unit = self._get_schema_values("unit")
        except TypeError:  # no schema defined
            return column
        if unit is None or unit == "dimensionless":
            return column

        if isinstance(column, np.ndarray) and len(column) == len(unit):
            return np.array(
                [
                    col * u.Unit(un) if un != "dimensionless" else col
                    for col, un in zip(column, unit)
                ],
                dtype=object,
            )
        if isinstance(unit, str):
            column_with_unit = column * u.Unit(unit)
            if isinstance(column, int | np.integer):
                return u.Quantity(int(column_with_unit.value), unit, dtype=type(column))
            return column_with_unit

        return None

    def _process_column(self, column, dtype):
        """
        Process and return column prepared in extract_value_from_sim_telarray_column.

        Parameters
        ----------
        column: list
            List of strings to process.
        dtype: str
            Data type to convert value to.
        """
        if len(column) == 1:
            if column[0] is not None:
                array_dtype = np.dtype(dtype) if dtype else None
                processed_value = np.array(column, dtype=array_dtype)[0]
                return processed_value, 1
            return None, 1
        if len(column) > 1:
            return np.array(column, dtype=np.dtype(dtype) if dtype else None), len(column)
        return None, None

    def _get_type_and_dimension_from_simtel_cfg(self, column):
        """
        Return type and dimension from sim_telarray configuration column.

        'Func' type from simtel is treated as string. Return number
        of camera pixel for a hard-wired set up parameters.

        Parameters
        ----------
        column: list
            List of strings to extract value from.

        Returns
        -------
        str, int
            Type and dimension.

        """
        if column[0].lower() == "text" or column[0].lower() == "func":
            return "str", 1
        if column[0].lower() == "ibool":
            return "bool", int(column[1])
        if self.camera_pixels is not None and self.simtel_parameter_name in ["NIGHTSKY_BACKGROUND"]:
            return str(np.dtype(column[0].lower())), self.camera_pixels
        return str(np.dtype(column[0].lower())), int(column[1])
