#!/usr/bin/python3
"""SimtelConfigRead to read sim_telarray configuration output."""

import logging
import re

import numpy as np

import simtools.utils.general as gen
from simtools.data_model import validate_data
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.utils import names

__all__ = ["SimtelConfigReader"]


class SimtelConfigReader:
    """
    SimtelConfigReader reads model parameters and converts to the simtools representation.

    Model parameters are read from sim_telarray configuration files.
    The sim_telarray configuration can be generated using e.g., the following simtel_array command:

    ... code-block:: console

        sim_telarray/bin/sim_telarray \
            -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal\
            -C typelist=no-internal -C maximum_telescopes=30\
            -DNSB_AUTOSCALE -DNECTARCAM -DHYPER_LAYOUT\
            -DNUM_TELESCOPES=30 /dev/null 2>|/dev/null | grep '(@cfg)'

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
        schema_file,
        simtel_config_file,
        simtel_telescope_name,
        parameter_name=None,
        camera_pixels=None,
    ):
        """Initialize SimtelConfigReader."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelConfigReader")

        self.schema_file = schema_file
        self.schema_dict = (
            gen.collect_data_from_file_or_dict(file_name=self.schema_file, in_dict=None)
            if self.schema_file is not None
            else None
        )
        self.parameter_name = self.schema_dict.get("name") if self.schema_dict else parameter_name
        self.simtel_parameter_name = self._get_simtel_parameter_name(self.parameter_name)
        self.simtel_telescope_name = simtel_telescope_name
        self.camera_pixels = camera_pixels
        self.parameter_dict = self._read_simtel_config_file(
            simtel_config_file, simtel_telescope_name
        )

    def get_validated_parameter_dict(self, telescope_name, model_version=None):
        """
        Return a validated model parameter dictionary as filled into the database.

        Parameters
        ----------
        telescope_name: str
            Telescope name (e.g., LSTN-01)
        model_version: str
            Model version string.

        Returns
        -------
        dict
            Model parameter dictionary.

        """
        self._logger.debug(f"Getting validated parameter dictionary for {telescope_name}")

        _json_dict = {
            "parameter": self.parameter_name,
            "instrument": telescope_name,
            "site": names.get_site_from_telescope_name(telescope_name),
            "version": model_version,
            "value": self.parameter_dict.get(self.simtel_telescope_name),
            "unit": self._get_unit_from_schema(),
            "type": (
                "string"
                if self.parameter_dict.get("type") == "str"
                else (
                    "boolean"
                    if self.parameter_dict.get("type") == "bool"
                    else self.parameter_dict.get("type")
                )
            ),
            "applicable": self._check_parameter_applicability(telescope_name),
            "file": self._parameter_is_a_file(),
        }
        return self._validate_parameter_dict(_json_dict)

    def export_parameter_dict_to_json(self, file_name, dict_to_write):
        """
        Export parameter dictionary to json.

        Parameters
        ----------
        file_name: str or Path
            File name to export to.
        dict_to_write: dict
            Dictionary to export.

        """
        try:
            dict_to_write["value"] = gen.convert_list_to_string(dict_to_write["value"])
            dict_to_write["unit"] = gen.convert_list_to_string(dict_to_write["unit"], True)
            dict_to_write["limits"] = gen.convert_list_to_string(dict_to_write["limits"])
        except KeyError:
            pass

        self._logger.info(f"Exporting parameter dictionary to {file_name}")
        ModelDataWriter.write_dict_to_model_parameter_json(
            file_name=file_name, data_dict=dict_to_write
        )

    def compare_simtel_config_with_schema(self):
        """
        Compare limits and defaults reported by simtel_array with schema.

        This is mostly for debugging purposes snd for simple printing.
        Check for differences in 'default' and 'limits' entries.

        """
        for data_type in ["default", "limits"]:
            _from_simtel = self.parameter_dict.get(data_type)
            # ignore limits checks for boolean
            if data_type == "limits" and self.parameter_dict.get("type") == "bool":
                continue
            try:
                if data_type == "limits":
                    _from_schema = [
                        self.schema_dict["data"][0]["allowed_range"].get("min"),
                        self.schema_dict["data"][0]["allowed_range"].get("max"),
                    ]
                    _from_schema = _from_schema[0] if _from_schema[1] is None else _from_schema
                else:
                    if len(self.schema_dict["data"]) == 1:
                        _from_schema = self.schema_dict["data"][0]["default"]
                    else:
                        _from_schema = [data.get("default") for data in self.schema_dict["data"]]
            except (KeyError, IndexError):
                _from_schema = None

            if isinstance(_from_schema, list):
                _from_schema = np.array(_from_schema, dtype=np.dtype(self.parameter_dict["type"]))

            try:
                if not isinstance(_from_schema, list | np.ndarray) and _from_simtel == _from_schema:
                    self._logger.debug(f"Values for {data_type} match")
                    continue
            except ValueError:
                pass
            try:
                if np.all(np.isclose(_from_simtel, _from_schema)):
                    self._logger.debug(f"Values for {data_type} match")
                    continue
            except (TypeError, ValueError):
                pass
            self._logger.warning(f"Values for {data_type} do not match:")
            self._logger.warning(
                f"  from simtel: {self.simtel_parameter_name} {_from_simtel}"
                f" ({type(_from_simtel)})"
            )
            self._logger.warning(
                f"  from schema: {self.parameter_name} {_from_schema}" f" ({type(_from_schema)})"
            )

    def _read_simtel_config_file(self, simtel_config_file, simtel_telescope_name):
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
            f"Reading simtel config file {simtel_config_file} "
            f"for parameter {self.parameter_name}"
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
                _para_dict[key], _ = self._add_value_from_simtel_cfg(
                    matching_lines[key],
                    dtype=_para_dict.get("type"),
                    n_dim=_para_dict.get("dimension"),
                    default=_para_dict.get("default"),
                )
            except KeyError:
                pass

        return _para_dict

    def _resolve_all_in_column(self, column):
        """
        Resolve 'all' entries in a column.

        This needs to resolve the following cases:
        no 'all' in any entry; ['all:', '5'], ['all: 5'], ['all:5', '3:1']
        This function is fine-tuned to the simtel configuration output.

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

    def _add_value_from_simtel_cfg(self, column, dtype=None, n_dim=1, default=None):
        """
        Extract value(s) from simtel configuration file columns.

        This function is fine-tuned to the simtel configuration output.

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
        self._logger.debug(
            f"Adding value from simtel config: {column} (n_dim={n_dim}, default={default})"
        )
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
        if len(except_from_all) > 0:
            for index, value in except_from_all.items():
                column[int(index)] = value
        if dtype == "bool":
            column = np.array([bool(int(item)) for item in column])

        if len(column) == 1:
            return (
                np.array(column, dtype=np.dtype(dtype) if dtype else None)[0]
                if column[0] is not None
                else None
            ), 1
        if len(column) > 1:
            return np.array(column, dtype=np.dtype(dtype) if dtype else None), len(column)
        return None, None

    def _get_type_and_dimension_from_simtel_cfg(self, column):
        """
        Return type and dimension from simtel configuration column.

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

    def _get_simtel_parameter_name(self, parameter_name):
        """
        Return parameter name as used in sim_telarray.

        This is documented in the schema file.

        Parameters
        ----------
        parameter_name: str
            Model parameter name (as used in simtools)

        Returns
        -------
        str
            Parameter name as used in sim_telarray.

        """
        try:
            for sim_soft in self.schema_dict["simulation_software"]:
                if sim_soft["name"] == "sim_telarray":
                    return sim_soft["internal_parameter_name"].upper()
        except (KeyError, TypeError):
            pass

        return parameter_name.upper()

    def _check_parameter_applicability(self, telescope_name):
        """
        Check if a parameter is applicable for a given telescope using the the schema file.

        First check for exact telescope name, if not listed in the schema use telescope type.

        Parameters
        ----------
        telescope_name: str
            Telescope name (e.g., LSTN-01)

        Returns
        -------
        bool
            True if parameter is applicable to telescope.

        """
        try:
            if telescope_name in self.schema_dict["instrument"]["type"]:
                return True
        except KeyError as exc:
            self._logger.error("Schema file does not contain 'instrument:type' key.")
            raise exc

        return (
            names.get_telescope_type_from_telescope_name(telescope_name)
            in self.schema_dict["instrument"]["type"]
        )

    def _parameter_is_a_file(self):
        """
        Check if parameter is a file.

        Returns
        -------
        bool
            True if parameter is a file.

        """
        try:
            return self.schema_dict["data"][0]["type"] == "file"
        except (KeyError, IndexError):
            pass
        return False

    def _get_unit_from_schema(self):
        """
        Return unit(s) from schema dict.

        Returns
        -------
        str or list
            Parameter unit(s)
        """
        try:
            unit_list = []
            for data in self.schema_dict["data"]:
                unit_list.append(data["unit"] if data["unit"] != "dimensionless" else None)
            return unit_list if len(unit_list) > 1 else unit_list[0]
        except (KeyError, IndexError):
            pass
        return None

    def _validate_parameter_dict(self, parameter_dict):
        """
        Validate json dictionary against model parameter data schema.

        Parameters
        ----------
        parameter_dict: dict
            Dictionary to validate.

        Returns
        -------
        dict
            Validated dictionary (possibly converted to reference units).

        """
        self._logger.debug(
            f"Validating parameter dictionary {parameter_dict} using {self.schema_file}"
        )
        data_validator = validate_data.DataValidator(
            schema_file=self.schema_file,
            data_dict=parameter_dict,
            check_exact_data_type=False,
        )
        data_validator.validate_and_transform()
        return data_validator.data_dict
