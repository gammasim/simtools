#!/usr/bin/python3

import json
import logging

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
from simtools.data_model import validate_data
from simtools.utils import names

__all__ = ["SimtelConfigReader"]


class JsonNumpyEncoder(json.JSONEncoder):
    """
    Convert numpy to python types as accepted by json.
    """

    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (u.core.CompositeUnit, u.core.IrreducibleUnit)):
            return str(o) if o != u.dimensionless_unscaled else None
        return super().default(o)


class SimtelConfigReader:
    """
    SimtelConfigReader reads sim_telarray configuration files and converts them to the simtools
    representation of a model parameter (json dict). The sim_telarray configuration can be
    generated using e.g., the following simtel_array command:

    ... code-block:: console

        sim_telarray/bin/sim_telarray \
            -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal\
            -C typelist=no-internal -C maximum_telescopes=30\
            -DNSB_AUTOSCALE -DNECTARCAM -DHYPER_LAYOUT\
            -DNUM_TELESCOPES=30 /dev/null 2>|/dev/null | grep '(@cfg)'

    Parameters
    ----------
    schema_dict: dict
        Schema dictionary describing the model parameter.
    simtel_config_file: str or Path
        Path of the file to read from.
    simtel_telescope_name: str
        Telescope name (sim_telarray convention)
    return_arrays_as_strings: bool
        If True, return arrays as comma separated strings.
    """

    def __init__(
        self,
        schema_file,
        simtel_config_file,
        simtel_telescope_name,
        return_arrays_as_strings=True,
    ):
        """
        Initialize SimtelConfigReader.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelConfigReader")

        self.schema_file = schema_file
        self.schema_dict = gen.collect_data_from_file_or_dict(
            file_name=self.schema_file, in_dict=None
        )
        self.parameter_name = self.schema_dict.get("name")
        self.simtel_parameter_name = self._get_simtel_parameter_name(self.parameter_name)
        self.simtel_telescope_name = simtel_telescope_name
        self.return_arrays_as_strings = return_arrays_as_strings
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
        self._logger.info(f"Getting validated parameter dictionary for {telescope_name}")

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

        self._logger.info(f"Exporting parameter dictionary to {file_name}")
        with open(file_name, "w", encoding="utf-8") as file:
            json.dump(
                dict_to_write,
                file,
                indent=4,
                sort_keys=False,
                cls=JsonNumpyEncoder,
            )
            file.write("\n")

    def compare_simtel_config_with_schema(self):
        """
        Compare limits and defaults reported by simtel_array with schema
        (for debugging purposes; simple printing).

        """

        print(
            f"Comparing simtel_array configuration with schema for {self.parameter_name}"
            f"(sim_telarray: {self.simtel_parameter_name})"
        )

        print("Limits:")
        print(f"  from simtel: {self.parameter_dict.get('limits')}")
        try:
            print(f"  from schema: {self.schema_dict['data'][0]['allowed_range']})")
        except (KeyError, IndexError):
            print("  from schema: None")
        print("Defaults:")
        print(f"  from simtel: {self.parameter_dict.get('default')}")
        try:
            print(f"  from schema: {self.schema_dict['data'][0]['default']}")
        except (KeyError, IndexError):
            print("  from schema: None")

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

        columns = []
        try:
            with open(simtel_config_file, "r", encoding="utf-8") as file:
                for line in file:
                    if self.simtel_parameter_name in line.upper():
                        columns.append(line.strip().split("\t"))
        except FileNotFoundError as exc:
            self._logger.error(f"File {simtel_config_file} not found.")
            raise exc

        _para_dict = {}
        # extract first column type (required for conversions and dimension)
        for column in columns:
            if column[0] == "type":
                _para_dict["type"], _para_dict["dimension"] = self._get_type_from_simtel_cfg(
                    column[2:]
                )
        # extract other fields
        for column in columns:
            if column[0] in [simtel_telescope_name, "default", "limits"]:
                _para_dict[column[0]], _ = self._add_value_from_simtel_cfg(
                    column[2:], _para_dict.get("type")
                )

        return _para_dict

    def _add_value_from_simtel_cfg(self, column, dtype=None):
        """
        Extract value(s) from simtel configuration file columns
        (this function needs to be fine tuned to those files).

        Parameters
        ----------
        column: list
            List of strings to extract value from.
        dtype: str
            Data type to convert value to.

        Returns
        -------
        object, int
            Values extracted from column. Of object is a list of array, return length of array.

        """
        # lists are space or comma separated
        if len(column) == 1:
            column = column[0].split(",") if "," in column[0] else column[0].split(" ")
            column = [item for item in column if item != "all:"]

        if len(column) == 1:
            return np.array(column, dtype=np.dtype(dtype) if dtype else None)[0], 1
        if len(column) > 1:
            if self.return_arrays_as_strings:
                return " ".join(column), len(column)
            return np.array(column, dtype=np.dtype(dtype) if dtype else None), len(column)
        return None, None

    def _get_type_from_simtel_cfg(self, column):
        """
        Return type and dimension from simtel configuration column.

        Parameters
        ----------
        column: list
            List of strings to extract value from.

        Returns
        -------
        str, int
            Type and dimension.

        """

        if column[0].lower() == "text":
            return "str", 1
        if column[0].lower() == "ibool":
            return "bool", int(column[1])
        if int(column[1]) > 1 and self.return_arrays_as_strings:
            return "str", 1
        # TODO - cannot handle arrays of different types
        # return np.dtype(column[0].lower()), int(column[1])
        return column[0].lower(), int(column[1])

    def _get_simtel_parameter_name(self, parameter_name):
        """
        Return parameter name as used in sim_telarray.

        Parameters
        ----------
        parameter_name: str
            Model parameter name (as used in simtools)

        Returns
        -------
        str
            Parameter name as used in sim_telarray.

        """

        self._logger.warning("TODO - convert simtools to simtel parameter")

        return parameter_name.upper()

    def _check_parameter_applicability(self, telescope_name):
        """
        Check if a parameter is applicable for a given telescope using
        the information available in the schema file.
        First check for exact telescope name, if not listed in the schema
        use telescope type.

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
        Return unit from schema dict.

        Returns
        -------
        str
            Parameter unit
        """
        try:
            return (
                self.schema_dict["data"][0]["unit"]
                if self.schema_dict["data"][0]["unit"] != "dimensionless"
                else None
            )
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

        self._logger.info(
            f"Validating parameter dictionary {parameter_dict} using {self.schema_file}"
        )
        data_validator = validate_data.DataValidator(
            schema_file=self.schema_file,
            data_dict=parameter_dict,
            check_exact_data_type=False,
        )
        data_validator.validate_and_transform()
        return data_validator.data_dict
