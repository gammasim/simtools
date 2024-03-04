#!/usr/bin/python3

import logging

import numpy as np

import simtools.utils.general as gen
from simtools.utils import names

__all__ = ["SimtelConfigReader"]


class SimtelConfigReader:
    """
    SimtelConfigReader reads sim_telarray configuration files generated using e.g., the
    following simtel_array command:

    ... code-block:: console

        sim_telarray/bin/sim_telarray \
            -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal\
            -C typelist=no-internal -C maximum_telescopes=30\
            -DNSB_AUTOSCALE -DNECTARCAM -DHYPER_LAYOUT\
            -DNUM_TELESCOPES=30 /dev/null 2>|/dev/null | grep '(@cfg)'

    Parameters
    ----------
    simtel_config_file: str or Path
        Path of the file to read from.
    simtel_telescope_name: str
        Telescope name (sim_telarray convention)
    parameter_name: str
        Model parameter name (as used in simtools)
    schema_url: str
        URL of schema file directory

    """

    def __init__(self, simtel_config_file, simtel_telescope_name, parameter_name, schema_url):
        """
        Initialize SimtelConfigReader.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelConfigReader")

        self.schema_dict = gen.collect_data_from_file_or_dict(
            file_name=f"{schema_url}/{parameter_name}.schema.yml", in_dict=None
        )
        self.parameter_name = parameter_name
        self.simtel_parameter_name = self._get_simtel_parameter_name(self.parameter_name)
        self.simtel_telescope_name = simtel_telescope_name
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

        Returns
        -------
        dict
            Model parameter dictionary.

        """
        self._logger.info(f"Validating parameter dictionary for {telescope_name}")

        _value, _type = self._get_value_and_type()

        _json_dict = {
            "parameter": self.parameter_name,
            "instrument": telescope_name,
            "site": names.get_site_from_telescope_name(telescope_name),
            "version": model_version,
            "value": _value,
            "unit": self.schema_dict.get("unit"),
            "type": _type,
            "applicable": self._check_applicability(telescope_name),
            "file": "TOO READ FROM SCHEMA",
        }

        self._logger.warning("TODO - add json dict validation")

        return self.parameter_dict, _json_dict

    def export_parameter_dict(self, file_name):
        """
        Export parameter dictionary to json.

        Parameters
        ----------
        file_name: str or Path
            File name to export to.

        """

        self._logger.info(f"Exporting parameter dictionary to {file_name}")
        self._logger.warning("TODO - write dict to json file")

    def _get_value_and_type(self):
        """
        Get value and type in the format expected by the database.
        Reduces to a single value if the array has only one element.

        Returns
        -------
        object, type
            Value and type of the parameter.

        """

        _type = self.parameter_dict.get("type")
        _value = self.parameter_dict.get(self.simtel_telescope_name)
        if _value is None:
            _value = self.parameter_dict.get("default")

        if isinstance(_value, np.ndarray) and len(_value) == 1:
            _value = _value[0]
            _type = _type[0] if isinstance(_type, np.ndarray) else _type

        return _value, _type

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

        self._logger.warning("TODO - determine if 'default' is needed.")
        _extraction_keys = [simtel_telescope_name, "default"]

        columns = []
        try:
            with open(simtel_config_file, "r", encoding="utf-8") as file:
                for line in file:
                    if self.simtel_parameter_name not in line:
                        continue
                    columns.append(line.strip().split("\t"))
        except FileNotFoundError as exc:
            self._logger.error(f"File {simtel_config_file} not found.")
            raise exc

        _para_dict = {}
        # extract first column type (required for conversions and dimension)
        for column in columns:
            if column[0] == "type":
                _para_dict["type"], _para_dict["dimension"] = self._add_value(column[2:], "type")
        # extract other fields
        for column in columns:
            if column[0] in _extraction_keys:
                _para_dict[column[0]], _ = self._add_value(
                    column[2:], column[0], _para_dict.get("type")
                )

        return _para_dict

    def _add_value(self, column, key, dtype=None):
        """
        Extract value(s) from columns depending on key.
        This function is fine tuned to the output of sim_telarray configuration files.

        Parameters
        ----------
        column: list
            List of strings to extract value from.
        key: str
            Key (type) to extract value for (e.g., 'type', 'default')
        dtype: type, optional
            Data type to convert value to.

        Returns
        -------
        object, int
            Values extracted from column. Of object is a list of array, return length of array.

        """
        if key == "type":
            if column[0].lower() == "text":
                return str, 1
            return np.dtype(column[0].lower()), int(column[1])
        # defaults are comma separated (all other by spaces)
        if key == "default" and len(column) == 1:
            column = column[0].split(",")
        else:
            column = column[0].split(" ")

        if len(column) > 0:
            if dtype is not None:
                return np.array(column, dtype=dtype), len(column)
            return np.array(column), len(column)
        return None, None

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

    def _check_applicability(self, telescope_name):
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
