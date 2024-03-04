#!/usr/bin/python3

import logging

import numpy as np

import simtools.utils.general as gen

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

        self.schema_dict = self._read_schema_file(parameter_name, schema_url)
        self.simtel_parameter_name = self._get_simtel_parameter_name(parameter_name)
        self.parameter_dict = self._read_simtel_config_file(
            simtel_config_file, simtel_telescope_name
        )

    def get_validated_parameter_dict(self, telescope_name):
        """
        Return a dictionary the parameter. The values are validated and in the format
        expected by the model data base.

        Parameters
        ----------
        telescope_name: str
            Telescope name (e.g., LSTN-01)

        Returns
        -------
        dict
            Parameter dictionary.

        """
        self._logger.info(f"Validating parameter dictionary for {telescope_name}")

        return self.parameter_dict

    def export_parameter_dict(self, file_name):
        """
        Export parameter dictionary to json.

        Parameters
        ----------
        file_name: str or Path
            File name to export to.

        """

        self._logger.info(f"Exporting parameter dictionary to {file_name}")

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
        # extract first column type
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

    def _read_schema_file(self, parameter_name, schema_url):
        """
        Read schema file and return a dictionary with the schema.

        Parameters
        ----------
        parameter_name: str
            Model parameter name for which to read the schema.
        schema_url: str
            URL of schema file directory

        Returns
        -------
        dict
            Dictionary with the schema.

        """

        return gen.collect_data_from_file_or_dict(
            file_name=f"{schema_url}/{parameter_name}.schema.yml", in_dict=None
        )

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

        return parameter_name.upper()
