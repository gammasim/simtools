#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging

from simtools.model.model_parameter import InvalidModelParameterError
from simtools.utils import names


class ReadParameters:
    """Read and manage model parameter data."""

    def __init__(self, telescope_model):
        """Initialise class with a telescope model."""
        self._logger = logging.getLogger(__name__)
        self.telescope_model = telescope_model

    def get_all_parameter_descriptions(self):
        """
        Get descriptions of all available parameters.

        Returns
        -------
            tuple: A tuple containing two dictionaries:
                - parameter_description: Maps parameter names to their descriptions.
                - short_description: Maps parameter names to their short descriptions.
        """
        all_parameter_dictionaries = names.telescope_parameters()
        all_parameter_names = all_parameter_dictionaries.keys()

        parameter_description = {}
        short_description = {}

        for parameter in all_parameter_names:

            parameter_description[parameter] = (
                f'{all_parameter_dictionaries[parameter].get("description")}'
            )
            short_description[parameter] = (
                f'{all_parameter_dictionaries[parameter].get("short_description")}'
            )

        return parameter_description, short_description

    def get_telescope_parameter_data(self):
        """
        Get model parameter data for a given telescope.

        Returns
        -------
            list: A list of lists containing parameter names, values with units,
                  descriptions, and short descriptions.
        """
        parameter_descriptions = self.get_all_parameter_descriptions()
        parameter_names = parameter_descriptions[0].keys()

        data = []
        for parameter in parameter_names:

            try:
                value = self.telescope_model.get_parameter_value_with_unit(parameter)
            except InvalidModelParameterError:  # if parameter not in telescope model
                self._logger.debug("{exc} encountered.")
                continue

            description = parameter_descriptions[0].get(parameter)
            short_description = parameter_descriptions[1].get(parameter)
            data.append([parameter, value, description, short_description])

        return data
