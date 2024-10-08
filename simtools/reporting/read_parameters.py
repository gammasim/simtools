#!/usr/bin/python3

import logging
import simtools.utils.names as names


class ReadParameters:

    def __init__(
            self
            ):

        self._logger = logging.getLogger(__name__)


    def _get_parameter_descriptions(self):
        '''
        A function to return the descriptions and short descriptions
        of all parameters as dictionary values
        with the model parameter names as the keys.
        '''
        all_parameter_dictionaries = names.telescope_parameters()
        all_parameter_names = all_parameter_dictionaries.keys()

        parameter_description = {}
        short_description = {}

        for parameter in all_parameter_names:

            parameter_description[parameter] =f'{all_parameter_dictionaries[parameter].get("description")}'
            short_description[parameter] = f'{all_parameter_dictionaries[parameter].get("short_description")}'
            

        return parameter_description, short_description



    def get_telescope_parameter_data(self, telescope_model):
        '''
        A function to get the name, value, unit, and descriptions
        of all the relevant parameters associated with a given telescope.
        '''

        parameter_dictionaries = telescope_model._parameters
        parameter_names = parameter_dictionaries.keys()

        data = []
        for parameter in parameter_names:
            value = parameter_dictionaries[parameter].get('value')
            unit = parameter_dictionaries[parameter].get('unit')
            description = self._get_parameter_descriptions()[0].get(parameter)
            short_description = self._get_parameter_descriptions()[1].get(parameter)	   
            data.append([parameter, value, unit, description, short_description])


        return data




