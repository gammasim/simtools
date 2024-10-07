#!/usr/bin/python3

import logging
import numpy as np
import simtools.utils.names as names


class ParamDicts:

    def __init__(
            self
            ):

        self._logger = logging.getLogger(__name__)
        #self.Desc = {}
        #self.shortDesc = {}
        #self.telType = {}



    #@cache
    def _get_paramDesc(self):
        '''
        A function to return the descriptions and short descriptions
        of parameters as dictionary values
        with the model parameter names as the keys.
        '''
        all_paramDicts = names.telescope_parameters()
        paramNames = all_paramDicts.keys()

        paramDesc = {}
        telType = {}
        shortDesc = {}

        for param in paramNames:

            paramDesc[param] =f'{all_paramDicts[param].get("description")}'
            shortDesc[param] = f'{all_paramDicts[param].get("short_description")}'
            

        return paramDesc, shortDesc



    def get_telescope_param_data(self, tel_model):
        '''
        A function to get all the relevant parameters associated with a given telescope.
        '''

        #telType = self._get_dicts()[2]
        #return [key for key, values in telType.items() if telescope in values]
        paramDicts = tel_model._parameters
        paramNames = paramDicts.keys()

        data = []
        for parameter in paramNames:
            value = paramDicts[parameter].get('value')
            unit = paramDicts[parameter].get('unit')
            desc = self._get_paramDesc()[0].get(parameter)
            shortDesc = self._get_paramDesc()[1].get(parameter)	   
            data.append([parameter, value, unit, desc, shortDesc])


        return data




