#!/usr/bin/python3

import logging
import math


__all__ = ['computeTelescopeTransmission', 'getTelescopeSize']


def computeTelescopeTransmission(pars, offAxis):
    ''' pars: parametric function of telescope transmission
        offAxis: in deg
    '''
    if pars[1] == 0:
        return pars[0]
    else:
        t = math.sin(offAxis * math.pi / 180.)/(pars[3] * math.pi / 180.)
        return pars[0] / (1. + pars[2] * t**pars[4])


def getTelescopeSize(telescopeType):
    '''
    Return the telescope size (SST, MST or LST) for a given telescopeType.

    Args:
        telescopeType (str): ex SST-2M-ASTRI, LST, ...

    Returns:
        str: 'SST', 'MST' or 'LST'

    '''

    if 'SST' in telescopeType:
        return 'SST'
    elif 'MST' in telescopeType:
        return 'MST'
    elif 'LST' in telescopeType:
        return 'LST'
    else:
        logger.error('Invalid telescopeType {}'.format(telescopeType))
        return None
