#!/usr/bin/python3

import logging
import math


__all__ = ['computeTelescopeTransmission']


def computeTelescopeTransmission(pars, offAxis):
    ''' pars: parametric function of telescope transmission
        offAxis: in deg
    '''
    if pars[1] == 0:
        return pars[0]
    else:
        t = math.sin(offAxis * math.pi / 180.)/(pars[3] * math.pi / 180.)
        return pars[0] / (1. + pars[2] * t**pars[4])
