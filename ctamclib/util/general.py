#!/usr/bin/python3

import logging


__all__ = ['configParameters']


def collectArguments(obj, parameters, **kwargs):
    for par in parameters:
        inPar = '_' + par
        if par in kwargs.keys():
            obj.__dict__[inPar] = kwargs[par]
            logging.info('Setting {}={}'.format(par, kwargs[par]))
        elif inPar in obj.__dict__:
            logging.info('Setting {}={} (default)'.format(par, obj.__dict__[inPar]))
        else:
            logging.error('Parameter {} has to be given'.format(par))
            # raise
