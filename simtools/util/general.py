#!/usr/bin/python3

import logging
import copy


__all__ = ['collectArguments', 'collectKwargs', 'setDefaultKwargs', 'sortArrays']


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


def collectKwargs(label, inKwargs):
    outKwargs = dict()
    for key, value in inKwargs.items():
        if label + '_' in key:
            outKwargs[key.replace(label + '_', '')] = value
    return outKwargs


def setDefaultKwargs(inKwargs, **kwargs):
    for par, value in kwargs.items():
        if par not in inKwargs.keys():
            inKwargs[par] = value
    return inKwargs


def sortArrays(*args):
    orderArray = copy.copy(args[0])
    newArgs = list()
    for arg in args:
        _, a = zip(*sorted(zip(orderArray, arg)))
        newArgs.append(list(a))
    return newArgs
