#!/usr/bin/python3

import logging
import copy

import astropy.units as u

__all__ = ['collectArguments', 'collectKwargs', 'setDefaultKwargs', 'sortArrays']


class ArgumentWithWrongUnit(Exception):
    pass


class ArgumentCannotBeCollected(Exception):
    pass


def unitsAreConvertible(quantity_1, quantity_2):
    try:
        quantity_1.to(quantity_2)
        return True
    except:
        return False


def unitIsValid(quantity, unit):
    if unit is None and not isinstance(1 * quantity, u.quantity.Quantity):
        return True
    else:
        return unitsAreConvertible(quantity, unit)


def convertUnit(quantity, unit):
    return quantity if unit is None else quantity.to(unit).value


def collectArguments(obj, args, allInputs, **kwargs):

    def processSingleArg(arg, inArgName, argG, argD):
        if unitIsValid(argG, argD['unit']):
            obj.__dict__[inArgName] = convertUnit(argG, argD['unit'])
        else:
            logging.error('Argument {} given with wrong unit'.format(arg))
            raise ArgumentWithWrongUnit()

    def processListArg(arg, inArgName, argG, argD):
        outArg = list()
        try:
            argG = list(argG)
        except:
            argG = [argG]

        for aa in argG:
            if unitIsValid(aa, argD['unit']):
                outArg.append(convertUnit(aa, argD['unit']))
            else:
                logging.error('Argument {} given with wrong unit'.format(arg))
                raise ArgumentWithWrongUnit()
        obj.__dict__[inArgName] = outArg

    for arg in args:
        inArgName = '_' + arg
        argData = allInputs[arg]

        if arg not in allInputs.keys():
            msg = 'Arg {} cannot be collected because it is not in allInputs'.format(arg)
            logging.error(msg)
            raise ArgumentCannotBeCollected(msg)

        if arg in kwargs.keys():
            argGiven = kwargs[arg]
            # List
            if 'isList' in argData and argData['isList']:
                processListArg(arg, inArgName, argGiven, argData)
            else:  # Not a list
                processSingleArg(arg, inArgName, argGiven, argData)

        elif 'default' in argData:
            obj.__dict__[inArgName] = argData['default']
        else:
            logging.warning('Argument (without default) {} was not given'.format(arg))
    return


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
