#!/usr/bin/python3

""" This module contains the ArrayModel class

    Todo:

"""

import logging
import yaml
from pathlib import Path

from simtools.util import names
from simtools import db_handler

__all__ = ['getArray', 'ArrayModel']


def getArray(arrayName, databaseLocation):
    """ Return the telescope size (SST, MST or LST) for a given telescopeType.

    Args:
        arrayName (str):

        databaseLocation (Path):

    Returns:
        dict:

    """

    arrayName = names.validateArrayName(arrayName)
    db = db_handler.DatabaseHandler(logger.name)
    allArrays = db.getArrayDB(databaseLocation)
    return allArrays[arrayName]


class ArrayModel:
    pass
