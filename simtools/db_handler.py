''' Module to handle interaction with DB. '''

import logging
import datetime
import yaml
from pathlib import Path

__all__ = ['getArrayDB']


def getArrayDB(databaseLocation):
    '''
    Get array db as a dict.

    Parameters
    ----------
    databaseLocation: str or Path

    Returns
    -------
    dict
    '''
    file = Path(databaseLocation).joinpath('arrays').joinpath('arrays.yml')
    out = dict()
    with open(file, 'r') as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)
    return out
