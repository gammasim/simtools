<<<<<<< HEAD
''' Module to handle interaction with DB. '''
=======
''' Module to handle interactions with DB '''
>>>>>>> master

import logging
import datetime
import yaml
from pathlib import Path

__all__ = ['getArrayDB']


def getArrayDB(databaseLocation):
    '''
<<<<<<< HEAD
    Get array db as a dict.
=======
    Get array db info as a dict.
>>>>>>> master

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
