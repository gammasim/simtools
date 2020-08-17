import logging

from simtools.util import names


__all__ = ['getArrayInfo']

logger = logging.getLogger(__name__)


ARRAYS = {
    '1LST': {
        0: {'size': 'LST', 'xPos': 0, 'yPos': 0}
    },
    '4LST': {
        0: {'size': 'LST', 'xPos': 57.5, 'yPos': 57.5},
        1: {'size': 'LST', 'xPos': -57.5, 'yPos': 57.5},
        2: {'size': 'LST', 'xPos': 57.5, 'yPos': -57.5},
        3: {'size': 'LST', 'xPos': -57.5, 'yPos': -57.5}
    },
    '1MST': {
        0: {'size': 'MST', 'xPos': 0, 'yPos': 0}
    },
    '4MST': {
        0: {'size': 'MST', 'xPos': 70, 'yPos': 70},
        1: {'size': 'MST', 'xPos': -70, 'yPos': 70},
        2: {'size': 'MST', 'xPos': 70, 'yPos': -70},
        3: {'size': 'MST', 'xPos': -70, 'yPos': -70}
    },
    '1SST': {
        0: {'size': 'SST', 'xPos': 0, 'yPos': 0}
    },
    '4SST': {
        0: {'size': 'SST', 'xPos': 80, 'yPos': 80},
        1: {'size': 'SST', 'xPos': -80, 'yPos': 80},
        2: {'size': 'SST', 'xPos': 80, 'yPos': -80},
        3: {'size': 'SST', 'xPos': -80, 'yPos': -80}
    }
}


def getArrayInfo(arrayName):
    '''
    Return a dict with the array info (telescope sizes and positions) for a given array name.

    Parameters
    ----------
    arrayName: str
        Name of the array (e.g. 4LST, 1MST, ...)

    Returns
    -------
    dict
        Telescope sizes and positions.
    '''
    arrayName = names.validateArrayName(arrayName)
    return ARRAYS[arrayName]
