''' Module to deal with the interface with the global config information.'''

from pathlib import Path
import logging
import yaml

__all__ = ['loadConfig', 'get', 'findFile']

logger = logging.getLogger(__name__)


def setConfigFileName(fileName):
    '''
    Redefines the config file name by resetting the a global variable.

    Parameters
    ----------
    fileName: str
        Config file name.
    '''
    logger.debug('Setting the config file name to {}'.format(fileName))
    global CONFIG_FILE_NAME
    CONFIG_FILE_NAME = fileName


def loadConfig(fileName=None):
    '''
    Load config file and return it as a dict.
    3 possible options for the config fileName:
    1st - fileName parameter is given (not None)
    2nd - CONFIG_FILE_NAME exists (set by setConfigFileName)
    3rd - ./config.yml

    Parameters
    ----------
    fileName: str, optional
        Config file name.

    Returns
    -------
    dict
        A dictionaty containing all the info from the global configuration setting.
    '''
    if fileName is not None:
        thisFileName = fileName
    elif 'CONFIG_FILE_NAME' in globals():
        thisFileName = CONFIG_FILE_NAME
    else:
        thisFileName = 'config.yml'

    with open(thisFileName, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def get(par):
    '''
    Get a single entry from the config settings.

    Parameters
    ----------
    par: str
        Name of the desired parameter.

    Returns
    -------
    Value of the entry from the config settings.
    '''
    config = loadConfig()
    if par not in config.keys():
        logger.error('Config does not contain {}'.format(par))
        raise KeyError()
    else:
        return config[par]


def collectConfigArg(name, value):
    '''
    Collect a config parameter if value is None. To be used to receive input arguments in classes.

    Warning
    -------
    Bad function naming - please redo it.

    Parameters
    ----------
    name: str
        Name of the parameter
    value: str
        Input value.

    Returns
    -------
    Path
        Path of the desired parameter.
    '''
    return Path(value) if value is not None else get(name)


def findFile(name, loc=None):
    '''
    Search for model files inside of given directories, recursively, and return its full path.

    Parameters
    ----------
    name: str
        File name to be searched for.
    loc: Path, optional
        Location of where to search for the file. If not given, config information will be used.

    Returns
    -------
    Full path of the file to be found if existing. Otherwise, None

    Raises
    ------
    FileNotFoundError
        If the desired file is not found.
    '''
    if loc is None:
        loc = get(par='modelFilesLocations')
    loc = [loc] if not isinstance(loc, list) else loc

    def _searchDirectory(directory, filename, rec=False):
        logger.debug('Searching directory {}'.format(directory))
        if not Path(directory).exists():
            msg = 'Directory {} does not exist'.format(directory)
            logger.warning(msg)
            return None

        f = Path(directory).joinpath(filename)
        if f.exists():
            logger.debug('File {} found in {}'.format(filename, directory))
            return f
        if not rec:  # Not recursively
            return None

        for subdir in Path(directory).iterdir():
            if not subdir.is_dir():
                continue
            f = _searchDirectory(subdir, filename, True)
            if f is not None:
                return f
        return None

    # Searching file locally
    ff = _searchDirectory('.', name)
    if ff is not None:
        return ff
    # Searching file in given locations
    for ll in loc:
        ff = _searchDirectory(ll, name, True)
        if ff is not None:
            return ff
    msg = 'File {} could not be found'.format(name)
    logger.error(msg)
    raise FileNotFoundError(msg)
