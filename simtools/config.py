""" Module to deal with the interface with the global config information."""

import copy
import logging
import os
from pathlib import Path

import yaml

__all__ = ["set_config_file_name", "load_config", "get", "find_file", "change"]


class ConfigEnvironmentalVariableNotSet(Exception):
    pass


class ParameterNotFoundInConfigFile(Exception):
    pass


def set_config_file_name(file_name):
    """
    Redefines the config file name by resetting a global variable.

    Parameters
    ----------
    file_name: str
        Config file name.
    """
    if not file_name:
        return

    _logger = logging.getLogger(__name__)
    _logger.debug("Setting the config file name to {}".format(file_name))
    global CONFIG_FILE_NAME
    CONFIG_FILE_NAME = file_name


def load_config(file_name=None, use_globals=True):
    """
    Load config file and return it as a dict.
    3 possible options for the config file_name:
    1st - file_name parameter is given (not None)
    2nd - CONFIG_FILE_NAME exists (set by set_config_file_name)
    3rd - ./config.yml

    Parameters
    ----------
    file_name: str, optional
        Config file name.
    use_globals: bool
        Use global config settings

    Returns
    -------
    dict
        A dictionary containing all the info from the global configuration setting.
    """
    if file_name is not None:
        this_file_name = file_name
    elif "CONFIG_FILE_NAME" in globals():
        this_file_name = CONFIG_FILE_NAME
    else:
        this_file_name = "config.yml"

    with open(this_file_name, "r") as stream:
        config = yaml.safe_load(stream)

    # Running over the parameters set for change
    if use_globals and "CONFIG_CHANGED_PARS" in globals():
        for par, value in CONFIG_CHANGED_PARS.items():
            config[par] = value

    return config


def get(par, use_globals=True):
    """
    Get a single entry from the config settings.

    Parameters
    ----------
    par: str
        Name of the desired parameter.
    use_globals: bool (default=True)
        Use global config settings

    Raises
    ------
    ParameterNotFoundInConfigFile
        In case the parameter is not in config.

    Returns
    -------
    Value of the entry from the config settings.
    """
    _logger = logging.getLogger(__name__)

    config = load_config(use_globals=use_globals)
    if par not in config.keys():
        msg = "Configuration file does not contain an entry for the parameter " "{}".format(par)
        _logger.error(msg)
        raise ParameterNotFoundInConfigFile(msg)
    else:
        # Enviroment variable
        if isinstance(config[par], str) and "$" in config[par]:
            return os.path.expandvars(config[par])
        else:
            return config[par]


def change(par, value):
    """
    Set to change a parameter to another value.

    Parameters
    ----------
    par: str
        Name of the parameter to change.
    value: any
        Value to be set to the parameter.
    """
    if "CONFIG_CHANGED_PARS" not in globals():
        global CONFIG_CHANGED_PARS
        CONFIG_CHANGED_PARS = dict()
    CONFIG_CHANGED_PARS[par] = value


def get_config_arg(name, value):
    """
    Get a config parameter if value is None. To be used to receive input arguments in classes.

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
    """
    return value if value is not None else get(name)


def find_file(name, loc=None):
    """
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
    """
    _logger = logging.getLogger(__name__)

    if loc is None:
        all_locations = get(par="model_files_locations")
    else:
        all_locations = copy.copy(loc)
    all_locations = [all_locations] if not isinstance(all_locations, list) else all_locations

    def _search_directory(directory, filename, rec=False):
        if not Path(directory).exists():
            msg = "Directory {} does not exist".format(directory)
            _logger.debug(msg)
            return None

        f = Path(directory).joinpath(filename)
        if f.exists():
            _logger.debug("File {} found in {}".format(filename, directory))
            return f
        if not rec:  # Not recursively
            return None

        for subdir in Path(directory).iterdir():
            if not subdir.is_dir():
                continue
            f = _search_directory(subdir, filename, True)
            if f is not None:
                return f
        return None

    # Searching file locally
    ff = _search_directory(".", name)
    if ff is not None:
        return ff
    # Searching file in given locations
    for ll in all_locations:
        ff = _search_directory(ll, name, True)
        if ff is not None:
            return ff
    msg = "File {} could not be found in {}".format(name, all_locations)
    _logger.error(msg)
    raise FileNotFoundError(msg)


def create_dummy_config_file(filename="config.yml", **kwargs):
    """
    Create a dummy config.yml file to be used in test enviroments only.

    Parameters
    ----------
    filename: str
        Name of the dummy config file (default=config.yml)
    **kwargs
        The default parameters can be overwritten using kwargs.
    """
    config = {
        "use_mongo_db": False,
        "mongo_db_config_file": None,
        "data_location": "./data/",
        "model_files_locations": ".",
        "simtel_path": ".",
        "output_location": ".",
        "extra_commands": [],
    }

    # # Overwritting parameters with kwargs
    if len(kwargs) > 0:
        for key, value in kwargs.items():
            config[key] = value

    with open(filename, "w") as outfile:
        yaml.dump(config, outfile)


def create_dummy_db_details(filename="db_details.yml", **kwargs):
    """
    Create a dummy db_details.yml file to be used in test enviroments only.

    Parameters
    ----------
    filename: str
        Name of the dummy db_details file (default=db_details.yml)
    **kwargs
        The default parameters can be overwritten using kwargs.
    """
    pars = {
        "db_api_port": None,
        "db_server": None,
        "db_api_user": None,
        "db_api_pw": None,
        "db_api_authentication_database": "admin",
    }

    if len(kwargs) > 0:
        for key, value in kwargs.items():
            pars[key] = int(value) if key == "db_api_port" else str(value)

    with open(filename, "w") as outfile:
        yaml.dump(pars, outfile)
